import sys
sys.path.append(".")

from mimo.configs.paths import ANIMATE_ANYONE_REPO, ML_RUNS, TRAIN_OUTPUTS, BASE_MODEL_FOLDER, ANIMATE_ANYONE_FOLDER
sys.path.append(ANIMATE_ANYONE_REPO)

import os, logging, time, random
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from safetensors.torch import load_file

from omegaconf import OmegaConf
import bitsandbytes as bnb
import mlflow

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.logging import get_logger as get_accelerate_logger

from transformers.utils import logging as transformers_logging

from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import logging as diffusers_logging
from diffusers.utils.import_utils import is_xformers_available

from src.utils.util import delete_additional_ckpt
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.resnet import InflatedConv3d

from mimo.utils.apose_ref_utils import download_base_model, download_anyone
from mimo.utils.torch_utils import seed_everything
from mimo.utils.general_utils import assert_file_exist

from mimo.training.training_utils import get_torch_weight_dtype, compute_snr, save_checkpoint, get_last_checkpoint, freeze_top_layer_reference_unet, unfreeze_motion_module
from mimo.training.training_dataset import TrainingDataset, collate_fn
from mimo.training.models import Net

class TrainingPipeline:
    def __init__(self, training_config):
        self.cfg = OmegaConf.load(assert_file_exist(training_config))
        self.infer_cfg = OmegaConf.load("./mimo/configs/inference/inference.yaml")

        self.config_seed()
        self.best_total_val_loss = np.infty
        
        self.weight_dtype = get_torch_weight_dtype(self.cfg.weight_dtype)
        self.save_dir = f"{TRAIN_OUTPUTS}/{self.cfg.exp_name}"
        self.save_dir_val = f"{TRAIN_OUTPUTS}/{self.cfg.exp_name}/val"

        self.accelerator = self.get_accelerator()
        self.logger = self.get_logger(self.accelerator)
        self.model, self.reference_control_writer, self.reference_control_reader = self.get_model()
        self.trainable_params = self.get_trainable_params(self.model)
        self.learning_rate = self.get_learning_rate(self.accelerator)
        self.train_noise_scheduler, self.val_noise_scheduler = self.get_noise_scheduler()
        self.optimizer = self.get_optimizer(self.trainable_params, self.learning_rate)
        self.lr_scheduler = self.get_lr_scheduler(self.optimizer)
        self.train_dataloader, self.train_dataset, self.val_dataloader, self.val_dataset = self.get_dataloaders(self.accelerator)

        self.validate_steps = (len(self.val_dataset)+self.accelerator.num_processes)//self.accelerator.num_processes

    def config_seed(self):
        if self.cfg.seed is not None:
            seed_everything(self.cfg.seed)

    def get_model(self):
        download_base_model()
        download_anyone()

        reference_unet = UNet2DConditionModel.from_pretrained(
            os.path.join(BASE_MODEL_FOLDER, "unet"),
            use_safetensors=True
        )
        reference_unet.load_state_dict(
            torch.load(assert_file_exist(ANIMATE_ANYONE_FOLDER, "reference_unet.pth"), map_location="cpu", weights_only=True),
        )
        
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            os.path.join(BASE_MODEL_FOLDER, "unet"),
            assert_file_exist(ANIMATE_ANYONE_FOLDER, "motion_module.pth"),
            unet_additional_kwargs=self.infer_cfg.unet_additional_kwargs,
        )
        denoising_unet.load_state_dict(
            torch.load(assert_file_exist(ANIMATE_ANYONE_FOLDER, "denoising_unet.pth"), map_location="cpu", weights_only=True),
            strict=False,
        )

        original_weights = denoising_unet.conv_in.weight.data  # Shape: (out_channels, in_channels, D, H, W)
        denoising_unet.conv_in = InflatedConv3d(
            3 * denoising_unet.conv_in.in_channels,  # multiply input channel by 3
            denoising_unet.conv_in.out_channels, 
            kernel_size=3, 
            padding=(1, 1)
        )
        denoising_unet.conv_in.weight = torch.nn.Parameter(torch.cat([original_weights] * 3, dim=1))

        pose_guider = PoseGuider(conditioning_embedding_channels=320)
        pose_guider.load_state_dict(
            torch.load(assert_file_exist(ANIMATE_ANYONE_FOLDER, "pose_guider.pth"), map_location="cpu", weights_only=True),
        )

        ## Freeze
        reference_unet = freeze_top_layer_reference_unet(reference_unet)
        if self.cfg.solver.freeze_referenceNet:
            reference_unet.requires_grad_(False)
        if self.cfg.solver.freeze_denoisingUnet:
            denoising_unet.requires_grad_(False)
        if self.cfg.solver.freeze_poseGuider:
            pose_guider.requires_grad_(False)
        if not self.cfg.solver.freeze_motionModule:
            denoising_unet = unfreeze_motion_module(denoising_unet)
        
        reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )
        
        if self.cfg.solver.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )
        
        if self.cfg.solver.gradient_checkpointing_referenceNet:
            reference_unet.enable_gradient_checkpointing()
        if self.cfg.solver.gradient_checkpointing_denoisingUnet:
            denoising_unet.enable_gradient_checkpointing()
        
        model = Net(
            reference_unet,
            denoising_unet,
            pose_guider,
            reference_control_writer,
            reference_control_reader,
        )
        
        return model, reference_control_writer, reference_control_reader
    
    def get_learning_rate(self, accelerator):
        if self.cfg.solver.scale_lr:
            return (
                self.cfg.solver.learning_rate
                * self.cfg.solver.gradient_accumulation_steps
                * self.cfg.data.train_batch_size
                * accelerator.num_processes
            )
        else:
            return self.cfg.solver.learning_rate

    def get_noise_scheduler(self):
        sched_kwargs = OmegaConf.to_container(self.cfg.noise_scheduler_kwargs)
        if self.cfg.enable_zero_snr:
            sched_kwargs.update(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        val_noise_scheduler = DDIMScheduler(**sched_kwargs)
        sched_kwargs.update({"beta_schedule": "scaled_linear"})
        train_noise_scheduler = DDIMScheduler(**sched_kwargs)

        return train_noise_scheduler, val_noise_scheduler
    
    def get_accelerator(self):
        return Accelerator(
            gradient_accumulation_steps=self.cfg.solver.gradient_accumulation_steps,
            mixed_precision=self.cfg.solver.mixed_precision,
            log_with="mlflow",
            project_dir=ML_RUNS,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=(not self.cfg.solver.freeze_referenceNet))],
        )

    def get_logger(self, accelerator):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        if accelerator.is_local_main_process:
            transformers_logging.set_verbosity_warning()
            diffusers_logging.set_verbosity_info()
        else:
            transformers_logging.set_verbosity_error()
            diffusers_logging.set_verbosity_error()

        return get_accelerate_logger(__name__, log_level="INFO")

    def accelerate(self):
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = int(np.ceil(
            len(self.train_dataloader) / self.cfg.solver.gradient_accumulation_steps
        ))
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = int(np.ceil(
            self.cfg.solver.max_train_steps / num_update_steps_per_epoch
        ))

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            run_time = datetime.now().strftime("%Y%m%d-%H%M")
            self.accelerator.init_trackers(
                self.cfg.exp_name,
                init_kwargs={"mlflow": {"run_name": run_time}},
            )
            # dump config file
            mlflow.log_dict(OmegaConf.to_container(self.cfg), "config.yaml")
        
        return num_update_steps_per_epoch, num_train_epochs
    
    def get_trainable_params(self, model):
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        total_params = sum(p.numel() for p in trainable_params)
        self.logger.info(f"Total trainable parameters: {total_params}")

        if self.cfg.add_noise_trainable_params:
            self.add_noise_to_trainable_params(model)

        return trainable_params
    
    def add_noise_to_trainable_params(self, model, noise_std=0.1):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * noise_std
                    param.add_(noise)

    def get_optimizer(self, trainable_params, learning_rate):
        if self.cfg.solver.use_8bit_adam:
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
        
        return optimizer_cls(
            trainable_params,
            lr=learning_rate,
            betas=(self.cfg.solver.adam_beta1, self.cfg.solver.adam_beta2),
            weight_decay=self.cfg.solver.adam_weight_decay,
            eps=self.cfg.solver.adam_epsilon,
        )
    
    def get_lr_scheduler(self, optimizer):
        return get_scheduler(
            self.cfg.solver.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.solver.lr_warmup_steps
            * self.cfg.solver.gradient_accumulation_steps,
            num_training_steps=self.cfg.solver.max_train_steps
            * self.cfg.solver.gradient_accumulation_steps,
        )

    def get_dataloaders(self, accelerator):
        dataset = TrainingDataset(
            window_length=self.cfg.data.window_length, 
            window_stride=self.cfg.data.window_stride,
            accelerator=accelerator
        )
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.cfg.data.train_batch_size, 
            shuffle=True, 
            num_workers=self.cfg.data.num_workers,
            collate_fn=lambda x: collate_fn(x, self.weight_dtype),
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.cfg.data.train_batch_size, 
            shuffle=False, 
            num_workers=self.cfg.data.num_workers,
            collate_fn=lambda x: collate_fn(x, self.weight_dtype),
            pin_memory=True
        )

        return train_dataloader, train_dataset, val_dataloader, val_dataset
    
    def log_infos(self, num_train_epochs):
        self.logger.info(self.accelerator.state, main_process_only=False)

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num train examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num val examples = {len(self.val_dataset)}")
        self.logger.info(f"  Num Epochs = {num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.cfg.data.train_batch_size}")

        total_batch_size = (
            self.cfg.data.train_batch_size
            * self.accelerator.num_processes
            * self.cfg.solver.gradient_accumulation_steps
        )
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        self.logger.info(
            f"  Gradient Accumulation steps = {self.cfg.solver.gradient_accumulation_steps}"
        )
        self.logger.info(f"  Total optimization steps = {self.cfg.solver.max_train_steps}")

    def load_from_checkpoints(self, num_update_steps_per_epoch):
        if self.accelerator.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
                
        first_epoch = 0
        global_step = 0

        if self.cfg.resume_from_checkpoint:
            if self.cfg.resume_from_checkpoint != "latest":
                resume_dir = self.cfg.resume_from_checkpoint
            else:
                resume_dir = self.save_dir
            
            path, global_step = get_last_checkpoint(resume_dir)
            
            try:
                self.accelerator.load_state(os.path.join(resume_dir, path))
            except Exception as e:
                self.accelerator.print(f"Only loading weights because unable to load the whole checkpoint: {e}")
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                checkpoint_path = os.path.join(resume_dir, path, "model.safetensors")
                checkpoint = load_file(checkpoint_path)
                unwrapped_model.load_state_dict(checkpoint)

            self.accelerator.print(f"Resuming from checkpoint {path}")

            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
        
        return first_epoch, global_step
    
    def get_progress_bar(self, current_step, max_step):
        # Only show the progress bar once on each machine.
        return tqdm(
            range(current_step, max_step),
            disable=not self.accelerator.is_local_main_process,
        )

    def get_noise(self, latents):
        noise = torch.randn_like(latents)
        if self.cfg.noise_offset > 0:
            noise += self.cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1, 1),
                device=latents.device,
            )
        return noise
    
    def get_timesteps(self, latents):
        return torch.randint(
                    0,
                    self.train_noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

    def get_target_noise(self, latents, noise, timesteps):
        if self.train_noise_scheduler.config.prediction_type == "epsilon":
            return noise
        elif self.train_noise_scheduler.config.prediction_type == "v_prediction":
            return self.train_noise_scheduler.get_velocity(
                latents, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.train_noise_scheduler.config.prediction_type}"
            )
    
    def get_loss(self, noise_pred, target_noise, timesteps):
        if self.cfg.snr_gamma == 0:
            loss = F.mse_loss(
                noise_pred.float(), target_noise.float(), reduction="mean"
            )
        else:
            snr = compute_snr(self.train_noise_scheduler, timesteps)
            if self.train_noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )
            loss = F.mse_loss(
                noise_pred.float(), target_noise.float(), reduction="none"
            )
            loss = (
                loss.mean(dim=list(range(1, len(loss.shape))))
                * mse_loss_weights
            )
            loss = loss.mean()
        
        avg_loss = self.accelerator.gather(loss.repeat(self.cfg.data.train_batch_size)).mean()
        global_loss = avg_loss.item() / self.cfg.solver.gradient_accumulation_steps
        
        return loss, global_loss

    def backpropagate(self, loss):
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.trainable_params,
                self.cfg.solver.max_grad_norm,
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
    
    def run_one_step(self, batch, is_validation=False):
        with self.accelerator.accumulate(self.model):
            rast_2d_joints, a_pose_clip, latents_scene, latents_occlusion, latent_video, latent_apose = batch

            latent_video = latent_video.transpose(1, 2)  # (b, c, f, h, w)

            noise = self.get_noise(latent_video)
            timesteps = self.get_timesteps(latent_video)

            noisy_latent_video = self.train_noise_scheduler.add_noise(
                latent_video, noise, timesteps
            )
            target_noise = self.get_target_noise(latent_video, noise, timesteps)

            uncond_fwd = random.random() < self.cfg.uncond_ratio
            
            noise_pred = self.model(
                        noisy_latent_video,
                        timesteps,
                        latent_apose,
                        a_pose_clip,
                        rast_2d_joints,
                        latents_scene,
                        latents_occlusion,
                        uncond_fwd=uncond_fwd,
                    )
            
            loss, global_loss = self.get_loss(noise_pred, target_noise, timesteps)
            if not is_validation:
                self.backpropagate(loss)

            return loss, global_loss
    
    def reset_after_run(self, progress_bar, t_data, loss, global_loss=None):
        if self.accelerator.sync_gradients:
            self.reference_control_reader.clear()
            self.reference_control_writer.clear()
            progress_bar.update(1)

            if global_loss is not None:
                self.accelerator.log({"train_loss": global_loss}, step=self.global_step)
                self.global_step += 1
                self.validate()
        
        logs = {
            "step_loss": loss.detach().item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
            "td": f"{t_data:.2f}s",
        }
        progress_bar.set_postfix(**logs)

    def run_one_epoch(self, progress_bar):
        t_data_start = time.time()
        for step, batch in enumerate(self.train_dataloader):
            t_data = time.time() - t_data_start
            loss, global_loss = self.run_one_step(batch)
            self.reset_after_run(progress_bar, t_data, loss, global_loss)
            t_data_start = time.time()

            if self.global_step >= self.cfg.solver.max_train_steps:
                break
    
    def validate(self):
        if self.global_step % self.cfg.val.validation_steps == 0:
            progress_bar = self.get_progress_bar(0, self.validate_steps)
            progress_bar.set_description(f"Validation")
            
            total_val_loss = 0.0
            with torch.no_grad():
                t_data_start = time.time()
                for step, batch in enumerate(self.val_dataloader):
                    t_data = time.time() - t_data_start
                    loss, global_loss = self.run_one_step(batch, is_validation=True)
                    total_val_loss += global_loss
                    self.reset_after_run(progress_bar, t_data, loss)
                    t_data_start = time.time()
                
            total_val_loss /= len(self.val_dataloader)
            self.accelerator.log({"val_loss": total_val_loss}, step=self.global_step)

            if total_val_loss < self.best_total_val_loss:
                self.best_total_val_loss = total_val_loss
                self.save_model(self.save_dir_val)

    def train(self):
        num_update_steps_per_epoch, num_train_epochs = self.accelerate()
        self.log_infos(num_train_epochs)
        
        first_epoch, self.global_step = self.load_from_checkpoints(num_update_steps_per_epoch)
        progress_bar = self.get_progress_bar(self.global_step, self.cfg.solver.max_train_steps)
        
        for epoch in range(first_epoch, num_train_epochs):
            progress_bar.set_description(f"Epoch {epoch}")
            self.run_one_epoch(progress_bar)
            self.save_model(self.save_dir)
        
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def save_model(self, save_dir):
        if self.accelerator.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"checkpoint-{self.global_step}")
            delete_additional_ckpt(save_dir, 1)
            self.accelerator.save_state(save_path)
            # save motion module and pose guider only
            # unwrap_model = self.accelerator.unwrap_model(self.model)
            # save_checkpoint(
            #     unwrap_model.denoising_unet,
            #     save_dir,
            #     ("motion_module", "pose_guider"),
            #     self.global_step,
            #     self.logger,
            #     total_limit=3,
            # )