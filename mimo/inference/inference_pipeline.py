import sys
sys.path.append(".")

from mimo.configs.paths import ANIMATE_ANYONE_REPO, SAM2_REPO, BASE_MODEL_FOLDER, ANIMATE_ANYONE_FOLDER, CHECKPOINTS_FOLDER, SAM2_REPO
sys.path.append(ANIMATE_ANYONE_REPO)
sys.path.append(SAM2_REPO)

import os, logging, cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf

from safetensors.torch import load_file

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.logging import get_logger as get_accelerate_logger

from transformers.utils import logging as transformers_logging

from diffusers import DDIMScheduler
from diffusers.utils import logging as diffusers_logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor

from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.resnet import InflatedConv3d

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from mimo.training.training_utils import get_torch_weight_dtype, get_last_checkpoint
from mimo.training.models import Net
from mimo.inference.inference_utils import create_video_from_frames, remove_tmp_dir, get_extra_kwargs_scheduler
from mimo.inference.inference_dataset import InferenceDataset, collate_fn, collate_fn_rast

from mimo.utils.general_utils import get_gpu_memory_usage
from mimo.utils.torch_utils import free_gpu_memory, seed_everything, NpzDataset
from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.depth_anything_v2_utils import DepthBatchPredictor
from mimo.utils.detectron2_utils import DetectronBatchPredictor
from mimo.utils.propainter_utils import ProPainterBatchPredictor
from mimo.utils.apose_ref_utils import download_base_model, download_anyone, download_dwpose, ReposerBatchPredictor, get_kps_image, CustomDWposeDetector
from mimo.utils.clip_embedding_utils import download_image_encoder, CLIPBatchPredictor
from mimo.utils.vae_encoding_utils import download_vae, VaeBatchPredictor
from mimo.utils.pose_4DH_utils import HMR2_4dhuman

from mimo.dataset_preprocessing.video_sampling_resizing import sampling_resizing, resize_frame
from mimo.dataset_preprocessing.depth_estimation import DEPTH_ANYTHING_MODEL_CONFIGS, get_depth
from mimo.dataset_preprocessing.human_detection_detectron2 import get_cfg_settings, post_processing_detectron2
from mimo.dataset_preprocessing.video_tracking_sam2 import get_instance_sam_output, process_layers, get_index_first_frame_with_character, get_global_index_most_central_human_in_frame
from mimo.dataset_preprocessing.video_inpainting import inpaint_frames
from mimo.dataset_preprocessing.get_apose_ref import get_apose_ref_img
from mimo.dataset_preprocessing.pose_estimation_4DH import get_cfg, get_data_from_4DH
from mimo.dataset_preprocessing.rasterizer_2d_joints import RasterizerBatchPredictor


class InferencePipeline():
    def __init__(  # See default values in inference/main.py
            self, 
            seed, 
            num_scheduler_steps,
            guidance_scale,
            weight_dtype, 
            num_workers,
            input_net_size,
            input_net_fps,
            a_pose_raw_path,
            depth_anything_encoder,
            score_threshold_detectron2,
            neighbor_context_mimo,
            batch_size_mimo,
            batch_size_depth,
            batch_size_detectron2,
            batch_size_propainter,
            batch_size_reposer,
            batch_size_clip,
            batch_size_vae,
            batch_size_rasterizer
        ):
        self.num_scheduler_steps = num_scheduler_steps
        self.guidance_scale = guidance_scale
        self.num_workers = num_workers
        self.input_net_size = input_net_size
        self.input_net_fps = input_net_fps
        self.a_pose_raw_path = a_pose_raw_path
        self.depth_anything_encoder = depth_anything_encoder
        self.score_threshold_detectron2 = score_threshold_detectron2
        self.neighbor_context_mimo = neighbor_context_mimo
        self.batch_size_mimo = batch_size_mimo
        self.batch_size_depth = batch_size_depth
        self.batch_size_detectron2 = batch_size_detectron2
        self.batch_size_propainter = batch_size_propainter
        self.batch_size_reposer = batch_size_reposer
        self.batch_size_clip = batch_size_clip
        self.batch_size_vae = batch_size_vae
        self.batch_size_rasterizer = batch_size_rasterizer

        self.infer_cfg = OmegaConf.load("./mimo/configs/inference/inference.yaml")

        seed_everything(seed)

        self.weight_dtype_str = weight_dtype
        self.weight_dtype = get_torch_weight_dtype(weight_dtype)

        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        self.accelerator = self.get_accelerator()
        self.logger = self.get_logger(self.accelerator)
    
    def get_accelerator(self):
        return Accelerator(
            mixed_precision=self.weight_dtype_str,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
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
    
    def get_model(self):
        download_base_model()
        download_anyone()

        reference_unet = UNet2DConditionModel.from_pretrained(
            os.path.join(BASE_MODEL_FOLDER, "unet"),
            use_safetensors=True
        )
        reference_unet.load_state_dict(
            torch.load(os.path.join(ANIMATE_ANYONE_FOLDER, "reference_unet.pth"), map_location="cpu", weights_only=True),
        )
        
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            os.path.join(BASE_MODEL_FOLDER, "unet"),
            os.path.join(ANIMATE_ANYONE_FOLDER, "motion_module.pth"),
            unet_additional_kwargs=self.infer_cfg.unet_additional_kwargs,
        )
        denoising_unet.load_state_dict(
            torch.load(os.path.join(ANIMATE_ANYONE_FOLDER, "denoising_unet.pth"), map_location="cpu", weights_only=True),
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
            torch.load(os.path.join(ANIMATE_ANYONE_FOLDER, "pose_guider.pth"), map_location="cpu", weights_only=True),
        )
        
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
        
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
        
        model = Net(
            reference_unet,
            denoising_unet,
            pose_guider,
            reference_control_writer,
            reference_control_reader,
        )

        if self.infer_cfg.load_from_checkpoint:
            if self.infer_cfg.load_from_checkpoint.endswith(".safetensors"):
                checkpoint_path = self.infer_cfg.load_from_checkpoint
            else:
                path, global_step = get_last_checkpoint(self.infer_cfg.load_from_checkpoint)
                checkpoint_path = os.path.join(self.infer_cfg.load_from_checkpoint, path, "model.safetensors")
            checkpoint = load_file(checkpoint_path)
            model.load_state_dict(checkpoint)
        
        return model, reference_control_writer, reference_control_reader

    def get_noise_scheduler(self):
        sched_kwargs = OmegaConf.to_container(self.infer_cfg.noise_scheduler_kwargs)
        noise_scheduler = DDIMScheduler(**sched_kwargs)
        noise_scheduler.set_timesteps(self.num_scheduler_steps)

        return noise_scheduler
    
    def get_progress_bar(self, max_step):
        # Only show the progress bar once on each machine.
        return tqdm(
            range(max_step),
            disable=not self.accelerator.is_local_main_process,
        )
    
    def depth_estimation(self, resized_frames):
        depth_anything = DepthBatchPredictor(self.batch_size_depth, 
                                             self.num_workers, 
                                             self.weight_dtype,
                                             self.input_net_size, 
                                             self.input_net_size, 
                                             **DEPTH_ANYTHING_MODEL_CONFIGS[self.depth_anything_encoder])
        depth_anything.load_state_dict(torch.load(os.path.join(CHECKPOINTS_FOLDER, f'depth_anything_v2_{self.depth_anything_encoder}.pth'), map_location='cpu', weights_only=True))

        depth_anything = self.accelerator.prepare(depth_anything)

        return get_depth(resized_frames, depth_anything)

    def get_detectron2_output(self, resized_frames):
        cfg = get_cfg_settings()
        predictor = DetectronBatchPredictor(cfg, self.batch_size_detectron2, self.num_workers)

        predictor.model = self.accelerator.prepare(predictor.model)

        return post_processing_detectron2(list(predictor(resized_frames)))
    
    def get_layers_sam2(self, input_video_path, resized_frames, detectron2_output, depth_frames):
        checkpoint = os.path.join(CHECKPOINTS_FOLDER, "sam2.1_hiera_large.pt")
        model_cfg = os.path.join(SAM2_REPO, "configs/sam2.1/sam2.1_hiera_l.yaml")
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)

        predictor = self.accelerator.prepare(predictor)

        (
            min_frame_idx, 
            max_frame_idx, 
            instance_sam_output, 
            foreground_mask
        ) = get_instance_sam_output(input_video_path, detectron2_output, depth_frames, predictor, self.score_threshold_detectron2)

        return process_layers(resized_frames, 
                                instance_sam_output, 
                                foreground_mask, 
                                min_frame_idx, 
                                max_frame_idx)
    
    def inpaint_scene_layer(self, scene_frames):
        predictor = ProPainterBatchPredictor(self.batch_size_propainter, self.num_workers)

        return inpaint_frames(scene_frames, predictor)

    def get_apose_ref(self, resized_frames):
        download_image_encoder()
        download_vae()
        download_base_model()
        download_anyone()
        download_dwpose()

        dw_pose_detector = CustomDWposeDetector()
        vae = VaeBatchPredictor(self.batch_size_reposer, self.num_workers)
        clip = CLIPBatchPredictor(self.batch_size_reposer, self.num_workers)
        reposer = ReposerBatchPredictor(self.batch_size_reposer, self.num_workers, clip, vae)

        reposer.pipe, dw_pose_detector = self.accelerator.prepare(reposer.pipe, dw_pose_detector)

        a_pose_kps, ref_points_2d = get_kps_image(self.a_pose_raw_path, dw_pose_detector)

        return get_apose_ref_img(resized_frames, reposer, a_pose_kps, ref_points_2d, dw_pose_detector)

    def clip_apose(self, apose_ref):
        download_image_encoder()

        clip = CLIPBatchPredictor(self.batch_size_clip, self.num_workers)

        clip.image_enc = self.accelerator.prepare(clip.image_enc)

        return list(clip([apose_ref]))[0]
    
    def vae_encoding(self, scene_frames, occlusion_frames, resized_frames, apose_ref):
        download_vae()

        vae = VaeBatchPredictor(self.batch_size_vae, self.num_workers)

        vae.vae = self.accelerator.prepare(vae.vae)

        scene_frames = np.concatenate(list(vae.encode(scene_frames)))
        occlusion_frames = np.concatenate(list(vae.encode(occlusion_frames)))
        resized_frames = np.concatenate(list(vae.encode(resized_frames)))
        apose_ref = np.concatenate(list(vae.encode([apose_ref])))

        return scene_frames, occlusion_frames, resized_frames, apose_ref

    def get_joints2d(self, input_video_path):
        phalp_tracker = HMR2_4dhuman(get_cfg())

        phalp_tracker.HMAR = self.accelerator.prepare(phalp_tracker.HMAR)

        data_4DH = get_data_from_4DH(input_video_path, phalp_tracker)

        return data_4DH["data_joints_2d"]

    def get_rasterized_joints_2d(self, data_joints_2d):
        rasterizer = RasterizerBatchPredictor(self.batch_size_rasterizer, self.num_workers, self.input_net_size, self.input_net_size)

        return np.concatenate(list(rasterizer(data_joints_2d)))

    def apply_reference_image(self, model, reference_control_reader, reference_control_writer, latent_apose, encoder_hidden_states):
        t = torch.zeros((1), dtype=latent_apose.dtype, device=latent_apose.device)

        model.reference_unet(
            latent_apose.repeat(
                (2 if self.do_classifier_free_guidance else 1), 1, 1, 1
            ),  # (2b, c, h, w) or (b, c, h, w)
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        reference_control_reader.update(reference_control_writer)

    def get_init_latent(self, latents_scene, noise_scheduler):
        latents = randn_tensor(latents_scene.shape, generator=torch.Generator(device=latents_scene.device), device=latents_scene.device, dtype=latents_scene.dtype)
        latents = latents * noise_scheduler.init_noise_sigma

        return latents

    def apply_guidance(self, noise_pred):
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        
        return noise_pred

    def get_noise_pred(self, timestep, model, noise_scheduler, latent_pose, latents, latents_scene, latents_occlusion, encoder_hidden_states):
        latents = torch.cat((latents, latents_scene, latents_occlusion), dim=1)  # (b, c+c+c, f, h, w)
        latents = latents.repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)  # (2b, c, f, h, w) or (b, c, f, h, w)
        latents = noise_scheduler.scale_model_input(latents, timestep)
        
        latent_pose = latent_pose.repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)  # (2b, c, f, h, w) or (b, c, f, h, w)

        noise_pred = model.denoising_unet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            pose_cond_fea=latent_pose,
            return_dict=False,
        )[0]
        
        noise_pred = self.apply_guidance(noise_pred)
        
        return noise_pred
    
    def update_progress_bar(self, progress_bar, i_step, noise_scheduler, num_warmup_steps):
        if i_step == len(noise_scheduler.timesteps) - 1 or (
                (i_step + 1) > num_warmup_steps and (i_step + 1) % noise_scheduler.order == 0
            ):
                progress_bar.update()
        
        return progress_bar

    def get_dataloader(self, latent_pose, latents_scene, latents_occlusion):
        window_stride = self.batch_size_mimo - self.neighbor_context_mimo*2
        if window_stride <= 0:
            raise Exception(f"neighbor_context_mimo ({self.neighbor_context_mimo}) is too big for the given batch_size_mimo ({self.batch_size_mimo}). neighbor_context_mimo should be strictly less than half of batch_size_mimo.")

        dataset = InferenceDataset(latent_pose, latents_scene, latents_occlusion, self.batch_size_mimo, window_stride)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn(x, self.weight_dtype),
            pin_memory=True
        )

        return dataloader
    
    def get_encoder_hidden_states(self, a_pose_clip):
        encoder_hidden_states = a_pose_clip.unsqueeze(1) # (b, 1, d)
        if self.do_classifier_free_guidance:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            ) # (2b, 1, d)
        
        return encoder_hidden_states
    
    def apply_pose_guider(self, model, rast_2d_joints):
        dataset = NpzDataset(rast_2d_joints)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size_mimo, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=lambda x: collate_fn_rast(x, self.weight_dtype),
            pin_memory=True
        )

        with torch.no_grad():
            model.pose_guider, dataloader = self.accelerator.prepare(model.pose_guider, dataloader)
            model.pose_guider = model.pose_guider.to(self.weight_dtype)

            for batch in tqdm(dataloader):
                batch = batch.unsqueeze(0).transpose(1, 2)  # (b, c, f, h, w)
                batch = model.pose_guider(batch)  # (b, c, f, h, w)
                batch = batch.transpose(1, 2).squeeze(0)  # (f, c, h, w)

                yield batch.cpu().float().numpy()

    def apply_diffusion(self, model, reference_control_writer, reference_control_reader, latent_pose, a_pose_clip, latents_scene, latents_occlusion, latent_apose):
        num_frames = len(latent_pose)

        dataloader = self.get_dataloader(latent_pose, latents_scene, latents_occlusion)
        noise_scheduler = self.get_noise_scheduler()
        extra_kwargs_scheduler = get_extra_kwargs_scheduler(torch.Generator(), eta=0.0, noise_scheduler=noise_scheduler)

        with torch.no_grad():
            model.reference_unet, model.denoising_unet, dataloader = self.accelerator.prepare(model.reference_unet, model.denoising_unet, dataloader)
            model.reference_unet = model.reference_unet.to(self.weight_dtype)
            model.denoising_unet = model.denoising_unet.to(self.weight_dtype)

            a_pose_clip = torch.from_numpy(a_pose_clip).to(self.weight_dtype).to(self.accelerator.device)
            latent_apose = torch.from_numpy(latent_apose).to(self.weight_dtype).to(self.accelerator.device)

            a_pose_clip = self.get_encoder_hidden_states(a_pose_clip)
            self.apply_reference_image(model, reference_control_reader, reference_control_writer, latent_apose, a_pose_clip)

            count_frames = 0
            for i_batch, batch in enumerate(tqdm(dataloader)):
                latent_pose_torch, latents_scene_torch, latents_occlusion_torch = batch
                latent_pose_torch = latent_pose_torch.transpose(1, 2)  # (b, c, f, h, w)
                latents_scene_torch = latents_scene_torch.transpose(1, 2)  # (b, c, f, h, w)
                latents_occlusion_torch = latents_occlusion_torch.transpose(1, 2)  # (b, c, f, h, w)
                latents = self.get_init_latent(latents_scene_torch, noise_scheduler)  # (b, c, f, h, w)

                num_warmup_steps = len(noise_scheduler.timesteps) - self.num_scheduler_steps * noise_scheduler.order
                progress_bar = self.get_progress_bar(self.num_scheduler_steps)
                
                # Possibility here to improve blending of windows by doing mean of predicted noises over several windows. This would increase compute time because some GPU / CPU transfer are needed to save GPU memory
                for i_step, timestep in enumerate(noise_scheduler.timesteps):
                    noise_pred = self.get_noise_pred(timestep, model, noise_scheduler, latent_pose_torch, latents, latents_scene_torch, latents_occlusion_torch, a_pose_clip)
                    latents = noise_scheduler.step(noise_pred, timestep, latents, **extra_kwargs_scheduler).prev_sample
                    progress_bar = self.update_progress_bar(progress_bar, i_step, noise_scheduler, num_warmup_steps)
                
                latents = latents.transpose(1, 2).squeeze(0)  # (f, c, h, w)

                start = self.neighbor_context_mimo
                end = -self.neighbor_context_mimo
                if i_batch == 0:
                    start = 0
                if i_batch == len(dataloader)-1:
                    start = len(latents) - (num_frames-count_frames)
                    end = len(latents)
                
                output = latents[start:end].cpu().float().numpy()
                count_frames += len(output)
                yield output
            
            reference_control_reader.clear()
            reference_control_writer.clear()

    def get_images_from_latents(self, latents):
        download_vae()

        vae = VaeBatchPredictor(self.batch_size_vae, self.num_workers)

        vae.vae = self.accelerator.prepare(vae.vae)

        output_images = np.concatenate(list(vae.decode(latents)))

        return output_images
    
    def resize_back_frames_and_save(self, frames, ori_frames, width, height, fps, output_video_path):
        output_file = cv2.VideoWriter(
            filename=output_video_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=fps,
            frameSize=(width, height),
            isColor=True,
        )
        
        for frame in frames:
            if width < height:
                scale = self.input_net_size/height
                frame = frame[:, int(scale*(height-width)//2): -int(scale*(height-width+1)//2)]
                ori_frames = cv2.resize(frame, (width, height), interpolation = cv2.INTER_LINEAR)
            else:
                scale = self.input_net_size/width
                frame = cv2.resize(frame, (height, height), interpolation = cv2.INTER_LINEAR)
                ori_frames[width//2-height//2: width//2+(height+1)//2, :] = frame
            
            output_file.write(ori_frames)
        
        output_file.release()
        print(f"Output video saved here: {output_video_path}")

        video2 = cv2.VideoCapture(output_video_path)

        basename = os.path.basename(output_video_path)
        width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video2.get(cv2.CAP_PROP_FPS)
        num_frames = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
        print("basename", basename)
        print("width", width)
        print("height", height)
        print("frames_per_second", frames_per_second)
        print("num_frames", num_frames)

        video2.release()

    def get_avatar_sam2_mask(self, avatar_image, data_pred_boxes):
        checkpoint = os.path.join(CHECKPOINTS_FOLDER, "sam2.1_hiera_large.pt")
        model_cfg = os.path.join(SAM2_REPO, "configs/sam2.1/sam2.1_hiera_l.yaml")

        sam2_model = build_sam2(model_cfg, checkpoint)
        predictor = SAM2ImagePredictor(sam2_model)

        predictor = self.accelerator.prepare(predictor)

        predictor.set_image(cv2.cvtColor(avatar_image, cv2.COLOR_BGR2RGB))

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=data_pred_boxes,
            multimask_output=False,
        )

        return masks.transpose(1, 2, 0)

    def get_avatar_image(self, input_avatar_image_path):
        avatar_image = resize_frame(cv2.imread(input_avatar_image_path), self.input_net_size)
        print("np.shape(avatar_image)", np.shape(avatar_image), type(avatar_image))

        detectron2_avatar_output = self.get_detectron2_output([avatar_image])
        print("np.shape(detectron2_avatar_output['data_frame_index'])", np.shape(detectron2_avatar_output['data_frame_index']), type(detectron2_avatar_output['data_frame_index']))
        print("np.shape(detectron2_avatar_output['data_pred_boxes'])", np.shape(detectron2_avatar_output['data_pred_boxes']), type(detectron2_avatar_output['data_pred_boxes']))
        print("np.shape(detectron2_avatar_output['data_scores'])", np.shape(detectron2_avatar_output['data_scores']), type(detectron2_avatar_output['data_scores']))
        print("np.shape(detectron2_avatar_output['data_pred_classes'])", np.shape(detectron2_avatar_output['data_pred_classes']), type(detectron2_avatar_output['data_pred_classes']))
        print("np.shape(detectron2_avatar_output['data_pred_masks'])", np.shape(detectron2_avatar_output['data_pred_masks']), type(detectron2_avatar_output['data_pred_masks']))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        i_frame = get_index_first_frame_with_character(detectron2_avatar_output, self.score_threshold_detectron2)
        i_global = get_global_index_most_central_human_in_frame(detectron2_avatar_output, i_frame, self.score_threshold_detectron2, self.input_net_size, self.input_net_size)

        masks = self.get_avatar_sam2_mask(avatar_image, detectron2_avatar_output['data_pred_boxes'][i_global])
        print("np.shape(masks)", np.shape(masks), type(masks))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        return (avatar_image*masks).astype(np.uint8)

    def __call__(self, input_video_path, input_avatar_image_path, input_motion_video_path, output_video_path):
        free_gpu_memory(self.accelerator)

        video = cv2.VideoCapture(input_video_path)

        basename = os.path.basename(input_video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("basename", basename)
        print("width", width)
        print("height", height)
        print("frames_per_second", frames_per_second)
        print("num_frames", num_frames)

        frame_gen = np.array(list(frame_gen_from_video(video)))[:50] # TODO: remove debug
        print("np.shape(frame_gen)", np.shape(frame_gen), type(frame_gen))
        video.release()
        get_gpu_memory_usage()

        resized_frames = np.array(sampling_resizing(frame_gen, 
                                           frames_per_second, 
                                           output_fps=self.input_net_fps, 
                                           output_width=self.input_net_size))
        print("np.shape(resized_frames)", np.shape(resized_frames), type(resized_frames))
        get_gpu_memory_usage()

        depth_frames = np.concatenate(self.depth_estimation(resized_frames))
        print("np.shape(depth_frames)", np.shape(depth_frames), type(depth_frames))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        detectron2_output = self.get_detectron2_output(resized_frames)
        print("np.shape(detectron2_output['data_frame_index'])", np.shape(detectron2_output['data_frame_index']), type(detectron2_output['data_frame_index']))
        print("np.shape(detectron2_output['data_pred_boxes'])", np.shape(detectron2_output['data_pred_boxes']), type(detectron2_output['data_pred_boxes']))
        print("np.shape(detectron2_output['data_scores'])", np.shape(detectron2_output['data_scores']), type(detectron2_output['data_scores']))
        print("np.shape(detectron2_output['data_pred_classes'])", np.shape(detectron2_output['data_pred_classes']), type(detectron2_output['data_pred_classes']))
        print("np.shape(detectron2_output['data_pred_masks'])", np.shape(detectron2_output['data_pred_masks']), type(detectron2_output['data_pred_masks']))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        temp_input_video_path = create_video_from_frames(resized_frames, self.input_net_fps)  # for SAM2 and 4DH

        human_frames, occlusion_frames, scene_frames = self.get_layers_sam2(temp_input_video_path, resized_frames, detectron2_output, depth_frames)
        del human_frames, detectron2_output, depth_frames
        occlusion_frames = np.array(occlusion_frames)
        scene_frames = np.array(scene_frames)
        print("np.shape(occlusion_frames)", np.shape(occlusion_frames), type(occlusion_frames))
        print("np.shape(scene_frames)", np.shape(scene_frames), type(scene_frames))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        if input_motion_video_path is not None:
            video_motion = cv2.VideoCapture(input_motion_video_path)
            frames_per_second_motion = video_motion.get(cv2.CAP_PROP_FPS)
            frame_gen_motion = np.array(list(frame_gen_from_video(video_motion)))
            video_motion.release()
            
            resized_motion_frames = np.array(sampling_resizing(frame_gen_motion, 
                                           frames_per_second_motion, 
                                           output_fps=self.input_net_fps, 
                                           output_width=self.input_net_size))
            temp_motion_video_path = create_video_from_frames(resized_motion_frames, self.input_net_fps)
            del frame_gen_motion, resized_motion_frames
            
            # TODO: check if character is present and adapt animation length to input video
            joints2d = self.get_joints2d(temp_motion_video_path)
            remove_tmp_dir(temp_motion_video_path)
        else:
            joints2d = self.get_joints2d(temp_input_video_path)
        print("np.shape(joints2d)", np.shape(joints2d), type(joints2d))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        remove_tmp_dir(temp_input_video_path)

        joints2d = self.get_rasterized_joints_2d(joints2d)
        print("np.shape(joints2d)", np.shape(joints2d), type(joints2d))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        scene_frames = np.array(self.inpaint_scene_layer(scene_frames))
        print("np.shape(scene_frames)", np.shape(scene_frames), type(scene_frames))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        if input_avatar_image_path is not None:
            avatar_image = self.get_avatar_image(input_avatar_image_path)
            print("np.shape(avatar_image)", np.shape(avatar_image), type(avatar_image))
            # cv2.imwrite("../../data/iron_man_square.png", avatar_image)
            apose_ref = self.get_apose_ref([avatar_image])
            # cv2.imwrite("../../data/iron_man_square_a_pose.png", apose_ref)
        else:
            apose_ref = self.get_apose_ref(resized_frames)
        print("np.shape(apose_ref)", np.shape(apose_ref), type(apose_ref))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        clip_embeddings = self.clip_apose(apose_ref)
        print("np.shape(clip_embeddings)", np.shape(clip_embeddings), type(clip_embeddings))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        scene_frames, occlusion_frames, resized_frames, apose_ref = self.vae_encoding(scene_frames, occlusion_frames, resized_frames, apose_ref)
        del resized_frames
        print("np.shape(scene_frames)", np.shape(scene_frames), type(scene_frames))
        print("np.shape(occlusion_frames)", np.shape(occlusion_frames), type(occlusion_frames))
        print("np.shape(apose_ref)", np.shape(apose_ref), type(apose_ref))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        model, reference_control_writer, reference_control_reader = self.get_model()

        joints2d = np.concatenate(list(self.apply_pose_guider(model, joints2d)))  # (f, c, h, w)
        print("np.shape(joints2d)", np.shape(joints2d), type(joints2d))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        resized_frames = np.concatenate(list(self.apply_diffusion(model, reference_control_writer, reference_control_reader, joints2d, clip_embeddings, scene_frames, occlusion_frames, apose_ref)))  # (f, c, h, w)
        del scene_frames, occlusion_frames, joints2d, clip_embeddings, apose_ref
        print("np.shape(resized_frames)", np.shape(resized_frames), type(resized_frames))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        resized_frames = self.get_images_from_latents(resized_frames)
        print("np.shape(resized_frames)", np.shape(resized_frames), type(resized_frames))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        if output_video_path is None:
            output_video_path = input_video_path.replace(".mp4", "_output.mp4")

        self.resize_back_frames_and_save(resized_frames, frame_gen, width, height, self.input_net_fps, output_video_path)
        return output_video_path