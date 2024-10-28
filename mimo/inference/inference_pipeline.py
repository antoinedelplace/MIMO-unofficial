import sys
sys.path.append(".")

from mimo.configs.paths import ANIMATE_ANYONE_REPO, SAM2_REPO, BASE_MODEL_FOLDER, ANIMATE_ANYONE_FOLDER, CHECKPOINTS_FOLDER, SAM2_REPO
sys.path.append(ANIMATE_ANYONE_REPO)
sys.path.append(SAM2_REPO)

import os, logging, cv2
import torch
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.logging import get_logger as get_accelerate_logger

from transformers.utils import logging as transformers_logging

from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import logging as diffusers_logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor

from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.resnet import InflatedConv3d
from src.pipelines.context import get_context_scheduler

from sam2.build_sam import build_sam2_video_predictor

from mimo.training.training_utils import get_torch_weight_dtype
from mimo.training.models import Net
from mimo.inference.inference_utils import create_video_from_frames, remove_tmp_dir, get_extra_kwargs_scheduler

from mimo.utils.general_utils import get_gpu_memory_usage
from mimo.utils.torch_utils import free_gpu_memory, seed_everything
from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.depth_anything_v2_utils import DepthBatchPredictor
from mimo.utils.detectron2_utils import DetectronBatchPredictor
from mimo.utils.propainter_utils import ProPainterBatchPredictor
from mimo.utils.apose_ref_utils import download_base_model, download_anyone, download_dwpose, ReposerBatchPredictor, get_kps_image, CustomDWposeDetector
from mimo.utils.clip_embedding_utils import download_image_encoder, CLIPBatchPredictor
from mimo.utils.vae_encoding_utils import download_vae, VaeBatchPredictor
from mimo.utils.pose_4DH_utils import HMR2_4dhuman

from mimo.dataset_preprocessing.video_sampling_resizing import sampling_resizing
from mimo.dataset_preprocessing.depth_estimation import DEPTH_ANYTHING_MODEL_CONFIGS, get_depth
from mimo.dataset_preprocessing.human_detection_detectron2 import get_cfg_settings, post_processing_detectron2
from mimo.dataset_preprocessing.video_tracking_sam2 import get_instance_sam_output, process_layers
from mimo.dataset_preprocessing.video_inpainting import inpaint_frames
from mimo.dataset_preprocessing.get_apose_ref import get_apose_ref_img
from mimo.dataset_preprocessing.pose_estimation_4DH import get_cfg, get_data_from_4DH
from mimo.dataset_preprocessing.rasterizer_2d_joints import get_rasterized_joints_2d


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
            batch_size_depth,
            batch_size_detectron2,
            batch_size_propainter,
            batch_size_reposer,
            batch_size_clip,
            batch_size_vae,
        ):
        self.num_scheduler_steps = num_scheduler_steps
        self.guidance_scale = guidance_scale
        self.num_workers = num_workers
        self.input_net_size = input_net_size
        self.input_net_fps = input_net_fps
        self.a_pose_raw_path = a_pose_raw_path
        self.depth_anything_encoder = depth_anything_encoder
        self.score_threshold_detectron2 = score_threshold_detectron2
        self.batch_size_depth = batch_size_depth
        self.batch_size_detectron2 = batch_size_detectron2
        self.batch_size_propainter = batch_size_propainter
        self.batch_size_reposer = batch_size_reposer
        self.batch_size_clip = batch_size_clip
        self.batch_size_vae = batch_size_vae

        self.infer_cfg = OmegaConf.load("./mimo/configs/inference/inference.yaml")

        self.generator = seed_everything(seed)

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

    def apply_reference_image(self, model, reference_control_reader, reference_control_writer, latent_apose, a_pose_clip, timestep):
        encoder_hidden_states = a_pose_clip.unsqueeze(1) # (b, 1, d)
        if self.do_classifier_free_guidance:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            ) # (2b, 1, d)

        model.reference_unet(
            latent_apose.repeat(
                (2 if self.do_classifier_free_guidance else 1), 1, 1, 1
            ),  # (2b, c, h, w) or (b, c, h, w)
            torch.zeros_like(timestep),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        reference_control_reader.update(reference_control_writer)

    def get_init_latent(self, latents_scene, noise_scheduler):
        latents = randn_tensor(latents_scene.shape, generator=self.generator)
        latents = latents * noise_scheduler.init_noise_sigma

        return latents
    
    def get_global_context(self, nb_frames):
        context_scheduler = get_context_scheduler("uniform")

        context_queue = list(
            context_scheduler(
                step=0,
                num_steps=self.num_inference_steps,
                num_frames=nb_frames,
                context_size=self.infer_cfg.context.context_frames,
                context_stride=self.infer_cfg.context_stride,
                context_overlap=self.infer_cfg.context_overlap,
            )
        )
        num_context_batches = np.ceil(len(context_queue) / self.infer_cfg.context_batch_size)
        global_context = []
        for i in range(num_context_batches):
            global_context.append(
                context_queue[
                i * self.infer_cfg.context_batch_size: (i + 1) * self.infer_cfg.context_batch_size
                ]
            )
        
        return global_context

    def apply_guidance(self, noise_pred, counter):
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)  # Mean along context
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        
        return noise_pred

    def get_noise_pred(self, timestep, model, a_pose_clip, noise_scheduler, pose_features, latents):
        encoder_hidden_states = a_pose_clip.unsqueeze(1) # (b, 1, d)

        (b, c, f, h, w) = latents.shape
        
        noise_pred = torch.zeros((b * (2 if self.do_classifier_free_guidance else 1), c, f, h, w))  # (2b, c, f, h, w) or (b, c, f, h, w)
        counter = torch.zeros((1, 1, f, 1, 1))

        for context in self.get_global_context(latents.shape[2]):
            latent_model_input = (
                torch.cat([latents[:, :, i_f] for i_f in context])
                .repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)
            )  # (2b, c, len(context), h, w) or (b, c, len(context), h, w)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep)
            
            latent_pose_input = torch.cat(
                [pose_features[:, :, i_f] for i_f in context]
            ).repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)  # (2b, c, len(context), h, w) or (b, c, len(context), h, w)

            pred = model.denoising_unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                pose_cond_fea=latent_pose_input,
                return_dict=False,
            )[0]

            for i_f in context:
                noise_pred[:, :, i_f] = noise_pred[:, :, i_f] + pred
                counter[:, :, i_f] = counter[:, :, i_f] + 1
        
        noise_pred = self.apply_guidance(noise_pred, counter)
        
        return noise_pred
    
    def update_progress_bar(self, progress_bar, i_step, noise_scheduler, num_warmup_steps):
        if i_step == len(noise_scheduler.timesteps) - 1 or (
                (i_step + 1) > num_warmup_steps and (i_step + 1) % noise_scheduler.order == 0
            ):
                progress_bar.update()
        
        return progress_bar

    def apply_diffusion(self, rast_2d_joints, a_pose_clip, latents_scene, latents_occlusion, latent_apose):
        model, reference_control_writer, reference_control_reader = self.get_model()

        noise_scheduler = self.get_noise_scheduler()
        extra_kwargs_scheduler = get_extra_kwargs_scheduler(self.generator, eta=0.0, noise_scheduler=noise_scheduler)

        model = self.accelerator.prepare(model)

        num_warmup_steps = len(noise_scheduler.timesteps) - self.num_scheduler_steps * noise_scheduler.order
        progress_bar = self.get_progress_bar(self.num_scheduler_steps)

        self.apply_reference_image(model, reference_control_reader, reference_control_writer, latent_apose, a_pose_clip, timestep)

        rast_2d_joints = rast_2d_joints.transpose(1, 2)  # (b, c, f, h, w)
        pose_features = self.pose_guider(rast_2d_joints)  # (b, c, f, h, w)

        latents_scene = latents_scene.transpose(1, 2)  # (b, c, f, h, w)
        latents_occlusion = latents_occlusion.transpose(1, 2)  # (b, c, f, h, w)
        noisy_latent_video = self.get_init_latent(latents_scene, noise_scheduler)  # (b, c, f, h, w)
        latents = torch.cat((noisy_latent_video, latents_scene, latents_occlusion), dim=1)  # (b, c+c+c, f, h, w)
        
        for i_step, timestep in enumerate(noise_scheduler.timesteps):
            noise_pred = self.get_noise_pred(timestep, model, noise_scheduler, a_pose_clip, pose_features, latents)
            latents = noise_scheduler(noise_pred, timestep, latents, **extra_kwargs_scheduler).prev_sample
            progress_bar = self.update_progress_bar(progress_bar, i_step, noise_scheduler, num_warmup_steps)
        
        reference_control_reader.clear()
        reference_control_writer.clear()

    def __call__(self, input_video_path, output_video_path):
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
                                           input_size=(width, height), 
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

        input_video_path = create_video_from_frames(resized_frames, self.input_net_fps)  # for SAM2 and 4DH

        human_frames, occlusion_frames, scene_frames = self.get_layers_sam2(input_video_path, resized_frames, detectron2_output, depth_frames)
        del human_frames, detectron2_output, depth_frames
        occlusion_frames = np.array(occlusion_frames)
        scene_frames = np.array(scene_frames)
        print("np.shape(occlusion_frames)", np.shape(occlusion_frames), type(occlusion_frames))
        print("np.shape(scene_frames)", np.shape(scene_frames), type(scene_frames))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        joints2d = self.get_joints2d(input_video_path)
        print("np.shape(joints2d)", np.shape(joints2d), type(joints2d))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        remove_tmp_dir(input_video_path)

        joints2d = get_rasterized_joints_2d(joints2d, self.input_net_size, self.input_net_size)
        print("np.shape(joints2d)", np.shape(joints2d), type(joints2d))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

        scene_frames = np.array(self.inpaint_scene_layer(scene_frames))
        print("np.shape(scene_frames)", np.shape(scene_frames), type(scene_frames))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()

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

        resized_frames = self.apply_diffusion(joints2d, clip_embeddings, scene_frames, occlusion_frames, apose_ref)
        del scene_frames, occlusion_frames, joints2d, clip_embeddings, apose_ref
        print("np.shape(resized_frames)", np.shape(resized_frames), type(resized_frames))
        free_gpu_memory(self.accelerator)
        get_gpu_memory_usage()