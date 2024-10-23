import sys
sys.path.append(".")

from configs.paths import ANIMATE_ANYONE_REPO, BASE_MODEL_FOLDER, ANIMATE_ANYONE_FOLDER
sys.path.append(ANIMATE_ANYONE_REPO)

import os, logging, cv2
import torch

from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.logging import get_logger as get_accelerate_logger

from transformers.utils import logging as transformers_logging

from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import logging as diffusers_logging
from diffusers.utils.import_utils import is_xformers_available

from src.utils.util import seed_everything
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.resnet import InflatedConv3d

from training.training_utils import get_torch_weight_dtype

from utils.video_utils import frame_gen_from_video
from utils.depth_anything_v2_utils import DepthBatchPredictor
from utils.vae_encoding_utils import download_vae, VaeBatchPredictor

from dataset_preprocessing.video_sampling_resizing import sampling_resizing

class InferencePipeline():
    def __init__(self, seed, weight_dtype, batch_size_depth, num_workers):
        self.batch_size_depth = batch_size_depth
        self.num_workers = num_workers

        self.infer_cfg = OmegaConf.load("./configs/inference/inference.yaml")

        seed_everything(seed)

        self.weight_dtype_str = weight_dtype
        self.weight_dtype = get_torch_weight_dtype(weight_dtype)

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
        vae = VaeBatchPredictor(batch_size, workers)

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

        # Set motion module learnable
        for name, module in denoising_unet.named_modules():
            if "motion_modules" in name:
                for params in module.parameters():
                    params.requires_grad = True
        
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
        
        model = Net(
            reference_unet,
            denoising_unet,
            pose_guider,
            reference_control_writer,
            reference_control_reader,
        )
        
        return model, reference_control_writer, reference_control_reader

    
    def depth_estimation(self, batch_size=12, workers=8, input_size=768, encoder='vitl'):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        depth_anything = DepthBatchPredictor(self.batch_size_depth, 
                                             self.num_workers, input_size, input_size, **model_configs[encoder])
        depth_anything.load_state_dict(torch.load(os.path.join(CHECKPOINTS_FOLDER, f'depth_anything_v2_{encoder}.pth'), map_location='cpu'))
        depth_anything = depth_anything.to(torch.bfloat16).to(device).eval()

    def __call__(self, input_video_path, output_video_path):
        video = cv2.VideoCapture(input_video_path)

        basename = os.path.basename(input_video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # print("basename", basename)
        # print("width", width)
        # print("height", height)
        # print("frames_per_second", frames_per_second)
        # print("num_frames", num_frames)

        self.frame_gen = list(frame_gen_from_video(video))

        self.resized_frames = sampling_resizing(self.frame_gen, frames_per_second, output_fps=24, input_size=(width, height), output_width=768)

        video.release()








    @torch.no_grad()
    def erbhfilbv(
        self,
        ref_image,
        pose_images,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        decoder_consistency=None,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            #ref_image.resize((224, 224)),
            self.square_pad(ref_image),
            return_tensors="pt",
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # Prepare a list of pose condition images
        pose_cond_tensor_list = []
        for pose_image in pose_images:
            pose_cond_tensor = self.cond_image_processor.preprocess(
                pose_image, height=height, width=width
            )
            pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
            pose_cond_tensor_list.append(pose_cond_tensor)
        pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=2)  # (bs, c, t, h, w)
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        pose_fea = self.pose_guider(pose_cond_tensor)

        context_scheduler = get_context_scheduler(context_schedule)

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        # t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        0,
                    )
                )
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        context_overlap,
                    )
                )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                global_context = []
                for i in range(num_context_batches):
                    global_context.append(
                        context_queue[
                        i * context_batch_size: (i + 1) * context_batch_size
                        ]
                    )

                for context in global_context:
                    # 3.1 expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    b, c, f, h, w = latent_model_input.shape
                    latent_pose_input = torch.cat(
                        [pose_fea[:, :, c] for c in context]
                    ).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)

                    pred = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states[:b],
                        pose_cond_fea=latent_pose_input,
                        return_dict=False,
                    )[0]

                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)
        # Post-processing
        images = self.decode_latents(latents, decoder_consistency=decoder_consistency)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)