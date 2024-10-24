import sys
sys.path.append(".")

from mimo.configs.paths import ANIMATE_ANYONE_REPO, SAM2_REPO, BASE_MODEL_FOLDER, ANIMATE_ANYONE_FOLDER, CHECKPOINTS_FOLDER, SAM2_REPO
sys.path.append(ANIMATE_ANYONE_REPO)
sys.path.append(SAM2_REPO)

import os, logging, cv2
import torch
import numpy as np

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

from sam2.build_sam import build_sam2_video_predictor

from mimo.training.training_utils import get_torch_weight_dtype
from mimo.inference.inference_utils import create_video_from_frames, remove_tmp_dir

from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.depth_anything_v2_utils import DepthBatchPredictor
from mimo.utils.vae_encoding_utils import download_vae, VaeBatchPredictor
from mimo.utils.detectron2_utils import DetectronBatchPredictor
from mimo.utils.propainter_utils import ProPainterBatchPredictor

from mimo.dataset_preprocessing.video_sampling_resizing import sampling_resizing
from mimo.dataset_preprocessing.depth_estimation import DEPTH_ANYTHING_MODEL_CONFIGS, get_depth
from mimo.dataset_preprocessing.human_detection_detectron2 import get_cfg_settings, post_processing_detectron2
from mimo.dataset_preprocessing.video_tracking_sam2 import get_instance_sam_output, process_layers
from mimo.dataset_preprocessing.video_inpainting import inpaint_frames

class InferencePipeline():
    def __init__(  # See default values in inference/main.py
            self, 
            seed, 
            weight_dtype, 
            num_workers,
            input_net_size,
            input_net_fps,
            depth_anything_encoder,
            score_threshold_detectron2,
            batch_size_depth,
            batch_size_detectron2,
            batch_size_propainter,
        ):
        self.num_workers = num_workers
        self.input_net_size = input_net_size
        self.input_net_fps = input_net_fps
        self.depth_anything_encoder = depth_anything_encoder
        self.score_threshold_detectron2 = score_threshold_detectron2
        self.batch_size_depth = batch_size_depth
        self.batch_size_detectron2 = batch_size_detectron2
        self.batch_size_propainter = batch_size_propainter

        self.infer_cfg = OmegaConf.load("./mimo/configs/inference/inference.yaml")

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
    
    def get_layers_sam2(self, resized_frames, detectron2_output, depth_frames):
        checkpoint = os.path.join(CHECKPOINTS_FOLDER, "sam2.1_hiera_large.pt")
        model_cfg = os.path.join(SAM2_REPO, "configs/sam2.1/sam2.1_hiera_l.yaml")
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)

        predictor = self.accelerator.prepare(predictor)

        input_video_path = create_video_from_frames(resized_frames, self.input_net_fps)

        (
            min_frame_idx, 
            max_frame_idx, 
            instance_sam_output, 
            foreground_mask
        ) = get_instance_sam_output(input_video_path, detectron2_output, depth_frames, predictor, self.score_threshold_detectron2)

        remove_tmp_dir(input_video_path)

        return process_layers(resized_frames, 
                                instance_sam_output, 
                                foreground_mask, 
                                min_frame_idx, 
                                max_frame_idx)
    
    def inpaint_scene_layer(self, scene_frames):
        predictor = ProPainterBatchPredictor(self.batch_size_propainter, self.num_workers)

        return inpaint_frames(scene_frames, predictor)

    def __call__(self, input_video_path, output_video_path):
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

        resized_frames = np.array(sampling_resizing(frame_gen, 
                                           frames_per_second, 
                                           output_fps=self.input_net_fps, 
                                           input_size=(width, height), 
                                           output_width=self.input_net_size))
        print("np.shape(resized_frames)", np.shape(resized_frames), type(resized_frames))

        depth_frames = np.concatenate(self.depth_estimation(resized_frames))
        print("np.shape(depth_frames)", np.shape(depth_frames), type(depth_frames))

        detectron2_output = self.get_detectron2_output(resized_frames)
        print("np.shape(detectron2_output['data_frame_index'])", np.shape(detectron2_output['data_frame_index']), type(detectron2_output['data_frame_index']))
        print("np.shape(detectron2_output['data_pred_boxes'])", np.shape(detectron2_output['data_pred_boxes']), type(detectron2_output['data_pred_boxes']))
        print("np.shape(detectron2_output['data_scores'])", np.shape(detectron2_output['data_scores']), type(detectron2_output['data_scores']))
        print("np.shape(detectron2_output['data_pred_classes'])", np.shape(detectron2_output['data_pred_classes']), type(detectron2_output['data_pred_classes']))
        print("np.shape(detectron2_output['data_pred_masks'])", np.shape(detectron2_output['data_pred_masks']), type(detectron2_output['data_pred_masks']))

        human_frames, occlusion_frames, scene_frames = self.get_layers_sam2(resized_frames, detectron2_output, depth_frames)
        human_frames = np.array(human_frames)
        occlusion_frames = np.array(occlusion_frames)
        scene_frames = np.array(scene_frames)
        print("np.shape(human_frames)", np.shape(human_frames), type(human_frames))
        print("np.shape(occlusion_frames)", np.shape(occlusion_frames), type(occlusion_frames))
        print("np.shape(scene_frames)", np.shape(scene_frames), type(scene_frames))

        scene_frames = np.array(self.inpaint_scene_layer(scene_frames))
        print("np.shape(scene_frames)", np.shape(scene_frames), type(scene_frames))

        video.release()