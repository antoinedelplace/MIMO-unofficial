import sys
sys.path.append(".")

from mimo.configs.paths import MIMIC_MOTION_REPO
sys.path.append(MIMIC_MOTION_REPO)

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cog import Input

from predict import Predictor
from utils.loader import create_pipeline
from dwpose.preprocess import get_image_pose
from dwpose.util import draw_pose

from torchvision.transforms.functional import pil_to_tensor, resize, center_crop

# w/h aspect ratio
ASPECT_RATIO = 9 / 16

class ReposerPredictor(Predictor):
    def __init__(self, dw_pose_detector):
        super().__init__(ReposerPredictor)
        self.dw_pose_detector = dw_pose_detector

    def input_validation(self, resolution, num_frames, frames_overlap, num_inference_steps, noise_aug_strength, guidance_scale, sample_stride):
        if resolution % 8 != 0:
            raise ValueError(f"Resolution must be a multiple of 8, got {resolution}")

        if resolution < 64 or resolution > 1024:
            raise ValueError(
                f"Resolution must be between 64 and 1024, got {resolution}"
            )

        if num_frames <= frames_overlap:
            raise ValueError(
                f"Number of frames ({num_frames}) must be greater than frames overlap ({frames_overlap})"
            )

        if num_frames < 2:
            raise ValueError(f"Number of frames must be at least 2, got {num_frames}")

        if frames_overlap < 0:
            raise ValueError(
                f"Frames overlap must be non-negative, got {frames_overlap}"
            )

        if num_inference_steps < 1 or num_inference_steps > 100:
            raise ValueError(
                f"Number of inference steps must be between 1 and 100, got {num_inference_steps}"
            )

        if noise_aug_strength < 0.0 or noise_aug_strength > 1.0:
            raise ValueError(
                f"Noise augmentation strength must be between 0.0 and 1.0, got {noise_aug_strength}"
            )

        if guidance_scale < 0.1 or guidance_scale > 10.0:
            raise ValueError(
                f"Guidance scale must be between 0.1 and 10.0, got {guidance_scale}"
            )

        if sample_stride < 1:
            raise ValueError(f"Sample stride must be at least 1, got {sample_stride}")

    def update_pipeline(self, use_fp16):
        need_pipeline_update = False
        checkpoint_version = "v1-1"

        # Check if we need to switch checkpoints
        if checkpoint_version != self.current_checkpoint:
            if checkpoint_version == "v1":
                self.config.ckpt_path = "models/MimicMotion.pth"
            else:  # v1-1
                self.config.ckpt_path = "models/MimicMotion_1-1.pth"
            need_pipeline_update = True
            self.current_checkpoint = checkpoint_version

        # Check if we need to switch dtype
        target_dtype = torch.float16 if use_fp16 else torch.float32
        if target_dtype != self.current_dtype:
            torch.set_default_dtype(target_dtype)
            need_pipeline_update = True
            self.current_dtype = target_dtype

        # Update pipeline if needed
        if need_pipeline_update:
            print(
                f"Updating pipeline with checkpoint: {self.config.ckpt_path} and dtype: {torch.get_default_dtype()}"
            )
            self.pipeline = create_pipeline(self.config, self.device)

    def get_video_pose(self, video_frames, ref_image):
        # select ref-keypoint from reference pose for pose rescale
        _, _, ref_pose = self.dw_pose_detector(ref_image)
        ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ref_keypoint_id = [i for i in ref_keypoint_id \
            if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
        ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

        height, width, _ = ref_image.shape

        # read input video
        video_frames = np.array(list(video_frames))
        detected_poses = []
        for frame in tqdm(video_frames, desc="DWPose"):
            input_image_pil = Image.fromarray(frame[:, :, ::-1])
            _, _, points_2d = self.dw_pose_detector(input_image_pil)
            detected_poses.append(points_2d)
        self.dw_pose_detector.release_memory()

        detected_bodies = np.stack(
            [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                        ref_keypoint_id]
        # compute linear-rescale params
        ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
        fh, fw, _ = video_frames[0].shape
        ax = ay / (fh / fw / height * width)
        bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
        a = np.array([ax, ay])
        b = np.array([bx, by])
        output_pose = []
        # pose rescale 
        for detected_pose in detected_poses:
            detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
            detected_pose['faces'] = detected_pose['faces'] * a + b
            detected_pose['hands'] = detected_pose['hands'] * a + b
            im = draw_pose(detected_pose, height, width)
            output_pose.append(np.array(im))
        return np.stack(output_pose)

    def get_image_pixels(self, appearance_image, resolution):
        image_pixels = Image.fromarray(appearance_image[:, :, ::-1]).convert("RGB")
        image_pixels = pil_to_tensor(image_pixels)  # (c, h, w)
        h, w = image_pixels.shape[-2:]

        if h > w:
            w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
        else:
            w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution

        h_w_ratio = float(h) / float(w)
        if h_w_ratio < h_target / w_target:
            h_resize, w_resize = h_target, int(h_target / h_w_ratio)
        else:
            h_resize, w_resize = int(w_target * h_w_ratio), w_target

        image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
        image_pixels = center_crop(image_pixels, [h_target, w_target])
        image_pixels = image_pixels.permute((1, 2, 0)).numpy()

        return image_pixels
    
    def predict(
        self,
        motion_frames,
        appearance_image,
        resolution: int = Input(
            description="Height of the output video in pixels. Width is automatically calculated.",
            default=576,
            ge=64,
            le=1024,
        ),
        num_frames: int = Input(
            description="Number of frames to generate in each processing chunk",
            default=16,
            ge=2,
        ),
        frames_overlap: int = Input(
            description="Number of overlapping frames between chunks for smoother transitions",
            default=6,
            ge=0,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps in the diffusion process. More steps can improve quality but increase processing time.",
            default=25,
            ge=1,
            le=100,
        ),
        noise_aug_strength: float = Input(
            description="Strength of noise augmentation. Higher values add more variation but may reduce coherence with the reference.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        guidance_scale: float = Input(
            description="Strength of guidance towards the reference. Higher values adhere more closely to the reference but may reduce creativity.",
            default=2.0,
            ge=0.1,
            le=10.0,
        ),
        sample_stride: int = Input(
            description="Interval for sampling frames from the reference video. Higher values skip more frames.",
            default=2,
            ge=1,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
    ):
        use_fp16 = True

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        self.update_pipeline(use_fp16)

        self.input_validation(resolution, num_frames, frames_overlap, num_inference_steps, noise_aug_strength, guidance_scale, sample_stride)

        image_pixels = self.get_image_pixels(appearance_image, resolution)

        image_pose = get_image_pose(image_pixels)
        video_pose = self.get_video_pose(motion_frames, image_pixels)

        pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
        image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))

        pose_pixels = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1,
        image_pixels = torch.from_numpy(image_pixels) / 127.5 - 1,

        video_frames = self.run_pipeline(
            image_pixels,
            pose_pixels,
            num_frames=num_frames,
            frames_overlap=frames_overlap,
            num_inference_steps=num_inference_steps,
            noise_aug_strength=noise_aug_strength,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return video_frames