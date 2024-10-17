import sys
sys.path.append(".")
sys.path.append("../AnimateAnyone")

import os
from pathlib import Path, PurePosixPath
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from diffusers import DDIMScheduler

from huggingface_hub import hf_hub_download

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.pose_guider import PoseGuider
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline

from utils.general_utils import argmedian
from utils.video_utils import frame_from_video
from utils.torch_utils import VideoDataset

def download_base_model(checkpoints_folder):
    local_dir = os.path.join(checkpoints_folder, "stable-diffusion-v1-5")
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["unet/diffusion_pytorch_model.safetensors", "unet/config.json"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

def download_anyone(checkpoints_folder):
    local_dir = os.path.join(checkpoints_folder, "AnimateAnyone")
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "denoising_unet.pth",
        "motion_module.pth",
        "pose_guider.pth",
        "reference_unet.pth",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="novita-ai/AnimateAnyone",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

def get_frame_with_median_mask(video, frame_gen):
    threshold = 0.1*255
    mask = np.any(np.array(list(frame_gen)) > threshold, axis=-1)
    area_mask = np.sum(mask, axis=(1, 2))
    # print("np.max(area_mask)", np.max(area_mask))
    # print("np.median(area_mask)", np.median(area_mask))
    # print("np.min(area_mask)", np.min(area_mask))
    index = argmedian(area_mask)
    # print("index", index)

    return frame_from_video(video, index)


class ReposerBatchPredictor():
    def __init__(
        self, 
        batch_size: int, 
        workers: int,
        checkpoints_folder,
        clip,
        vae
    ):
        infer_config = OmegaConf.load("./configs/inference/inference.yaml")

        pose_guider = PoseGuider(conditioning_embedding_channels=320).to(
            dtype=torch.float16, device="cuda"
        )
        pose_guider.load_state_dict(
            torch.load(os.path.join(checkpoints_folder, "AnimateAnyone", "pose_guider.pth"), map_location="cpu", weights_only=True),
        )
        
        reference_unet = UNet2DConditionModel.from_pretrained(
            os.path.join(checkpoints_folder, "stable-diffusion-v1-5", "unet"),
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        reference_unet.load_state_dict(
            torch.load(os.path.join(checkpoints_folder, "AnimateAnyone", "reference_unet.pth"), map_location="cpu", weights_only=True),
        )

        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            os.path.join(checkpoints_folder, "stable-diffusion-v1-5", "unet"),
            os.path.join(checkpoints_folder, "AnimateAnyone", "motion_module.pth"),
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(torch.float16).to("cuda")
        denoising_unet.load_state_dict(
            torch.load(os.path.join(checkpoints_folder, "AnimateAnyone", "denoising_unet.pth"), map_location="cpu", weights_only=True),
            strict=False,
        )

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        self.pipe = Pose2VideoPipeline(
            vae=vae.vae,
            image_encoder=clip.image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        ).to("cuda", dtype=torch.float16)

        self.batch_size = batch_size
        self.workers = workers
    
    def collate(self, batch):
        data_pose_image_pil = []
        for pose_frame in batch:
            data_pose_image_pil.append(Image.fromarray(pose_frame).convert("RGB"))
        return data_pose_image_pil

    def __call__(self, reference_image, pose_frame_gen, seed=12345):
        cfg = 3.5
        steps = 30

        dataset = VideoDataset(pose_frame_gen)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.collate,
            pin_memory=True
        )

        reference_image_pil = Image.fromarray(reference_image).convert("RGB")

        with torch.no_grad():
            for batch_pose_images_pil in loader:
                video = self.pipe(
                    reference_image_pil,
                    batch_pose_images_pil,
                    width=768,
                    height=768,
                    video_length=len(batch_pose_images_pil),
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=torch.manual_seed(seed),
                ).videos

                video = video.squeeze(2)
                video = (video * 255).round().clamp(0, 255).to(torch.uint8)
                video = video.permute(0, 2, 3, 1)
                video = video[:, :, :, [2, 1, 0]] #RGB 2 BGR

                yield video.cpu().numpy()