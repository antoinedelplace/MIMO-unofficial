import sys
sys.path.append(".")

from mimo.configs.paths import BASE_MODEL_FOLDER, ANIMATE_ANYONE_FOLDER, DWPOSE_FOLDER, CHECKPOINTS_FOLDER, ANIMATE_ANYONE_REPO
sys.path.append(ANIMATE_ANYONE_REPO)

import os, cv2
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
from src.dwpose import DWposeDetector
from src.dwpose.wholebody import Wholebody

from mimo.utils.general_utils import argmedian
from mimo.utils.video_utils import frame_from_video
from mimo.utils.torch_utils import VideoDataset

def download_base_model():
    os.makedirs(BASE_MODEL_FOLDER, exist_ok=True)
    for hub_file in ["unet/diffusion_pytorch_model.safetensors", "unet/config.json"]:
        path = Path(hub_file)
        saved_path = BASE_MODEL_FOLDER / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=BASE_MODEL_FOLDER,
        )

def download_anyone():
    os.makedirs(ANIMATE_ANYONE_FOLDER, exist_ok=True)
    for hub_file in [
        "denoising_unet.pth",
        "motion_module.pth",
        "pose_guider.pth",
        "reference_unet.pth",
    ]:
        path = Path(hub_file)
        saved_path = ANIMATE_ANYONE_FOLDER / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="novita-ai/AnimateAnyone",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=ANIMATE_ANYONE_FOLDER,
        )

def download_dwpose():
    os.makedirs(DWPOSE_FOLDER, exist_ok=True)
    for hub_file in [
        "dw-ll_ucoco_384.onnx",
        "yolox_l.onnx",
    ]:
        path = Path(hub_file)
        saved_path = DWPOSE_FOLDER / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="yzd-v/DWPose",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=DWPOSE_FOLDER,
        )

def get_frame_with_median_mask(video, frame_gen):
    threshold = 0.1*255
    mask = np.any(np.array(list(frame_gen)) > threshold, axis=-1)
    area_mask = np.sum(mask, axis=(1, 2))
    index = argmedian(area_mask)

    return frame_from_video(video, index)

def get_frame_closest_pose(video, frame_gen, ref_points_2d):
    i_chosen = 0
    distance = np.infty

    detector = CustomDWposeDetector()
    detector = detector.to("cuda")

    ref_points_2d_norm = (ref_points_2d['bodies']['candidate']-np.mean(ref_points_2d['bodies']['candidate']))/np.std(ref_points_2d['bodies']['candidate'])

    for i_frame, image in enumerate(frame_gen):
        input_image_pil = Image.fromarray(image[:, :, ::-1])
        _, _, points_2d = detector(input_image_pil)

        points_2d_norm = (points_2d['bodies']['candidate']-np.mean(points_2d['bodies']['candidate']))/np.std(points_2d['bodies']['candidate'])
        new_distance = np.sum(np.abs(points_2d_norm-ref_points_2d_norm))
        if new_distance < distance:
            distance = new_distance
            i_chosen = i_frame
    
    return frame_from_video(video, i_chosen)

def get_kps_image(input_image_path):
    detector = CustomDWposeDetector()
    detector = detector.to("cuda")

    input_image = cv2.imread(input_image_path)
    input_image_pil = Image.fromarray(input_image[:, :, ::-1])

    result_pil, score, points_2d = detector(input_image_pil)

    return np.array(result_pil), points_2d

class CustomDWposeDetector(DWposeDetector):
    def to(self, device):
        self.pose_estimation = Wholebody(device, pathPrefix=Path(CHECKPOINTS_FOLDER))
        return self

class ReposerBatchPredictor():
    def __init__(
        self, 
        batch_size: int, 
        workers: int,
        clip,
        vae
    ):
        infer_config = OmegaConf.load("./mimo/configs/inference/inference.yaml")

        pose_guider = PoseGuider(conditioning_embedding_channels=320).to(
            dtype=torch.float16, device="cuda"
        )
        pose_guider.load_state_dict(
            torch.load(os.path.join(ANIMATE_ANYONE_FOLDER, "pose_guider.pth"), map_location="cpu", weights_only=True),
        )
        
        reference_unet = UNet2DConditionModel.from_pretrained(
            os.path.join(BASE_MODEL_FOLDER, "unet"),
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        reference_unet.load_state_dict(
            torch.load(os.path.join(ANIMATE_ANYONE_FOLDER, "reference_unet.pth"), map_location="cpu", weights_only=True),
        )

        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            os.path.join(BASE_MODEL_FOLDER, "unet"),
            os.path.join(ANIMATE_ANYONE_FOLDER, "motion_module.pth"),
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(torch.float16).to("cuda")
        denoising_unet.load_state_dict(
            torch.load(os.path.join(ANIMATE_ANYONE_FOLDER, "denoising_unet.pth"), map_location="cpu", weights_only=True),
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
            data_pose_image_pil.append(Image.fromarray(pose_frame[:, :, ::-1]))
        return data_pose_image_pil

    def __call__(self, reference_image, pose_frame_gen, seed=12345):
        cfg = 3.5 #5
        steps = 30 #60

        dataset = VideoDataset(pose_frame_gen)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.collate,
            pin_memory=True
        )

        reference_image_pil = Image.fromarray(reference_image[:, :, ::-1])

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

                video = video.squeeze(0)
                video = (video * 255).round().clamp(0, 255).to(torch.uint8)
                video = video.permute(1, 2, 3, 0)
                video = video[:, :, :, [2, 1, 0]] #RGB 2 BGR

                yield video.cpu().numpy()