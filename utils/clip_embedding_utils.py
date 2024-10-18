import sys
sys.path.append(".")

import torch, os
from torch.utils.data import DataLoader
import numpy as np

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

from utils.torch_utils import VideoDataset

from configs.paths import IMAGE_ENCODER_FOLDER


def download_image_encoder():
    os.makedirs(IMAGE_ENCODER_FOLDER, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = IMAGE_ENCODER_FOLDER / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=IMAGE_ENCODER_FOLDER,
        )
   

class CLIPBatchPredictor():
    def __init__(
        self, 
        batch_size: int, 
        workers: int,
    ):
        self.clip_image_processor = CLIPImageProcessor()
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(IMAGE_ENCODER_FOLDER, "image_encoder"), 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        self.batch_size = batch_size
        self.workers = workers

    def collate(self, batch):
        return self.clip_image_processor.preprocess(batch,
            return_tensors="pt",
            do_convert_rgb=True
        ).pixel_values.to(torch.bfloat16)

    def __call__(self, frame_gen):
        dataset = VideoDataset(frame_gen)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.collate,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in loader:
                batch_gpu = batch.to("cuda")
                embeds = self.image_enc(batch_gpu).image_embeds
                yield embeds.cpu().float().numpy()