import torch, os
from torch.utils.data import DataLoader
import numpy as np

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

from utils.torch_utils import VideoDataset


def download_image_encoder(checkpoints_folder):
    local_dir = os.path.join(checkpoints_folder, "sd-image-variations-diffusers")
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )
   

class CLIPBatchPredictor():
    def __init__(
        self, 
        batch_size: int, 
        workers: int,
        checkpoints_folder
    ):
        self.clip_image_processor = CLIPImageProcessor()
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(checkpoints_folder, "sd-image-variations-diffusers"), 
            torch_dtype=torch.float16,
            revision="fp16"
        ).to("cuda")
        self.batch_size = batch_size
        self.workers = workers

    def collate(self, batch):
        return self.clip_image_processor.preprocess(batch,
            return_tensors="pt",
        ).pixel_values

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
                embeds = self.vae(batch_gpu).image_embeds
                yield embeds.cpu().numpy()