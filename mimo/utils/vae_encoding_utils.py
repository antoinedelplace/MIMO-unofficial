import sys
sys.path.append(".")

import torch, os
from torch.utils.data import DataLoader
import numpy as np

from diffusers import AutoencoderKL

from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

from mimo.utils.torch_utils import VideoDataset, NpzDataset

from mimo.configs.paths import VAE_FOLDER


def download_vae():
    os.makedirs(VAE_FOLDER, exist_ok=True)
    for hub_file in ["diffusion_pytorch_model.safetensors", "config.json"]:
        path = Path(hub_file)
        saved_path = VAE_FOLDER / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=VAE_FOLDER,
        )

class VaeBatchPredictor():
    def __init__(
        self, 
        batch_size: int, 
        workers: int,
    ):
        self.vae = AutoencoderKL.from_pretrained(
            VAE_FOLDER, 
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to("cuda")
        self.batch_size = batch_size
        self.workers = workers

    def encode_collate(self, batch):
        data = []
        for image in batch:
            # the model expects RGB inputs
            image = image[:, :, ::-1] / 255.0 * 2 - 1

            image = image.transpose(2, 0, 1)
            image = torch.as_tensor(image, dtype=torch.bfloat16)
            data.append(image)
        return torch.stack(data, dim=0)
    
    def decode_collate(self, batch):
        data = []
        for latent in batch:
            latent = latent / 0.18215
            latent = torch.as_tensor(latent, dtype=torch.bfloat16)
            data.append(latent)
        return torch.stack(data, dim=0)

    def encode(self, frame_gen):
        dataset = VideoDataset(frame_gen)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.encode_collate,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in loader:
                batch_gpu = batch.to("cuda")
                latent = self.vae.encode(batch_gpu)
                latent = 0.18215 * latent.latent_dist.mean
                yield latent.cpu().float().numpy()
    
    def decode(self, nparray_data):
        dataset = NpzDataset(nparray_data)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.decode_collate,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in loader:
                batch_gpu = batch.to("cuda")
                image = self.vae.decode(batch_gpu).sample
                image = ((image + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
                image = image.permute(0, 2, 3, 1)
                image = image[:, :, :, [2, 1, 0]] #RGB 2 BGR

                yield image.cpu().numpy()