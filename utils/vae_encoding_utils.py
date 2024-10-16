import torch, os
from torch.utils.data import DataLoader
import numpy as np

from diffusers import AutoencoderKL

from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

from utils.torch_utils import VideoDataset, NpzDataset


def download_vae(checkpoints_folder):
    local_dir = os.path.join(checkpoints_folder, "sd-vae-ft-mse")
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["diffusion_pytorch_model.safetensors", "config.json"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

class VaeBatchPredictor():
    def __init__(
        self, 
        batch_size: int, 
        workers: int,
        checkpoints_folder
    ):
        self.vae = AutoencoderKL.from_pretrained(
            os.path.join(checkpoints_folder, "sd-vae-ft-mse"), 
            torch_dtype=torch.float16,
            revision="fp16",
            use_safetensors=True
        ).to("cuda")
        self.batch_size = batch_size
        self.workers = workers

    def encode_collate(self, batch):
        data = []
        for image in batch:
            # the model expects RGB inputs
            image = image[:, :, ::-1] / 255.0 * 2 - 1

            image = image.astype("float16").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append(image)
        return torch.stack(data, dim=0)
    
    def decode_collate(self, batch):
        data = []
        for latent in batch:
            latent = latent / 0.18215
            latent = torch.as_tensor(latent.astype("float16"))
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
                yield latent.cpu().numpy()
    
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
                image = ((image + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                image = image.permute(0, 2, 3, 1)
                image = image[:, :, :, [2, 1, 0]] #RGB 2 BGR

                yield image.cpu().numpy()