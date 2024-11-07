import torch
import numpy as np
from PIL import Image

from diffusers import FluxImg2ImgPipeline


class UpscalerPredictor():
    def __init__(self, num_inference_steps, denoise_strength, cfg_guidance_scale, seed):
        self.num_inference_steps = num_inference_steps
        self.denoise_strength = denoise_strength
        self.cfg_guidance_scale = cfg_guidance_scale
        self.seed = seed

        self.pipeline = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        self.pipeline.enable_sequential_cpu_offload()

    def __call__(self, input_image, prompt):
        control_image = Image.fromarray(input_image[:, :, ::-1])

        image = self.pipeline(
            prompt=prompt, 
            image=control_image, 
            num_inference_steps=self.num_inference_steps, 
            strength=self.denoise_strength, 
            guidance_scale=self.cfg_guidance_scale, 
            generator=torch.manual_seed(self.seed)
        ).images[0]

        return np.array(image)[:, :, ::-1]