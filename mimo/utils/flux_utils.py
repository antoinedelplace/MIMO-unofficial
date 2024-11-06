import torch
from PIL import Image

from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel


class ReposerPredictor():
    def __init__(self):
        controlnet_union = FluxControlNetModel.from_pretrained('Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro', torch_dtype=torch.bfloat16)
        controlnet = FluxMultiControlNetModel([controlnet_union])

        self.pipeline = FluxControlNetPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', controlnet=controlnet, torch_dtype=torch.bfloat16)

        # self.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        # self.pipeline.set_ip_adapter_scale(0.6)

        self.pipeline.enable_sequential_cpu_offload(gpu_id=1)

    def __call__(self, reference_image, pose_image):
        control_image = Image.fromarray(pose_image[:, :, ::-1])
        prompt = ""

        width, height = control_image.size

        image = self.pipeline(
            prompt,
            control_image=[control_image],
            control_mode=[4],  # canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6)
            # width=width,
            # height=height,
            controlnet_conditioning_scale=[0.8],  # recommended controlnet_conditioning_scale is 0.3-0.8
            num_inference_steps=24,
            guidance_scale=3.5,
            generator=torch.manual_seed(42),
        ).images[0]

        return image