import os
import torch

from collections import OrderedDict

def get_torch_weight_dtype(str_weight_dtype):
    if str_weight_dtype == "fp16":
        return torch.float16
    elif str_weight_dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {str_weight_dtype} during training"
        )

def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def save_checkpoint(model, save_dir, modules_to_save, ckpt_num, logger, total_limit=None):
    for prefix in modules_to_save:
        save_path = os.path.join(save_dir, f"{prefix}-{ckpt_num}.pth")

        if total_limit is not None:
            checkpoints = os.listdir(save_dir)
            checkpoints = [d for d in checkpoints if d.startswith(prefix)]
            checkpoints = sorted(
                checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
            )

            if len(checkpoints) >= total_limit:
                num_to_remove = len(checkpoints) - total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]
                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                    os.remove(removing_checkpoint)

        mm_state_dict = OrderedDict()
        state_dict = model.state_dict()
        for key in state_dict:
            if prefix in key:
                mm_state_dict[key] = state_dict[key]

        torch.save(mm_state_dict, save_path)

def get_last_checkpoint(folder):
    dirs = os.listdir(folder)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))

    if len(dirs) == 0:
        raise Exception(f"No checkpoint-xxx folder found in {folder}")
    path = dirs[-1]
    global_step = int(path.split("-")[1])

    return path, global_step

def freeze_top_layer_reference_unet(reference_unet):
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    
    return reference_unet

def unfreeze_motion_module(denoising_unet):
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True
    
    return denoising_unet