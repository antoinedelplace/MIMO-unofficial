import torch
import torch.nn as nn

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latent_video,
        timesteps,
        latent_apose,
        a_pose_clip,
        rast_2d_joints,
        latents_scene,
        latents_occlusion,
        uncond_fwd: bool = False,
    ):
        # print("noisy_latent_video", noisy_latent_video.shape)
        # print("timesteps", timesteps.shape)
        # print("latent_apose", latent_apose.shape)
        # print("a_pose_clip", a_pose_clip.shape)
        # print("rast_2d_joints", rast_2d_joints.shape)
        # print("latents_scene", latents_scene.shape)
        # print("latents_occlusion", latents_occlusion.shape)

        rast_2d_joints = rast_2d_joints.transpose(1, 2)  # (b, c, f, h, w)
        pose_features = self.pose_guider(rast_2d_joints)  # (b, c, f, h, w)

        # print("pose_features", pose_features.shape)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            encoder_hidden_states = a_pose_clip.unsqueeze(1) # (b, 1, d)
            self.reference_unet(
                latent_apose, # (b, c, h, w)
                ref_timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        else:
            encoder_hidden_states = torch.zeros_like(a_pose_clip).unsqueeze(1) # (b, 1, d)

        latents_scene = latents_scene.transpose(1, 2)  # (b, c, f, h, w)
        latents_occlusion = latents_occlusion.transpose(1, 2)  # (b, c, f, h, w)
        noisy_latent_video = torch.cat((noisy_latent_video, latents_scene, latents_occlusion), dim=1)  # (b, c+c+c, f, h, w)

        model_pred = self.denoising_unet(
            noisy_latent_video,
            timesteps,
            pose_cond_fea=pose_features,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        return model_pred