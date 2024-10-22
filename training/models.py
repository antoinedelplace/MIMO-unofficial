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
        apose,
        a_pose_clip,
        rast_2d_joints,
        latents_scene,
        latents_occlusion,
        uncond_fwd: bool = False,
    ):
        print("noisy_latent_video", noisy_latent_video.shape)
        print("timesteps", timesteps.shape)
        print("apose", apose.shape)
        print("a_pose_clip", a_pose_clip.shape)
        print("rast_2d_joints", rast_2d_joints.shape)
        print("latents_scene", latents_scene.shape)
        print("latents_occlusion", latents_occlusion.shape)

        rast_2d_joints = rast_2d_joints.to(device="cuda")
        pose_features = self.pose_guider(rast_2d_joints)

        encoder_hidden_states = torch.cat((latents_scene, latents_occlusion))

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                apose,
                ref_timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latent_video,
            timesteps,
            pose_cond_fea=pose_features,
            encoder_hidden_states=a_pose_clip,
        ).sample

        return model_pred