import sys
sys.path.append(".")

import torch
import numpy as np
from torch.utils.data import Dataset

def collate_fn_rast(batch, weight_dtype):
    data_rast_2d_joints = []
    for rast_2d_joints in batch:
        # the model expects RGB inputs
        rast_2d_joints = rast_2d_joints[:, :, ::-1] / 255.0
        rast_2d_joints = rast_2d_joints.transpose(2, 0, 1)
        rast_2d_joints = torch.as_tensor(rast_2d_joints, dtype=weight_dtype)
        data_rast_2d_joints.append(rast_2d_joints)

    return torch.stack(data_rast_2d_joints, dim=0)

def collate_fn(batch, weight_dtype):
    data_pose_encoded = []
    data_scene_encoded = []
    data_occlusion_encoded = []
    for latent_pose, latents_scene, latents_occlusion in batch:
        data_pose_encoded.append(torch.as_tensor(latent_pose, dtype=weight_dtype))
        data_scene_encoded.append(torch.as_tensor(latents_scene, dtype=weight_dtype))
        data_occlusion_encoded.append(torch.as_tensor(latents_occlusion, dtype=weight_dtype))

    return (
        torch.stack(data_pose_encoded, dim=0),
        torch.stack(data_scene_encoded, dim=0),
        torch.stack(data_occlusion_encoded, dim=0)
    )

def mirror_padding(array, start, end, window_length):
    # Slice the array and check if it's shorter than window_length
    segment = array[start:end]
    if len(segment) < window_length:
        # Mirror the last values to reach the batch size
        repeats = (window_length - len(segment)) // len(segment) + 1
        segment = np.concatenate([segment] + [segment[-1:]] * repeats)[:window_length]
    return segment

class InferenceDataset(Dataset):
    def __init__(self, rast_2d_joints, latents_scene, latents_occlusion, window_length, window_stride):
        super().__init__()
        
        self.rast_2d_joints = rast_2d_joints
        self.latents_scene = latents_scene
        self.latents_occlusion = latents_occlusion

        self.window_length = window_length
        self.window_stride = window_stride

        self.num_frames = len(self.rast_2d_joints)

        self.num_windows = ((max(0, self.num_frames - self.window_length) + self.window_stride - 1) // self.window_stride) + 1

    def __getitem__(self, index):
        start = index * self.window_stride
        end = start + self.window_length

        if end > self.num_frames:
            start = self.num_frames - self.window_length
            end = self.num_frames

        rast_2d = mirror_padding(self.rast_2d_joints, start, end, self.window_length)
        latents_scene = mirror_padding(self.latents_scene, start, end, self.window_length)
        latents_occlusion = mirror_padding(self.latents_occlusion, start, end, self.window_length)

        return rast_2d, latents_scene, latents_occlusion

    def __len__(self):
        return self.num_windows