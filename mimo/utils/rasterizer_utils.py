import sys
sys.path.append(".")

from mimo.configs.paths import NVDIFFRAST_REPO
sys.path.append(NVDIFFRAST_REPO)

import numpy as np
import torch
from torch.utils.data import DataLoader
import scipy.spatial

import nvdiffrast.torch as dr

from mimo.utils.torch_utils import NpzDataset

def get_triangles_from_2d_joints(data_joints_2d):
    mean_points_2d = np.mean(data_joints_2d[:, :, :2], axis=0)
    tri = scipy.spatial.Delaunay(mean_points_2d)
    triangles = torch.from_numpy(tri.simplices).to(torch.int32).cuda()

    return triangles

def get_vertex_attrs(n_joints):
    vertex_attrs = torch.zeros(n_joints, 3, device='cuda')

    # Fill the joint_ids tensor with unique values for each joint
    for i in range(n_joints):
        vertex_attrs[i, 0] = 1.0 - (i % 4) / 3.0
        vertex_attrs[i, 1] = 1.0 - ((i // 4) % 4) / 3.0
        vertex_attrs[i, 2] = 1.0 - ((i // 16) % 16) / 3.0
    
    return vertex_attrs

triangles = torch.tensor([[10, 25, 23],
        [ 2, 33, 41],
        [17,  2, 32],
        [ 2, 17, 33],
        [ 2,  3, 32],
        [ 3,  2, 41],
        [ 6, 28, 41],
        [ 6,  5, 35],
        [ 5,  6, 41],
        [28, 39, 41],
        [19, 21, 20],
        [19, 22, 21],
        [18, 38, 43],
        [33, 40, 41],
        [40,  5, 41],
        [ 5, 34, 18],
        [40, 34,  5],
        [34, 40, 37],
        [17,  1, 33],
        [ 1, 40, 33],
        [40,  1, 37],
        [ 4, 26, 10],
        [ 4, 10, 23],
        [ 4, 23, 32],
        [ 3,  4, 32],
        [ 4,  3, 31],
        [27,  3, 41],
        [39, 27, 41],
        [ 3, 27, 31],
        [ 8, 39, 28],
        [ 8, 27, 39],
        [30, 25, 10],
        [12, 29, 44],
        [12,  8, 28],
        [20,  7, 35],
        [29,  7, 20],
        [ 7, 12, 28],
        [12,  7, 29],
        [30, 11, 25],
        [25, 11, 23],
        [34, 42, 18],
        [42, 34, 37],
        [ 1, 42, 37],
        [ 0, 42, 17],
        [42,  1, 17],
        [16, 38, 18],
        [42, 16, 18],
        [16, 42,  0],
        [15,  0, 17],
        [15, 17, 43],
        [38, 15, 43],
        [16, 15, 38],
        [15, 16,  0],
        [ 9,  4, 31],
        [27,  9, 31],
        [ 8,  9, 27],
        [26,  9, 44],
        [ 4,  9, 26],
        [ 9, 12, 44],
        [12,  9,  8],
        [21, 14, 20],
        [14, 30, 20],
        [11, 14, 21],
        [14, 11, 30],
        [13, 29, 20],
        [30, 13, 20],
        [13, 30, 10],
        [26, 13, 10],
        [13, 26, 44],
        [29, 13, 44],
        [36,  6, 35],
        [ 7, 36, 35],
        [ 6, 36, 28],
        [36,  7, 28],
        [22, 24, 21],
        [24, 11, 21],
        [24, 22, 23],
        [11, 24, 23]], device="cuda", dtype=torch.int32)

vertex_attrs = torch.tensor([[1.0000, 1.0000, 1.0000],
        [0.6667, 1.0000, 1.0000],
        [0.3333, 1.0000, 1.0000],
        [0.0000, 1.0000, 1.0000],
        [1.0000, 0.6667, 1.0000],
        [0.6667, 0.6667, 1.0000],
        [0.3333, 0.6667, 1.0000],
        [0.0000, 0.6667, 1.0000],
        [1.0000, 0.3333, 1.0000],
        [0.6667, 0.3333, 1.0000],
        [0.3333, 0.3333, 1.0000],
        [0.0000, 0.3333, 1.0000],
        [1.0000, 0.0000, 1.0000],
        [0.6667, 0.0000, 1.0000],
        [0.3333, 0.0000, 1.0000],
        [0.0000, 0.0000, 1.0000],
        [1.0000, 1.0000, 0.6667],
        [0.6667, 1.0000, 0.6667],
        [0.3333, 1.0000, 0.6667],
        [0.0000, 1.0000, 0.6667],
        [1.0000, 0.6667, 0.6667],
        [0.6667, 0.6667, 0.6667],
        [0.3333, 0.6667, 0.6667],
        [0.0000, 0.6667, 0.6667],
        [1.0000, 0.3333, 0.6667],
        [0.6667, 0.3333, 0.6667],
        [0.3333, 0.3333, 0.6667],
        [0.0000, 0.3333, 0.6667],
        [1.0000, 0.0000, 0.6667],
        [0.6667, 0.0000, 0.6667],
        [0.3333, 0.0000, 0.6667],
        [0.0000, 0.0000, 0.6667],
        [1.0000, 1.0000, 0.3333],
        [0.6667, 1.0000, 0.3333],
        [0.3333, 1.0000, 0.3333],
        [0.0000, 1.0000, 0.3333],
        [1.0000, 0.6667, 0.3333],
        [0.6667, 0.6667, 0.3333],
        [0.3333, 0.6667, 0.3333],
        [0.0000, 0.6667, 0.3333],
        [1.0000, 0.3333, 0.3333],
        [0.6667, 0.3333, 0.3333],
        [0.3333, 0.3333, 0.3333],
        [0.0000, 0.3333, 0.3333],
        [1.0000, 0.0000, 0.3333]], device="cuda", dtype=torch.float32)

class RasterizerBatchPredictor():
    def __init__(
        self, 
        batch_size: int, 
        workers: int,
        height, 
        width
    ):
        self.rast_ctx = dr.RasterizeCudaContext()
        self.batch_size = batch_size
        self.workers = workers

        self.height = height
        self.width = width

    def __call__(self, data_joints_2d):
        dataset = NpzDataset(data_joints_2d)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in loader:
                n_batch, n_joints, _ = batch.shape

                batch = batch.to(torch.float32).cuda()

                # Normalize the 2D joint coordinates to the range [-1, 1]
                batch = batch * 2 - 1

                # Add a dummy z coordinate to make the vertices 3D (for rasterization)
                vertices = torch.cat([batch, torch.zeros(n_batch, n_joints, 1).cuda()], dim=-1)  # Shape: [n_batch, n_joints, 3]
                vertices = torch.cat([vertices, torch.ones(n_batch, n_joints, 1).cuda()], dim=-1)  # Shape: [n_batch, n_joints, 4]

                # Rasterization step: get pixel coverage and barycentric coordinates
                rast_out, _ = dr.rasterize(self.rast_ctx, vertices, triangles, resolution=[self.height, self.width])

                batch_vertex_attrs = vertex_attrs.unsqueeze(0).expand(n_batch, -1, -1).contiguous()  # Shape: [n_batch, n_joints, 3]

                # Interpolate vertex features over the rasterized output
                interpolated_features, _ = dr.interpolate(batch_vertex_attrs, rast_out, triangles)

                # Apply masking to retain only valid pixels
                mask = rast_out[..., 3:] > 0  # Valid pixels have alpha > 0
                interpolated_features = interpolated_features * mask

                # Resulting interpolated 2D feature map (n_batch, height, width)
                yield (interpolated_features.squeeze().cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8)