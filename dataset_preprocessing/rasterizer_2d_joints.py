import sys
sys.path.append(".")
sys.path.append("../nvdiffrast")

import os, cv2, torch, tqdm
import numpy as np

import nvdiffrast.torch as dr

from utils.general_utils import try_wrapper, set_memory_limit
from utils.rasterizer_utils import triangles, vertex_attrs

input_folder = "../../data/poses_4DH_data/"
output_folder = "../../data/rasterized_2D_joints_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

width = 768
height = 768
fps = 24.0
batch_size = 12
workers = 8
input_size = 768
set_memory_limit(60)

def save(feature_map, basename):
    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, basename).replace(".npz", ".mp4"),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=fps,
        frameSize=(width, height),
        isColor=True,
    )

    for frame in tqdm.tqdm(feature_map):
        output_file.write((frame * 255).clip(min=0, max=255).astype(np.uint8))

    output_file.release()
    
def run_on_video(input_path):
    basename = os.path.basename(input_path)

    rast_ctx = dr.RasterizeCudaContext()

    outputs = dict(np.load(input_path))
    data_joints_2d = outputs["data_joints_2d"]  # Shape: [n_batch, n_joints, 2]

    n_batch, n_joints, _ = data_joints_2d.shape

    data_joints_2d = torch.from_numpy(data_joints_2d).to(torch.float32).cuda()

    # Normalize the 2D joint coordinates to the range [-1, 1]
    data_joints_2d = data_joints_2d * 2 - 1

    # Add a dummy z coordinate to make the vertices 3D (for rasterization)
    vertices = torch.cat([data_joints_2d, torch.zeros(n_batch, n_joints, 1).cuda()], dim=-1)  # Shape: [n_batch, n_joints, 3]
    vertices = torch.cat([vertices, torch.ones(n_batch, n_joints, 1).cuda()], dim=-1)  # Shape: [n_batch, n_joints, 4]

    # Rasterization step: get pixel coverage and barycentric coordinates
    rast_out, _ = dr.rasterize(rast_ctx, vertices, triangles, resolution=[height, width])

    batch_vertex_attrs = vertex_attrs.unsqueeze(0).expand(n_batch, -1, -1).contiguous()  # Shape: [n_batch, n_joints, 3]

    # Interpolate vertex features over the rasterized output
    interpolated_features, _ = dr.interpolate(batch_vertex_attrs, rast_out, triangles)

    # Apply masking to retain only valid pixels
    mask = rast_out[..., 3:] > 0  # Valid pixels have alpha > 0
    interpolated_features = interpolated_features * mask

    # Resulting interpolated 2D feature map (n_batch, height, width)
    feature_map = interpolated_features.squeeze().cpu().numpy()

    save(feature_map, basename)


# input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.npz"]
input_files = sorted(os.listdir(input_folder))
output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

for filename in tqdm.tqdm(input_files):
    basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
    if basename_wo_ext in output_files:
        continue

    input_path = os.path.join(input_folder, filename)
    try_wrapper(lambda: run_on_video(input_path), filename, log_path)


# python dataset_preprocessing/rasterizer_2d_joints.py