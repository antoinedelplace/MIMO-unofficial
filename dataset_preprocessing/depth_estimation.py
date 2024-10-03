import sys
sys.path.append(".")
sys.path.append("../Depth-Anything-V2")

import os, cv2, torch, tqdm
import numpy as np
import matplotlib

from utils.video_utils import frame_gen_from_video
from utils.general_utils import time_it
from utils.depth_anything_v2_utils import BatchPredictor

input_path = "../../data/resized_data/df5afa6a-b7a2-485e-ae12-e3d045e4ebc0-original.mp4"
output_folder = "../../data/detectron2_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

encoder = 'vitl'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

def run_on_video(input_path):
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("basename", basename)
    print("width", width)
    print("height", height)
    print("frames_per_second", frames_per_second)
    print("num_frames", num_frames)

    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    batch_size = 12
    workers = 8

    depth_anything = BatchPredictor(batch_size, workers, width, height, **model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'../../checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    frame_gen = frame_gen_from_video(video)

    for output_batch in depth_anything.infer_video(frame_gen):
        mini = output_batch.min()
        depth = (output_batch - mini) / (output_batch.max() - mini) * 255.0
        depth = depth.astype(np.uint8)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        output_file.write(depth)

    video.release()
    output_file.release()

    video = cv2.VideoCapture(os.path.join(output_folder, basename))

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("basename", basename)
    print("width", width)
    print("height", height)
    print("frames_per_second", frames_per_second)
    print("num_frames", num_frames)

run_on_video(input_path)


# python dataset_preprocessing/depth_estimation.py