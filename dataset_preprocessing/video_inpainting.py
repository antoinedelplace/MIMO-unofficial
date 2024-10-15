import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from utils.general_utils import try_wrapper, set_memory_limit
from utils.propainter_utils import inpaint, load_models

input_folder = "../../data/scene_data/"
output_folder = "../../data/filled_scene_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

set_memory_limit(60)
fix_raft, fix_flow_complete, model = load_models()

def save(comp_frames, input_path, masks_dilated=None):
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("basename", basename)
    # print("width", width)
    # print("height", height)
    # print("frames_per_second", frames_per_second)
    # print("num_frames", num_frames)

    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    if masks_dilated is None:
        for frame in comp_frames:
            output_file.write(frame)
    else:
        alpha = 0.5
        for frame, mask in zip(comp_frames, masks_dilated):
            blended_frame = cv2.addWeighted(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), alpha, frame, 1 - alpha, 0)
            output_file.write(blended_frame)

    output_file.release()
    video.release()

def run_on_video(input_path):
    comp_frames, _ = inpaint(input_path, fix_raft, fix_flow_complete, model)

    save(comp_frames, input_path)


# input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
input_files = sorted(os.listdir(input_folder))
output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

for filename in tqdm.tqdm(input_files):
    basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
    if basename_wo_ext in output_files:
        continue

    input_path = os.path.join(input_folder, filename)
    try_wrapper(lambda: run_on_video(input_path), filename, log_path)


# python dataset_preprocessing/video_inpainting.py