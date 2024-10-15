import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from utils.video_utils import frame_gen_from_video
from utils.general_utils import try_wrapper, set_memory_limit
from utils.vae_encoding_utils import download_vae, VaeBatchPredictor

scene_input_folder = "../../data/filled_scene_data/"
occlusion_input_folder = "../../data/occlusion_data/"
output_folder = "../../data/encoded_occlusion_scene_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

checkpoints_folder = "../../checkpoints"
download_vae(checkpoints_folder)

batch_size = 16
workers = 8
set_memory_limit(60)

vae = VaeBatchPredictor(batch_size, workers, checkpoints_folder)

def visualize(latent, video, input_path):
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

    for frames in vae.decode(latent):
        for frame in frames:
            output_file.write(frame)

    output_file.release()


def run_on_video(input_path):
    basename = os.path.basename(input_path)

    video = cv2.VideoCapture(input_path)
    frame_gen = frame_gen_from_video(video)

    latent_scene = np.concatenate(list(vae.encode(frame_gen)))
    print("np.shape(latent_scene)", np.shape(latent_scene))
    # visualize(latent_scene, video, input_path)

    video.release()

    video = cv2.VideoCapture(os.path.join(occlusion_input_folder, basename))
    frame_gen = frame_gen_from_video(video)

    latent_occlusion = np.concatenate(list(vae.encode(frame_gen)))
    print("np.shape(latent_occlusion)", np.shape(latent_occlusion))
    # visualize(latent_occlusion, video, input_path)
    
    video.release()
    
    output_path = os.path.join(output_folder, basename).replace(".mp4", ".npz")
    np.savez_compressed(output_path, 
                        latent_scene=latent_scene, 
                        latent_occlusion=latent_occlusion)

# input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
input_files = sorted(os.listdir(scene_input_folder))
output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

for filename in tqdm.tqdm(input_files):
    basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
    if basename_wo_ext in output_files:
        continue

    input_path = os.path.join(scene_input_folder, filename)
    try_wrapper(lambda: run_on_video(input_path), filename, log_path)


# python dataset_preprocessing/vae_encoding.py