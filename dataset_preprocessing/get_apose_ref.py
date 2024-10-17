import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from utils.video_utils import frame_gen_from_video
from utils.general_utils import try_wrapper, set_memory_limit
from utils.apose_ref_utils import download_base_model, download_anyone, download_dwpose, get_frame_with_median_mask, ReposerBatchPredictor, get_kps_image
from utils.clip_embedding_utils import download_image_encoder, CLIPBatchPredictor
from utils.vae_encoding_utils import download_vae, VaeBatchPredictor

input_folder = "../../data/human_data/"
output_folder = "../../data/apose_ref_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

# a_pose_raw_path = "../../data/a_pose_raw.png"
a_pose_kps_path = "../../data/a_pose_kps.png"

checkpoints_folder = "../../checkpoints"
download_image_encoder(checkpoints_folder)
download_vae(checkpoints_folder)
download_base_model(checkpoints_folder)
download_anyone(checkpoints_folder)
# download_dwpose(checkpoints_folder)

batch_size = 16
workers = 8
set_memory_limit(60)

vae = VaeBatchPredictor(batch_size, workers, checkpoints_folder)
clip = CLIPBatchPredictor(batch_size, workers, checkpoints_folder)
reposer = ReposerBatchPredictor(batch_size, workers, checkpoints_folder, clip, vae)

def run_on_video(input_path):
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

    frame_gen = frame_gen_from_video(video)

    input_image = get_frame_with_median_mask(video, frame_gen)
    print("np.shape(input_image)", np.shape(input_image))

    # a_pose_kps = get_kps_image(a_pose_raw_path, checkpoints_folder)
    # cv2.imwrite(a_pose_kps_path, a_pose_kps)

    a_pose_kps = cv2.imread(a_pose_kps_path)
    print("np.shape(a_pose_image)", np.shape(a_pose_kps))

    output_image = list(reposer(input_image, [a_pose_kps]))[0][0]
    print("np.shape(output_image)", np.shape(output_image))

    output_path = os.path.join(output_folder, basename).replace(".mp4", ".png")
    cv2.imwrite(output_path, output_image)

    video.release()

run_on_video("../../data/human_data/03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4")
print(1/0)

# input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
input_files = sorted(os.listdir(input_folder))
output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

for filename in tqdm.tqdm(input_files):
    basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
    if basename_wo_ext in output_files:
        continue

    input_path = os.path.join(input_folder, filename)
    try_wrapper(lambda: run_on_video(input_path), filename, log_path)


# python dataset_preprocessing/get_apose_ref.py