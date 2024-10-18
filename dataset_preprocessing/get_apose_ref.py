import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from utils.video_utils import frame_gen_from_video
from utils.general_utils import try_wrapper, set_memory_limit, parse_args
from utils.apose_ref_utils import download_base_model, download_anyone, download_dwpose, get_frame_closest_pose, ReposerBatchPredictor, get_kps_image
from utils.clip_embedding_utils import download_image_encoder, CLIPBatchPredictor
from utils.vae_encoding_utils import download_vae, VaeBatchPredictor

from configs.paths import APOSE_REF_FOLDER, HUMAN_FOLDER, DATA_FOLDER

def run_on_video(input_path, reposer, a_pose_kps, ref_points_2d, output_folder):
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)

    frame_gen = frame_gen_from_video(video)

    input_image = get_frame_closest_pose(video, frame_gen, ref_points_2d)
    # print("np.shape(input_image)", np.shape(input_image))
    # cv2.imwrite("../../data/ref_pose.png", input_image)

    output_image = np.concatenate(list(reposer(input_image, [a_pose_kps]*24))) # Need 24 frames to get relevant outputs
    # print("np.shape(output_image)", np.shape(output_image))

    output_path = os.path.join(output_folder, basename).replace(".mp4", ".png")
    cv2.imwrite(output_path, output_image[0])

    video.release()

def main(
        input_folder=HUMAN_FOLDER,
        output_folder=APOSE_REF_FOLDER,
        batch_size=24,
        workers=8,
        cpu_memory_limit_gb=60,
        a_pose_raw_path = os.path.join(DATA_FOLDER, "a_pose_raw.png")
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    download_image_encoder()
    download_vae()
    download_base_model()
    download_anyone()
    download_dwpose()

    set_memory_limit(cpu_memory_limit_gb)

    vae = VaeBatchPredictor(batch_size, workers)
    clip = CLIPBatchPredictor(batch_size, workers)
    reposer = ReposerBatchPredictor(batch_size, workers, clip, vae)

    a_pose_kps, ref_points_2d = get_kps_image(a_pose_raw_path)
    # print("np.shape(a_pose_kps)", np.shape(a_pose_kps))
    # print("np.shape(ref_points_2d['bodies']['candidate'])", np.shape(ref_points_2d['bodies']['candidate']))

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_video(input_path, reposer, a_pose_kps, ref_points_2d, output_folder), filename, log_path)

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python dataset_preprocessing/get_apose_ref.py