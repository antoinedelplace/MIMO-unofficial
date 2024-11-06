import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist
from mimo.utils.apose_ref_utils import download_dwpose, get_kps_image, CustomDWposeDetector, get_frame_closest_pose
from mimo.utils.flux_utils import ReposerPredictor

from mimo.configs.paths import APOSE_REF_FOLDER, HUMAN_FOLDER, DATA_FOLDER

def get_apose_ref_img(frame_gen, reposer, dw_pose_detector, ref_points_2d, a_pose_kps):
    # input_image = get_frame_closest_pose(frame_gen, ref_points_2d, dw_pose_detector)
    # print("np.shape(input_image)", np.shape(input_image))
    # cv2.imwrite("../../data/ref_pose.png", input_image)

    input_image = None

    output_image = np.array(reposer(input_image, a_pose_kps))
    print("np.shape(output_image)", np.shape(output_image))

    return output_image

def run_on_video(input_path, reposer, dw_pose_detector, ref_points_2d, a_pose_kps, output_folder):
    assert_file_exist(input_path)
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)

    frame_gen = frame_gen_from_video(video)

    apose_ref_img = get_apose_ref_img(frame_gen, reposer, dw_pose_detector, ref_points_2d, a_pose_kps)

    output_path = os.path.join(output_folder, basename).replace(".mp4", ".png")
    cv2.imwrite(output_path, apose_ref_img)

    video.release()

def main(
        input_folder=HUMAN_FOLDER,
        output_folder=APOSE_REF_FOLDER,
        cpu_memory_limit_gb=60,
        a_pose_raw_path = os.path.join(DATA_FOLDER, "a_pose_raw.png")
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    download_dwpose()

    set_memory_limit(cpu_memory_limit_gb)

    dw_pose_detector = CustomDWposeDetector()
    reposer = ReposerPredictor()

    assert_file_exist(a_pose_raw_path)
    a_pose_kps, ref_points_2d = get_kps_image(a_pose_raw_path, dw_pose_detector)
    # print("np.shape(a_pose_kps)", np.shape(a_pose_kps))
    # print("np.shape(ref_points_2d['bodies']['candidate'])", np.shape(ref_points_2d['bodies']['candidate']))

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    run_on_video("../../data/human_data/03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4", reposer, dw_pose_detector, ref_points_2d, a_pose_kps, output_folder)
    print(1/0)

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_video(input_path, reposer, dw_pose_detector, ref_points_2d, a_pose_kps, output_folder), filename, log_path)

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/get_apose_ref_flux.py