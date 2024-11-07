import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist
from mimo.utils.rasterizer_utils import RasterizerBatchPredictor
from mimo.utils.video_utils import is_video_empty

from mimo.configs.paths import POSES_4DH_FOLDER, RASTERIZED_2D_JOINTS_FOLDER


def save(feature_map, basename, fps, output_folder):
    _, height, width, _ = feature_map.shape

    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, basename).replace(".npz", ".mp4"),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=fps,
        frameSize=(width, height),
        isColor=True,
    )

    for frame in tqdm.tqdm(feature_map):
        output_file.write(frame)

    output_file.release()
    
    
def run_on_video(input_path, fps, rasterizer, output_folder):
    assert_file_exist(input_path)
    basename = os.path.basename(input_path)

    outputs = dict(np.load(input_path))
    data_joints_2d = outputs["data_joints_2d"]  # Shape: [n_batch, n_joints, 2]

    feature_map = np.concatenate(list(rasterizer(data_joints_2d)))

    save(feature_map, basename, fps, output_folder)

def main(
        input_folder=POSES_4DH_FOLDER,
        output_folder=RASTERIZED_2D_JOINTS_FOLDER,
        width = 768,
        height = 768,
        fps = 24.0,
        batch_size=256,
        workers=8,
        cpu_memory_limit_gb=60
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    set_memory_limit(cpu_memory_limit_gb)

    rasterizer = RasterizerBatchPredictor(batch_size, workers, height, width)

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.npz"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            if not is_video_empty(os.path.join(output_folder, f"{basename_wo_ext}.mp4")):
                continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_video(input_path, fps, rasterizer, output_folder), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/rasterizer_2d_joints.py