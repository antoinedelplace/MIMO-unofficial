import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from mimo.utils.video_utils import frame_gen_from_video, is_video_empty
from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist

from mimo.configs.paths import RAW_FOLDER, RESIZED_FOLDER


def resize_frame(frame, output_width):
    height, width, _ = np.shape(frame)

    if width < height:
        square_image = cv2.copyMakeBorder(
            frame, 
            top=0, 
            bottom=0, 
            left=(height-width)//2, 
            right=(height-width+1)//2, 
            borderType=cv2.BORDER_REFLECT_101
        )
    else:
        square_image = frame[:, width//2-height//2: width//2+(height+1)//2]
    
    return cv2.resize(square_image, (output_width, output_width), interpolation = cv2.INTER_LINEAR)

def sampling_resizing(frame_gen, input_fps, output_fps, output_width, output_file=None):
    output_frames = None if output_file is not None else []

    time_per_frame_original = 1 / input_fps
    time_per_frame_target = 1 / output_fps
    accumulated_time = 0

    for frame in frame_gen:
        accumulated_time += time_per_frame_original

        if accumulated_time >= time_per_frame_target:
            accumulated_time -= time_per_frame_target

            resized_image = resize_frame(frame, output_width)

            if output_file is not None:
                output_file.write(resized_image)
            else:
                output_frames.append(resized_image)
    
    return output_frames

def process_video(input_path, output_size, output_fps, input_fps, output_folder):
    assert_file_exist(input_path)
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

    if input_fps is not None:
        frames_per_second = input_fps

    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(output_fps),
        frameSize=(output_size, output_size),
        isColor=True,
    )

    frame_gen = frame_gen_from_video(video)

    sampling_resizing(frame_gen, frames_per_second, output_fps, output_size, output_file)

    video.release()
    output_file.release()

    # video_path = assert_file_exist(output_folder, basename)
    # video2 = cv2.VideoCapture(video_path)

    # width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frames_per_second = video2.get(cv2.CAP_PROP_FPS)
    # num_frames = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("basename", basename)
    # print("width", width)
    # print("height", height)
    # print("frames_per_second", frames_per_second)
    # print("num_frames", num_frames)

    # video2.release()

def main(
        input_folder=RAW_FOLDER,
        output_folder=RESIZED_FOLDER,
        output_size=768,
        output_fps=24,
        input_fps=None,
        cpu_memory_limit_gb=60
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    set_memory_limit(cpu_memory_limit_gb)

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            if not is_video_empty(os.path.join(output_folder, f"{basename_wo_ext}.mp4")):
                continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: process_video(input_path, output_size, output_fps, input_fps, output_folder), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/video_sampling_resizing.py