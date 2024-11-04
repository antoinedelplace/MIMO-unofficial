import sys
sys.path.append(".")

import os, cv2, tqdm

from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist
from mimo.utils.propainter_utils import ProPainterBatchPredictor
from mimo.utils.video_utils import frame_gen_from_video

from mimo.configs.paths import SCENE_FOLDER, FILLED_SCENE_FOLDER

def inpaint_frames(frame_gen, predictor, output_file=None):
    output_frames = None if output_file is not None else []

    for comp_frames, masks_dilated in tqdm.tqdm(predictor.inpaint(frame_gen)):
        for frame, mask in zip(comp_frames, masks_dilated):
            alpha = 0.5
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # frame_bgr = cv2.addWeighted(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), alpha, frame_bgr, 1 - alpha, 0)
            
            if output_file is not None:
                output_file.write(frame_bgr)
            else:
                output_frames.append(frame_bgr)
    
    return output_frames

def run_on_video(input_path, predictor, output_folder):
    assert_file_exist(input_path)
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

    frame_gen = frame_gen_from_video(video)

    inpaint_frames(frame_gen, predictor, output_file)

    video.release()
    output_file.release()

    video_path = assert_file_exist(output_folder, basename)
    video2 = cv2.VideoCapture(video_path)

    width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video2.get(cv2.CAP_PROP_FPS)
    num_frames = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    print("basename", basename)
    print("width", width)
    print("height", height)
    print("frames_per_second", frames_per_second)
    print("num_frames", num_frames)

    video2.release()

def main(
        input_folder=SCENE_FOLDER,
        output_folder=FILLED_SCENE_FOLDER,
        batch_size=64,
        workers=8,
        cpu_memory_limit_gb=60
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    set_memory_limit(cpu_memory_limit_gb)
    predictor = ProPainterBatchPredictor(batch_size, workers)

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_video(input_path, predictor, output_folder), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/video_inpainting.py