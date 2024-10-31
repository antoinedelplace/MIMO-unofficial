import sys
sys.path.append(".")

import os, cv2, torch, tqdm
import numpy as np

from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.depth_anything_v2_utils import DepthBatchPredictor
from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist

from mimo.configs.paths import RESIZED_FOLDER, DEPTH_FOLDER, CHECKPOINTS_FOLDER

DEPTH_ANYTHING_MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

def get_depth(frame_gen, depth_anything, output_file=None):
    output_frames = None if output_file is not None else []

    for output_batch in depth_anything.infer_video(frame_gen):
        mini = output_batch.min()
        depth = (output_batch - mini) / (output_batch.max() - mini) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        if output_file is not None:
            for frame in depth:
                output_file.write(frame)
        else:
            output_frames.append(depth)
    
    return output_frames

def run_on_video(input_path, depth_anything, output_folder):
    assert_file_exist(input_path)
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )
    
    frame_gen = frame_gen_from_video(video)

    get_depth(frame_gen, depth_anything, output_file)

    video.release()
    output_file.release()


def main(
        input_folder=RESIZED_FOLDER,
        output_folder=DEPTH_FOLDER,
        batch_size=12,
        workers=8,
        input_size=768,
        encoder='vitl',
        cpu_memory_limit_gb=60
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    set_memory_limit(cpu_memory_limit_gb)

    depth_anything = DepthBatchPredictor(batch_size, workers, torch.bfloat16, input_size, input_size, **DEPTH_ANYTHING_MODEL_CONFIGS[encoder])
    checkpoint_path = assert_file_exist(CHECKPOINTS_FOLDER, f'depth_anything_v2_{encoder}.pth')
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    depth_anything = depth_anything.to(torch.bfloat16).to(device).eval()

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_video(input_path, depth_anything, output_folder), filename, log_path)

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/depth_estimation.py