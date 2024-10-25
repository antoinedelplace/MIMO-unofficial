import sys
sys.path.append(".")

import os

from mimo.utils.general_utils import set_memory_limit, parse_args
from mimo.inference.inference_pipeline import InferencePipeline

from mimo.configs.paths import DATA_FOLDER

def main(
        input_video_path, 
        output_video_path=None, 
        seed=123456, 
        weight_dtype="fp16", 
        num_workers=8, 
        cpu_memory_limit_gb=60,
        input_net_size=768,
        input_net_fps=24,
        a_pose_raw_path=os.path.join(DATA_FOLDER, "a_pose_raw.png"),
        depth_anything_encoder='vitl',
        score_threshold_detectron2 = 0.9,
        batch_size_depth=12,
        batch_size_detectron2=32,
        batch_size_propainter=128,
        batch_size_reposer=24,
    ):
    set_memory_limit(cpu_memory_limit_gb)

    pipe = InferencePipeline(
        seed=seed, 
        weight_dtype=weight_dtype, 
        num_workers=num_workers,
        input_net_size=input_net_size,
        input_net_fps=input_net_fps,
        a_pose_raw_path=a_pose_raw_path,
        depth_anything_encoder=depth_anything_encoder,
        score_threshold_detectron2=score_threshold_detectron2,
        batch_size_depth=batch_size_depth,
        batch_size_detectron2=batch_size_detectron2,
        batch_size_propainter=batch_size_propainter,
        batch_size_reposer=batch_size_reposer,
    )

    pipe(input_video_path, output_video_path)

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))


# accelerate config
#    - multi-GPU
#    - numa efficiency
#    - fp16

# accelerate launch mimo/inference/main.py -i ../../data/mimo_video_cropped/demo_motion_parkour.mp4