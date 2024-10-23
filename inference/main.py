import sys
sys.path.append(".")

from utils.general_utils import set_memory_limit, parse_args
from inference.inference_pipeline import InferencePipeline

def main(
        input_video_path, 
        output_video_path=None, 
        seed=123456, 
        weight_dtype="fp16", 
        batch_size_depth=12, 
        num_workers=8, 
        cpu_memory_limit_gb=60
    ):
    set_memory_limit(cpu_memory_limit_gb)

    pipe = InferencePipeline(seed, weight_dtype, batch_size_depth, num_workers)

    pipe(input_video_path, output_video_path)

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))


# accelerate launch inference/main.py -i ../../data/mimo_video_cropped/demo_motion_parkour.mp4