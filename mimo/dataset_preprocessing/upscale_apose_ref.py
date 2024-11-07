import sys
sys.path.append(".")

import os, cv2, tqdm

from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist
from mimo.utils.flux_utils import UpscalerPredictor

from mimo.configs.paths import APOSE_REF_FOLDER, UPSCALED_APOSE_FOLDER


def run_on_image(input_path, prompt, upscaler, output_folder):
    assert_file_exist(input_path)
    input_image = cv2.imread(input_path)

    output_image = upscaler(input_image, prompt)

    basename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, basename)
    cv2.imwrite(output_path, output_image)

def main(
        input_folder=APOSE_REF_FOLDER,
        output_folder=UPSCALED_APOSE_FOLDER,
        cpu_memory_limit_gb=60,
        prompt="Someone in A pose",
        num_inference_steps=24, 
        denoise_strength=0.3, 
        cfg_guidance_scale=3.5, 
        seed=123456
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    set_memory_limit(cpu_memory_limit_gb)

    upscaler = UpscalerPredictor(num_inference_steps, denoise_strength, cfg_guidance_scale, seed)

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.png"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_image(input_path, prompt, upscaler, output_folder), filename, log_path)

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/upscale_apose_ref.py