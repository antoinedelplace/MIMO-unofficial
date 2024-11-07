import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist
from mimo.utils.clip_embedding_utils import download_image_encoder, CLIPBatchPredictor

from mimo.configs.paths import UPSCALED_APOSE_FOLDER, APOSE_CLIP_EMBEDS_FOLDER


def run_on_image(input_path, clip, output_folder):
    basename = os.path.basename(input_path)

    assert_file_exist(input_path)
    image = cv2.imread(input_path)

    image_embeds = list(clip([image]))[0]
    print("np.shape(image_embeds)", np.shape(image_embeds))
    
    output_path = os.path.join(output_folder, basename).replace(".png", ".npz")
    np.savez_compressed(output_path, 
                        image_embeds=image_embeds)

def main(
        input_folder=UPSCALED_APOSE_FOLDER,
        output_folder=APOSE_CLIP_EMBEDS_FOLDER,
        batch_size=16,
        workers=8,
        cpu_memory_limit_gb=60
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    download_image_encoder()

    set_memory_limit(cpu_memory_limit_gb)

    clip = CLIPBatchPredictor(batch_size, workers)

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.png"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_image(input_path, clip, output_folder), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/clip_embedding.py