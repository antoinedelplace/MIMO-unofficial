import sys
sys.path.append(".")

import os, tqdm
from utils.video_utils import hash_file
from utils.general_utils import try_wrapper, parse_args

from configs.paths import RESIZED_FOLDER


def process_video(input_path, hashset):
    hash = hash_file(input_path)
    if hash in hashset:
        print(f"File deleted: {input_path}")
        os.remove(input_path)
    else:
        hashset.add(hash)
    
    return hashset

def main(
        input_folder=RESIZED_FOLDER,
        ):
    log_path = os.path.join(input_folder, "error_log.txt")

    hashset = set()

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))

    for filename in tqdm.tqdm(input_files):
        input_path = os.path.join(input_folder, filename)
        
        hashset = try_wrapper(lambda: process_video(input_path, hashset), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python dataset_preprocessing/remove_duplicate_videos.py