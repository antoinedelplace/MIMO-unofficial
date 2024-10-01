import sys
sys.path.append(".")

import os, tqdm
from utils.video_utils import hash_file
from utils.general_utils import try_wrapper


input_folder = "../../data/resized_data/"
log_path = os.path.join(input_folder, "error_log_remove_duplicate.txt")

hashset = set()

def process_video(input_path):
    hash = hash_file(input_path)
    if hash in hashset:
        print(f"File deleted: {input_path}")
        os.remove(input_path)
    else:
        hashset.add(hash)

for filename in tqdm.tqdm(os.listdir(input_folder)):
    input_path = os.path.join(input_folder, filename)
    
    try_wrapper(lambda: process_video(input_path), filename, log_path)

# python dataset_preprocessing/remove_duplicate_videos.py