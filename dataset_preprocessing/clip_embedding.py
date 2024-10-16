import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from utils.general_utils import try_wrapper, set_memory_limit
from utils.clip_embedding_utils import download_image_encoder, CLIPBatchPredictor

input_folder = "../../data/apose_ref_data/"
output_folder = "../../data/apose_clip_embeds_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

checkpoints_folder = "../../checkpoints"
download_image_encoder(checkpoints_folder)

batch_size = 16
workers = 8
set_memory_limit(60)

clip = CLIPBatchPredictor(batch_size, workers, checkpoints_folder)

def run_on_video(input_path):
    basename = os.path.basename(input_path)

    image = cv2.imread(input_path)

    image_embeds = clip(image)[0]
    print("np.shape(image_embeds)", np.shape(image_embeds))
    
    output_path = os.path.join(output_folder, basename).replace(".jpg", ".npz")
    np.savez_compressed(output_path, 
                        image_embeds=image_embeds)

# input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.jpg"]
input_files = sorted(os.listdir(input_folder))
output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

for filename in tqdm.tqdm(input_files):
    basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
    if basename_wo_ext in output_files:
        continue

    input_path = os.path.join(input_folder, filename)
    try_wrapper(lambda: run_on_video(input_path), filename, log_path)


# python dataset_preprocessing/clip_embedding.py