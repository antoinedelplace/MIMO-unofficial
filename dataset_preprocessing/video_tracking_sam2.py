import sys
sys.path.append(".")
sys.path.append("../sam2")

import os, cv2, torch, tqdm
import numpy as np

from sam2.build_sam import build_sam2_video_predictor

from utils.video_utils import frame_gen_from_video

input_path = "../../data/resized_data/03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"
depth_input_folder = "../../data/depth_data/"
detectron2_input_folder = "../../data/detectron2_data/"
human_output_folder = "../../data/human_data/"
scene_output_folder = "../../data/scene_data/"
oclusion_output_folder = "../../data/occlusion_data/"
os.makedirs(human_output_folder, exist_ok=True)
os.makedirs(scene_output_folder, exist_ok=True)
os.makedirs(oclusion_output_folder, exist_ok=True)
log_path = os.path.join(human_output_folder, "error_log.txt")

batch_size = 24
workers = 8
score_threshold = 0.7
checkpoint = "../../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

def get_index_first_frame_with_character(detectron2_data):
    zero = np.where((detectron2_data["data_pred_classes"] == 0) & (detectron2_data["data_scores"] > score_threshold))
    return detectron2_data["data_frame_index"][zero[0][0]]

def get_global_index_biggest_human_in_frame(detectron2_data, i_frame):
    indexes_humans_at_input_frame = np.where((detectron2_data["data_pred_classes"] == 0) 
                                             & (detectron2_data["data_frame_index"] == i_frame) 
                                             & (detectron2_data["data_scores"] > score_threshold))

    i_sub = np.argmax(detectron2_data["data_pred_masks"][indexes_humans_at_input_frame].sum(axis=(1, 2)))
    return indexes_humans_at_input_frame[0][i_sub]

def run_on_video(input_path):
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)

    detectron2_data = np.load(os.path.join(detectron2_input_folder, basename).replace(".mp4", ".npz"))
    depth_video = cv2.VideoCapture(os.path.join(depth_input_folder, basename))

    i_first_frame = get_index_first_frame_with_character(detectron2_data)
    print("i_first_frame", i_first_frame)
    i_biggest_human = get_global_index_biggest_human_in_frame(detectron2_data, i_first_frame)
    print("i_biggest_human", i_biggest_human)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(input_path)

        # add new prompts and instantly get the output on the same frame
        predictor.add_new_points_or_box(inference_state=state, 
                                        frame_idx=detectron2_data["data_frame_index"][i_biggest_human],
                                        obj_id=0,
                                        box=detectron2_data["data_pred_boxes"][i_biggest_human])

        # propagate the prompts to get masklets throughout the video
        outputs = predictor.propagate_in_video(state)

        print(outputs)

    
    # frame_gen = frame_gen_from_video(video)

    # outputs = list(predictor(frame_gen))

    # print(outputs)

    depth_video.release()
    video.release()

run_on_video(input_path)


# python dataset_preprocessing/video_tracking_sam2.py