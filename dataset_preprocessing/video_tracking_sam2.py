import sys
sys.path.append(".")
sys.path.append("../sam2")
sys.path.append("../detectron2")

import os, cv2, torch, tqdm
import numpy as np

from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

from sam2.build_sam import build_sam2_video_predictor

from utils.video_utils import frame_gen_from_video
from utils.general_utils import iou

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

def get_global_indexes_foreground_objects(instance_sam_output, depth_video, detectron2_data):
    global_indexes = []

    depth = np.array(list(frame_gen_from_video(depth_video)))[:, :, :, 0]

    mean_depth_human = np.zeros(len(instance_sam_output))
    for i_frame, sam_frame in enumerate(instance_sam_output):
        if len(sam_frame) > 0:  # Human is detected in this frame
            mean_depth_human[i_frame] = np.mean(depth[i_frame][sam_frame.pred_masks[0]])

    for i_global, (i_frame, detectron_mask) in enumerate(zip(detectron2_data["data_frame_index"], detectron2_data["data_pred_masks"])):
        if len(instance_sam_output[i_frame]) > 0:  # Human is detected in this frame
            if iou(instance_sam_output[i_frame].pred_masks[0], detectron_mask) < 0.7:  # It is not the same human
                mean_depth = np.mean(depth[i_frame][detectron_mask])
                if mean_depth < mean_depth_human[i_frame]:
                    global_indexes.append(i_global)

    return global_indexes

def visualize(instance_sam_output, video, input_path):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0) # Set video at the beginning

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
        filename=os.path.join(human_output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    metadata = metadata = MetadataCatalog.get("coco_2017_test")
    video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)

    frame_gen = frame_gen_from_video(video)

    for frame, pred in tqdm.tqdm(zip(frame_gen, instance_sam_output)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_cpu = pred.to(torch.device("cpu"))
        vis_frame = video_visualizer.draw_instance_predictions(frame, pred_cpu)

        # Converts Matplotlib RGB format to OpenCV BGR format
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        output_file.write(vis_frame)

    output_file.release()

    video2 = cv2.VideoCapture(os.path.join(human_output_folder, basename))

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

def run_on_video(input_path):
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

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
                                        obj_id=detectron2_data["data_pred_classes"][i_biggest_human],
                                        box=detectron2_data["data_pred_boxes"][i_biggest_human])

        # propagate the prompts to get masklets throughout the video
        sam_output = list(predictor.propagate_in_video(state))
    
    out_frames_idx = [x[0] for x in sam_output]
    min_frame_idx = min(out_frames_idx)
    max_frame_idx = max(out_frames_idx)
    if max_frame_idx - min_frame_idx < 24:
        raise Exception("Number of frames where human is detected is too low.")
    
    instance_sam_output = [Instances((width, height))]*num_frames
    
    for out_frame_idx, out_obj_ids, out_mask_logits in sam_output:
        instance_sam_output[out_frame_idx] = instance_sam_output[out_frame_idx].cat([Instances((width, height), 
                                      pred_classes=torch.tensor(out_obj_ids, dtype=torch.int8), 
                                      pred_masks=(out_mask_logits > 0)[0].to(torch.bool).cpu())])
    
    # instance_detectron2 = [Instances((width, height))]*num_frames
    
    # for frame_index, pred_classes, pred_boxes, data_scores, data_pred_masks in zip(detectron2_data["data_frame_index"], 
    #                                                        detectron2_data["data_pred_classes"],
    #                                                        detectron2_data["data_pred_boxes"],
    #                                                        detectron2_data["data_scores"],
    #                                                        detectron2_data["data_pred_masks"]):
    #     instance_detectron2[frame_index] = instance_detectron2[frame_index].cat([Instances((width, height), 
    #                                   pred_classes=pred_classes, 
    #                                   pred_boxes=pred_boxes,
    #                                   scores=data_scores,
    #                                   pred_masks=data_pred_masks)])

    global_indexes_foreground_objects = get_global_indexes_foreground_objects(instance_sam_output, depth_video, detectron2_data)
    print("global_indexes_foreground_objects", global_indexes_foreground_objects)
    #visualize(instance_sam_output, video, input_path)

    depth_video.release()
    video.release()

run_on_video(input_path)


# python dataset_preprocessing/video_tracking_sam2.py