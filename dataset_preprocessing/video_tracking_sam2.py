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
from utils.general_utils import iou, set_memory_limit, try_wrapper

input_folder = "../../data/resized_data/"
depth_input_folder = "../../data/depth_data/"
detectron2_input_folder = "../../data/detectron2_data/"
human_output_folder = "../../data/human_data/"
scene_output_folder = "../../data/scene_data/"
occlusion_output_folder = "../../data/occlusion_data/"
os.makedirs(human_output_folder, exist_ok=True)
os.makedirs(scene_output_folder, exist_ok=True)
os.makedirs(occlusion_output_folder, exist_ok=True)
log_path = os.path.join(human_output_folder, "error_log.txt")

score_threshold = 0.9
set_memory_limit(60)
checkpoint = "../../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

def get_index_first_frame_with_character(detectron2_data):
    zero = np.where((detectron2_data["data_pred_classes"] == 0) & (detectron2_data["data_scores"] > score_threshold))

    # Check if zero[0] is empty, meaning no frames matched the condition
    if zero[0].size == 0:
        raise ValueError("No frames found with the character and score above the threshold.")
    
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

    for i_global, (i_frame, detectron_mask, score) in enumerate(zip(detectron2_data["data_frame_index"], detectron2_data["data_pred_masks"], detectron2_data["data_scores"])):
        if len(instance_sam_output[i_frame]) > 0:  # Human is detected in this frame
            if score > score_threshold:
                if iou(instance_sam_output[i_frame].pred_masks[0], detectron_mask) < 0.7:  # It is not the same human
                    mean_depth = np.mean(depth[i_frame][detectron_mask])
                    if mean_depth > mean_depth_human[i_frame]:
                        global_indexes.append(i_global)

    return global_indexes

def get_global_indexes_filtered_already_in_mask(global_indexes, sam_output, detectron2_data):
    indexes_to_remove = []
    i_global_indexes = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in sam_output:
        pred_mask = (out_mask_logits > 0).squeeze().to(torch.bool).cpu()
        while i_global_indexes < len(global_indexes) and detectron2_data["data_frame_index"][global_indexes[i_global_indexes]] < out_frame_idx:
            i_global_indexes += 1
        while i_global_indexes < len(global_indexes) and detectron2_data["data_frame_index"][global_indexes[i_global_indexes]] <= out_frame_idx:
            if iou(pred_mask, detectron2_data["data_pred_masks"][global_indexes[i_global_indexes]]) > 0.3:
                indexes_to_remove.append(i_global_indexes)
            i_global_indexes += 1

    new_global_indexes = [global_indexes[i] for i in range(len(global_indexes)) if i not in indexes_to_remove]
    return new_global_indexes

def visualize(sam_output, video, input_path):
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

    instance_sam_output = [Instances((width, height))]*num_frames
        
    for out_frame_idx, out_obj_ids, out_mask_logits in sam_output:
        instance_sam_output[out_frame_idx] = instance_sam_output[out_frame_idx].cat([Instances((width, height), 
                                    pred_classes=torch.tensor(out_obj_ids, dtype=torch.int8), 
                                    pred_masks=(out_mask_logits > 0)[0].to(torch.bool).cpu())])

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

def save(min_frame_idx, max_frame_idx, instance_sam_output, foreground_mask, video, input_path):
    num_frames_output = max_frame_idx-min_frame_idx+1
    video.set(cv2.CAP_PROP_POS_FRAMES, min_frame_idx)

    basename = os.path.basename(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)

    output_human_file = cv2.VideoWriter(
        filename=os.path.join(human_output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    output_scene_file = cv2.VideoWriter(
        filename=os.path.join(scene_output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    output_occlusion_file = cv2.VideoWriter(
        filename=os.path.join(occlusion_output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    frames = list(frame_gen_from_video(video))

    for i_frame in range(num_frames_output):
        human_mask = instance_sam_output[i_frame].pred_masks[0].numpy()
        occlusion_mask = foreground_mask[i_frame]

        human_mask = np.expand_dims(human_mask, axis=-1)
        occlusion_mask = np.expand_dims(occlusion_mask, axis=-1)

        occlusion_wo_human_mask = occlusion_mask & ~human_mask
        scene_mask = ~occlusion_mask & ~human_mask

        output_human_file.write(frames[i_frame]*human_mask)
        output_occlusion_file.write(frames[i_frame]*occlusion_wo_human_mask)
        output_scene_file.write(frames[i_frame]*scene_mask)

    output_human_file.release()
    output_scene_file.release()
    output_occlusion_file.release()

def run_on_video(input_path):
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    detectron2_data = dict(np.load(os.path.join(detectron2_input_folder, basename).replace(".mp4", ".npz")))
    depth_video = cv2.VideoCapture(os.path.join(depth_input_folder, basename))

    i_first_frame = get_index_first_frame_with_character(detectron2_data)
    print("i_first_frame", i_first_frame)
    i_biggest_human = get_global_index_biggest_human_in_frame(detectron2_data, i_first_frame)
    print("i_biggest_human", i_biggest_human)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(input_path)

        predictor.add_new_points_or_box(inference_state=state, 
                                        frame_idx=detectron2_data["data_frame_index"][i_biggest_human],
                                        obj_id=detectron2_data["data_pred_classes"][i_biggest_human],
                                        box=detectron2_data["data_pred_boxes"][i_biggest_human])

        sam_output = list(predictor.propagate_in_video(state, start_frame_idx=0))
    
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

        #visualize(sam_output, video, input_path)

        global_indexes_foreground_objects = get_global_indexes_foreground_objects(instance_sam_output, depth_video, detectron2_data)
        print("len(global_indexes_foreground_objects)", len(global_indexes_foreground_objects))

        foreground_mask = torch.zeros((num_frames, width, height), dtype=bool)
        while len(global_indexes_foreground_objects) > 0:
            predictor.reset_state(state)

            i_global_indexes = global_indexes_foreground_objects.pop(0)

            predictor.add_new_points_or_box(inference_state=state, 
                                            frame_idx=detectron2_data["data_frame_index"][i_global_indexes],
                                            obj_id=detectron2_data["data_pred_classes"][i_global_indexes],
                                            box=detectron2_data["data_pred_boxes"][i_global_indexes])

            sam_output = list(predictor.propagate_in_video(state, start_frame_idx=0))
            # visualize(sam_output, video, f"{i_global_indexes}_{basename}")

            for out_frame_idx, out_obj_ids, out_mask_logits in sam_output:
                foreground_mask[out_frame_idx] += (out_mask_logits > 0).squeeze().to(torch.bool).cpu()

            global_indexes_foreground_objects = get_global_indexes_filtered_already_in_mask(global_indexes_foreground_objects, sam_output, detectron2_data)
            print("len(global_indexes_foreground_objects)", len(global_indexes_foreground_objects))

    save(min_frame_idx, max_frame_idx, instance_sam_output, foreground_mask.numpy(), video, input_path)

    depth_video.release()
    video.release()

# input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
input_files = sorted(os.listdir(input_folder))
output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(human_output_folder)])

for filename in tqdm.tqdm(input_files):
    basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
    if basename_wo_ext in output_files:
        continue

    input_path = os.path.join(input_folder, filename)
    try_wrapper(lambda: run_on_video(input_path), filename, log_path)


# python dataset_preprocessing/video_tracking_sam2.py