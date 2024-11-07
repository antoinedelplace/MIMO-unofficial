import sys
sys.path.append(".")

from mimo.configs.paths import SAM2_REPO, DETECTRON2_REPO, RESIZED_FOLDER, DEPTH_FOLDER, DETECTRON2_FOLDER, HUMAN_FOLDER, SCENE_FOLDER, OCCLUSION_FOLDER, CHECKPOINTS_FOLDER
sys.path.append(SAM2_REPO)
sys.path.append(DETECTRON2_REPO)

import os, cv2, torch, tqdm
import numpy as np

from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

from sam2.build_sam import build_sam2_video_predictor

from mimo.utils.video_utils import frame_gen_from_video, is_video_empty
from mimo.utils.general_utils import iou, set_memory_limit, try_wrapper, parse_args, assert_file_exist


def get_index_first_frame_with_character(detectron2_data, score_threshold):
    zero = np.where((detectron2_data["data_pred_classes"] == 0) & (detectron2_data["data_scores"] > score_threshold))

    # Check if zero[0] is empty, meaning no frames matched the condition
    if zero[0].size == 0:
        raise ValueError("No frames found with the character and score above the threshold.")
    
    return detectron2_data["data_frame_index"][zero[0][0]]

def get_global_index_biggest_human_in_frame(detectron2_data, i_frame, score_threshold):
    indexes_humans_at_input_frame = np.where((detectron2_data["data_pred_classes"] == 0) 
                                             & (detectron2_data["data_frame_index"] == i_frame) 
                                             & (detectron2_data["data_scores"] > score_threshold))

    i_sub = np.argmax(detectron2_data["data_pred_masks"][indexes_humans_at_input_frame].sum(axis=(1, 2)))
    return indexes_humans_at_input_frame[0][i_sub]

def get_global_index_most_central_human_in_frame(detectron2_data, i_frame, score_threshold, width, height):
    indexes_humans_at_input_frame = np.where((detectron2_data["data_pred_classes"] == 0) 
                                             & (detectron2_data["data_frame_index"] == i_frame) 
                                             & (detectron2_data["data_scores"] > score_threshold))

    
    centers = Boxes(detectron2_data["data_pred_boxes"][indexes_humans_at_input_frame]).get_centers()
    i_sub = np.argmin((centers[:, 0] - width/2)**2+(centers[:, 1] - height/2)**2)
    return indexes_humans_at_input_frame[0][i_sub]

def get_global_indexes_foreground_objects(instance_sam_output, depth, detectron2_data, score_threshold):
    global_indexes = []

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

def visualize(sam_output, video, input_path, human_output_folder):
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

    video_path = assert_file_exist(human_output_folder, basename)
    video2 = cv2.VideoCapture(video_path)

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

def process_layers(frames, 
                   instance_sam_output, 
                   foreground_mask, 
                   min_frame_idx, 
                   max_frame_idx, 
                   output_human_file=None,
                   output_occlusion_file=None,
                   output_scene_file=None):
    human_frames = None if output_human_file is not None else []
    occlusion_frames = None if output_occlusion_file is not None else []
    scene_frames = None if output_scene_file is not None else []

    num_frames_output = max_frame_idx-min_frame_idx+1

    for i_frame in range(num_frames_output):
        human_mask = instance_sam_output[i_frame].pred_masks[0].numpy()
        occlusion_mask = foreground_mask[i_frame]

        human_mask = np.expand_dims(human_mask, axis=-1)
        occlusion_mask = np.expand_dims(occlusion_mask, axis=-1)

        occlusion_wo_human_mask = occlusion_mask & ~human_mask
        scene_mask = ~occlusion_mask & ~human_mask

        human_frame = frames[i_frame]*human_mask
        if output_human_file is not None:
            output_human_file.write(human_frame)
        else:
            human_frames.append(human_frame)
        
        occlusion_frame = frames[i_frame]*occlusion_wo_human_mask
        if output_occlusion_file is not None:
            output_occlusion_file.write(occlusion_frame)
        else:
            occlusion_frames.append(occlusion_frame)
        
        scene_frame = frames[i_frame]*scene_mask
        if output_scene_file is not None:
            output_scene_file.write(scene_frame)
        else:
            scene_frames.append(scene_frame)

    return human_frames, occlusion_frames, scene_frames

def save(
        min_frame_idx, 
        max_frame_idx, 
        instance_sam_output, 
        foreground_mask, 
        video, 
        input_path, 
        human_output_folder, 
        scene_output_folder, 
        occlusion_output_folder):
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

    process_layers(frames, 
                   instance_sam_output, 
                   foreground_mask, 
                   min_frame_idx, 
                   max_frame_idx, 
                   output_human_file,
                   output_occlusion_file,
                   output_scene_file)

    output_human_file.release()
    output_scene_file.release()
    output_occlusion_file.release()

def get_instance_sam_output(input_path, detectron2_data, depth, predictor, score_threshold):
    depth = depth[:, :, :, 0]
    num_frames, width, height = np.shape(depth)

    i_first_frame = get_index_first_frame_with_character(detectron2_data, score_threshold)
    print("i_first_frame", i_first_frame)
    i_central_human = get_global_index_most_central_human_in_frame(detectron2_data, i_first_frame, score_threshold, width, height)
    print("i_central_human", i_central_human)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(input_path)

        predictor.add_new_points_or_box(inference_state=state, 
                                        frame_idx=detectron2_data["data_frame_index"][i_central_human],
                                        obj_id=detectron2_data["data_pred_classes"][i_central_human],
                                        box=detectron2_data["data_pred_boxes"][i_central_human])

        sam_output = [(out_frame_idx, out_obj_ids, out_mask_logits.cpu()) for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, start_frame_idx=0)]
    
        out_frames_idx = [x[0] for x in sam_output]
        min_frame_idx = min(out_frames_idx)
        max_frame_idx = max(out_frames_idx)
        if max_frame_idx - min_frame_idx < 24:
            raise Exception("Number of frames where human is detected is too low.")
        
        instance_sam_output = [Instances((width, height))]*num_frames
        
        for out_frame_idx, out_obj_ids, out_mask_logits in sam_output:
            instance_sam_output[out_frame_idx] = instance_sam_output[out_frame_idx].cat([Instances((width, height), 
                                        pred_classes=torch.tensor(out_obj_ids, dtype=torch.int8), 
                                        pred_masks=(out_mask_logits > 0)[0].to(torch.bool))])

        del sam_output
        #visualize(sam_output, video, input_path, human_output_folder)

        global_indexes_foreground_objects = get_global_indexes_foreground_objects(instance_sam_output, depth, detectron2_data, score_threshold)
        print("len(global_indexes_foreground_objects)", len(global_indexes_foreground_objects))

        foreground_mask = torch.zeros((num_frames, width, height), dtype=bool)
        while len(global_indexes_foreground_objects) > 0:
            predictor.reset_state(state)

            i_global_indexes = global_indexes_foreground_objects.pop(0)

            predictor.add_new_points_or_box(inference_state=state, 
                                            frame_idx=detectron2_data["data_frame_index"][i_global_indexes],
                                            obj_id=detectron2_data["data_pred_classes"][i_global_indexes],
                                            box=detectron2_data["data_pred_boxes"][i_global_indexes])

            sam_output = [(out_frame_idx, out_obj_ids, out_mask_logits.cpu()) for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, start_frame_idx=0)]
            # visualize(sam_output, video, f"{i_global_indexes}_{basename}", human_output_folder)

            for out_frame_idx, out_obj_ids, out_mask_logits in sam_output:
                foreground_mask[out_frame_idx] += (out_mask_logits > 0).squeeze().to(torch.bool)

            global_indexes_foreground_objects = get_global_indexes_filtered_already_in_mask(
                global_indexes_foreground_objects, 
                sam_output, 
                detectron2_data
            )
            print("len(global_indexes_foreground_objects)", len(global_indexes_foreground_objects))

            del sam_output
    
    predictor.reset_state(state)
    
    return min_frame_idx, max_frame_idx, instance_sam_output, foreground_mask

def run_on_video(input_path, 
                 depth_input_folder, 
                 detectron2_input_folder, 
                 predictor, 
                 score_threshold,
                 human_output_folder, 
                 scene_output_folder, 
                 occlusion_output_folder):
    assert_file_exist(input_path)
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)

    detectron2_path = assert_file_exist(detectron2_input_folder, basename.replace(".mp4", ".npz"))
    detectron2_data = dict(np.load(detectron2_path))
    depth_path = assert_file_exist(depth_input_folder, basename)
    depth_video = cv2.VideoCapture(depth_path)

    detectron2_data["data_frame_index"] = detectron2_data["data_frame_index"].astype(np.int32)
    depth = np.array(list(frame_gen_from_video(depth_video)))
    
    min_frame_idx, max_frame_idx, instance_sam_output, foreground_mask = get_instance_sam_output(input_path, detectron2_data, depth, predictor, score_threshold)

    save(min_frame_idx, 
         max_frame_idx, 
         instance_sam_output, 
         foreground_mask.numpy(), 
         video, 
         input_path, 
         human_output_folder, 
         scene_output_folder, 
         occlusion_output_folder)

    depth_video.release()
    video.release()

def main(
        input_folder=RESIZED_FOLDER,
        depth_input_folder = DEPTH_FOLDER,
        detectron2_input_folder = DETECTRON2_FOLDER,
        human_output_folder = HUMAN_FOLDER,
        scene_output_folder = SCENE_FOLDER,
        occlusion_output_folder = OCCLUSION_FOLDER,
        score_threshold = 0.9,
        cpu_memory_limit_gb=60
        ):
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise Exception(f"score_threshold parameter should be between 0.0 and 1.0. score_threshold={score_threshold}")

    os.makedirs(human_output_folder, exist_ok=True)
    os.makedirs(scene_output_folder, exist_ok=True)
    os.makedirs(occlusion_output_folder, exist_ok=True)
    log_path = os.path.join(human_output_folder, "error_log.txt")

    set_memory_limit(cpu_memory_limit_gb)
    checkpoint = assert_file_exist(CHECKPOINTS_FOLDER, "sam2.1_hiera_large.pt")
    model_cfg = os.path.join(SAM2_REPO, "configs/sam2.1/sam2.1_hiera_l.yaml")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(human_output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            if not is_video_empty(os.path.join(human_output_folder, f"{basename_wo_ext}.mp4")):
                continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_video(
            input_path, 
            depth_input_folder, 
            detectron2_input_folder, 
            predictor, 
            score_threshold,
            human_output_folder, 
            scene_output_folder, 
            occlusion_output_folder), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/video_tracking_sam2.py