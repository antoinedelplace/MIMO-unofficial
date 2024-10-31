import sys
sys.path.append(".")

from mimo.configs.paths import RESIZED_FOLDER, DETECTRON2_FOLDER, DETECTRON2_REPO
sys.path.append(DETECTRON2_REPO)

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog

import os, cv2, torch, tqdm
import numpy as np

from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.detectron2_utils import DetectronBatchPredictor
from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist


def get_cfg_settings():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    return cfg

def visualize(predictions, video, input_path, output_folder):
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
        filename=os.path.join(output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    metadata = metadata = MetadataCatalog.get("coco_2017_test")
    video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)

    frame_gen = frame_gen_from_video(video)

    for frame, pred in tqdm.tqdm(zip(frame_gen, predictions)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_cpu = pred.to(torch.device("cpu"))
        vis_frame = video_visualizer.draw_instance_predictions(frame, pred_cpu)

        # Converts Matplotlib RGB format to OpenCV BGR format
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        output_file.write(vis_frame)

    output_file.release()

    video_path = assert_file_exist(output_folder, basename)
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

def post_processing_detectron2(outputs, output_path=None):
    data_frame_index = []
    data_pred_boxes = []
    data_scores = []
    data_pred_classes = []
    data_pred_masks = []

    for i in range(len(outputs)):
        data_scores.append(outputs[i].scores.cpu().numpy())
        data_frame_index.append(np.ones(len(data_scores[-1]), dtype=np.int32)*i)
        data_pred_boxes.append(outputs[i].pred_boxes.tensor.cpu().numpy())
        data_pred_classes.append(outputs[i].pred_classes.cpu().numpy())
        data_pred_masks.append(outputs[i].pred_masks.cpu().numpy())

    data_frame_index = np.concatenate(data_frame_index)
    data_pred_boxes = np.concatenate(data_pred_boxes)
    data_scores = np.concatenate(data_scores)
    data_pred_classes = np.concatenate(data_pred_classes)
    data_pred_masks = np.concatenate(data_pred_masks)

    if output_path is not None:
        np.savez_compressed(output_path, 
                            data_frame_index=data_frame_index, 
                            data_pred_boxes=data_pred_boxes,
                            data_scores=data_scores,
                            data_pred_classes=data_pred_classes,
                            data_pred_masks=data_pred_masks)
    
    return {"data_frame_index":data_frame_index, 
            "data_pred_boxes":data_pred_boxes,
            "data_scores":data_scores,
            "data_pred_classes":data_pred_classes,
            "data_pred_masks":data_pred_masks}

def run_on_video(input_path, predictor, output_folder):
    assert_file_exist(input_path)
    video = cv2.VideoCapture(input_path)
    
    frame_gen = frame_gen_from_video(video)

    outputs = list(predictor(frame_gen))

    basename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, basename.replace(".mp4", ".npz"))

    # visualize(outputs, video, input_path, output_folder)
    post_processing_detectron2(outputs, output_path)

    video.release()

def main(
        input_folder=RESIZED_FOLDER,
        output_folder=DETECTRON2_FOLDER,
        batch_size=32,
        workers=16,
        cpu_memory_limit_gb=60
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    cfg = get_cfg_settings()
    predictor = DetectronBatchPredictor(cfg, batch_size, workers)
    set_memory_limit(cpu_memory_limit_gb)

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_video(input_path, predictor, output_folder), filename, log_path)

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/human_detection_detectron2.py