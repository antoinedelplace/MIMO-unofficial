import sys
sys.path.append(".")

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog

import os, cv2, torch, tqdm

from utils.video_utils import frame_gen_from_video
from utils.general_utils import time_it
from utils.detectron2_utils import BatchPredictor

input_path = "../../data/resized_data/df5afa6a-b7a2-485e-ae12-e3d045e4ebc0-original.mp4"
output_folder = "../../data/detectron2_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

def get_cfg_settings():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    return cfg

def visualize(predictions, video, cfg):
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

    metadata = metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
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

    video = cv2.VideoCapture(os.path.join(output_folder, basename))

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("basename", basename)
    print("width", width)
    print("height", height)
    print("frames_per_second", frames_per_second)
    print("num_frames", num_frames)

def run_on_video(input_path):
    video = cv2.VideoCapture(input_path)

    cfg = get_cfg_settings()
    batch_size = 24
    workers = 8
    predictor = BatchPredictor(cfg, batch_size, workers)
    
    frame_gen = frame_gen_from_video(video)

    outputs = list(predictor(frame_gen))

    print(outputs)
    visualize(outputs, video, cfg)

    video.release()

run_on_video(input_path)


# python dataset_preprocessing/human_detection_detectron2.py