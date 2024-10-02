import sys
sys.path.append(".")

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

import os, cv2

from utils.video_utils import frame_gen_from_video
from utils.general_utils import time_it
from utils.detectron2_utils import BatchPredictor

input_path = "../../data/resized_data/df5afa6a-b7a2-485e-ae12-e3d045e4ebc0-original.mp4"

# class_names = MetadataCatalog.get("coco_2017_train").thing_classes
# print("class_names", class_names)

def get_cfg_settings():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    return cfg

# def process_predictions(frame, predictions):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     if "panoptic_seg" in predictions:
#         panoptic_seg, segments_info = predictions["panoptic_seg"]
#         vis_frame = video_visualizer.draw_panoptic_seg_predictions(
#             frame, panoptic_seg.to(self.cpu_device), segments_info
#         )
#     elif "instances" in predictions:
#         predictions = predictions["instances"].to(self.cpu_device)
#         vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
#     elif "sem_seg" in predictions:
#         vis_frame = video_visualizer.draw_sem_seg(
#             frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
#         )

#     # Converts Matplotlib RGB format to OpenCV BGR format
#     vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
#     return vis_frame

@time_it
def run_on_video(input_path):
    cfg = get_cfg_settings()
    batch_size = 24
    workers = 8
    predictor = BatchPredictor(cfg, batch_size, workers)
    
    outputs = list(predictor(input_path))
    print(outputs)

@time_it
def run_on_video2(input_path):
    video = cv2.VideoCapture(input_path)

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

    frame_gen = frame_gen_from_video(video)

    cfg = get_cfg_settings()
    predictor = DefaultPredictor(cfg)

    def get_predictions():
        for frame in frame_gen:
            yield predictor(frame)
    
    outputs = list(get_predictions())
    print(outputs)

    video.release()

run_on_video(input_path)
# run_on_video2(input_path)

# outputs = predictor(video)

# print(outputs)
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)

# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow(out.get_image()[:, :, ::-1])

# python dataset_preprocessing/human_detection_detectron2.py