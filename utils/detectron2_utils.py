from typing import Iterable, List, NamedTuple

import cv2
import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.modeling import build_model
from detectron2.structures import Instances
from numpy import ndarray
from torch.utils.data import DataLoader, Dataset

from utils.video_utils import frame_gen_from_video


class Prediction(NamedTuple):
    x: float
    y: float
    width: float
    height: float
    score: float
    class_name: str


class VideoDataset(Dataset):
    def __init__(self, input_video_path: str):
        self.video = cv2.VideoCapture(input_video_path)
        self.frame_gen = list(frame_gen_from_video(self.video))

    def __getitem__(self, index) -> ndarray:
        return self.frame_gen[index]

    def __len__(self):
        return len(self.frame_gen)
    
    def __del__(self):
        # Release video when the dataset object is deleted
        if self.video.isOpened():
            self.video.release()


class BatchPredictor:
    def __init__(self, cfg: CfgNode, batch_size: int, workers: int):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.batch_size = batch_size
        self.workers = workers
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __collate(self, batch):
        data = []
        for image in batch:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            image = self.aug.get_transform(image).apply_image(image)
            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append({"image": image, "height": height, "width": width})
        return data

    def __call__(self, input_video_path) -> Iterable[List[Prediction]]:
        dataset = VideoDataset(input_video_path)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.__collate,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in loader:
                results: List[Instances] = self.model(batch)
                yield from [result['instances'] for result in results]