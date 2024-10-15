from numpy import ndarray
from torch.utils.data import Dataset
import numpy as np
import cv2

from utils.video_utils import frame_from_video

class NpzDataset(Dataset):
    def __init__(self, nparray_data):
        self.frames = nparray_data

    def __getitem__(self, index) -> ndarray:
        return self.frames[index]

    def __len__(self):
        return len(self.frames)
    
class VideoDataset(Dataset):
    def __init__(self, frame_gen):
        self.frames = list(frame_gen)

    def __getitem__(self, index) -> ndarray:
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

class VideoDatasetLazyLoad(Dataset):
    def __init__(self, video):
        self.nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video = video

    def __getitem__(self, index) -> ndarray:
        return frame_from_video(self.video, index)

    def __len__(self):
        return self.nb_frames

class VideoDatasetSlidingWindow(Dataset):
    def __init__(self, frame_gen, window_length, window_stride):
        self.frames = np.array(list(frame_gen))
        self.window_length = window_length
        self.window_stride = window_stride

        self.num_windows = (len(self.frames)+self.window_stride-1) // self.window_stride

    def __getitem__(self, index) -> ndarray:
        start = index * self.window_stride
        end = start + self.window_length
        return self.frames[start:end]

    def __len__(self):
        return self.num_windows