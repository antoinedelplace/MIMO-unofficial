from numpy import ndarray
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, frame_gen):
        self.frames = list(frame_gen)

    def __getitem__(self, index) -> ndarray:
        return self.frames[index]

    def __len__(self):
        return len(self.frames)