import torch, cv2

from torch.utils.data import DataLoader
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2

from utils.torch_utils import VideoDataset


class BatchPredictor(DepthAnythingV2):
    def __init__(
        self, 
        batch_size: int, workers: int,
        width, height,
        *args,
        **kargs
    ):
        super(BatchPredictor, self).__init__(*args, **kargs)

        self.batch_size = batch_size
        self.workers = workers
        self.width = width
        self.height = height

    def collate(self, batch):
        data = []
        for image in batch:
            # the model expects RGB inputs
            image = image[:, :, ::-1] / 255.0

            image = cv2.resize(image, ((self.width // 14) * 14, (self.height // 14) * 14), interpolation=cv2.INTER_LINEAR)
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append(image)
        return torch.stack(data, dim=0)

    def infer_video(self, frame_gen):
        dataset = VideoDataset(frame_gen)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.collate,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in loader:
                batch_gpu = batch.to("cuda")
                depth = self.forward(batch_gpu)
                depth = F.interpolate(depth[:, None], (self.height, self.width), mode="bilinear", align_corners=True)[:, 0]
                yield depth.cpu().numpy()