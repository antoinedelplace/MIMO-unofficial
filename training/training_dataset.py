import sys
sys.path.append(".")

import os, cv2
import numpy as np
from torch.utils.data import Dataset

from utils.video_utils import frame_gen_from_video

from dataset_preprocessing.video_tracking_sam2 import get_index_first_frame_with_character

from configs.paths import RASTERIZED_2D_JOINTS_FOLDER, APOSE_REF_FOLDER, APOSE_CLIP_EMBEDS_FOLDER, ENCODED_OCCLUSION_SCENE_FOLDER, DETECTRON2_FOLDER


def get_begin_frame(input_files, window_length, window_stride, detectron_score_threshold):
    input_filename = []
    begin_frame_scene = []
    begin_frame_video = []
    for filename in input_files:
        data = np.load(os.path.join(ENCODED_OCCLUSION_SCENE_FOLDER, f"{filename}.npz"))

        n_frames_scene = len(data["latent_scene"])
        n_frames_occlusion = len(data["latent_occlusion"])
        n_frames_video = len(data["latent_video"])

        if n_frames_scene != n_frames_occlusion:
            print(f"Something is wrong with {filename}. Number of frames for scene ({n_frames_scene}) is not the same as occlusion ({n_frames_occlusion}). We take into account the number of frames of the scene.")

        i_first_frame = 0
        if n_frames_scene != n_frames_video:
            detectron2_data = dict(np.load(os.path.join(DETECTRON2_FOLDER, f"{filename}.npz")))
            # Here might have a problem: SAM2 first frame may not be frame with first character
            i_first_frame = get_index_first_frame_with_character(detectron2_data, detectron_score_threshold)
        
        num_windows = (n_frames_scene-window_length+window_stride-1) // window_stride
        for i_window in range(num_windows):
            input_filename.append(filename)
            begin_frame_scene.append(0 + i_window * window_stride)
            begin_frame_video.append(i_first_frame + i_window * window_stride)
    
    return input_filename, begin_frame_scene, begin_frame_video


class TrainingDataset(Dataset):
    def __init__(self, window_length, window_stride, detectron_score_threshold = 0.9):
        super().__init__()

        self.window_length = window_length

        encoded_files = set([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(ENCODED_OCCLUSION_SCENE_FOLDER)])
        rast_2d_joints_files = set([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(RASTERIZED_2D_JOINTS_FOLDER)])
        apose_files = set([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(APOSE_REF_FOLDER)])
        apose_clip_files = set([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(APOSE_CLIP_EMBEDS_FOLDER)])
        input_files = sorted(list(set.intersection(encoded_files, rast_2d_joints_files, apose_files, apose_clip_files)))

        self.input_filename, self.begin_frame_scene, self.begin_frame_video = get_begin_frame(input_files, window_length, window_stride, detectron_score_threshold)

    def __getitem__(self, index):
        filename = self.input_filename[index]
        begin_frame_scene = self.begin_frame_scene[index]
        begin_frame_video = self.begin_frame_video[index]

        a_pose = cv2.imread(f"{filename}.png")

        rast_2d_joints_video = cv2.VideoCapture(os.path.join(RASTERIZED_2D_JOINTS_FOLDER, f"{filename}.mp4"))
        rast_2d_joints = frame_gen_from_video(rast_2d_joints_video)
        rast_2d_joints_video.release()
        rast_2d_joints = rast_2d_joints[begin_frame_scene:begin_frame_scene+self.window_length]

        a_pose_clip = dict(np.load(os.path.join(APOSE_CLIP_EMBEDS_FOLDER, f"{filename}.npz")))
        a_pose_clip["image_embeds"] = a_pose_clip["image_embeds"][begin_frame_scene:begin_frame_scene+self.window_length]

        encoded_frames = dict(np.load(os.path.join(ENCODED_OCCLUSION_SCENE_FOLDER, f"{filename}.npz")))
        encoded_frames["latent_scene"] = encoded_frames["latent_scene"][begin_frame_scene:begin_frame_scene+self.window_length]
        encoded_frames["latent_occlusion"] = encoded_frames["latent_occlusion"][begin_frame_scene:begin_frame_scene+self.window_length]
        encoded_frames["latent_video"] = encoded_frames["latent_video"][begin_frame_video:begin_frame_video+self.window_length]

        return a_pose, rast_2d_joints, encoded_frames

    def __len__(self):
        return len(self.begin_frame)