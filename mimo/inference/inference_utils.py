import sys
sys.path.append(".")

import tempfile
import shutil
import os
import inspect

from mimo.utils.video_utils import save_video

def create_video_from_frames(frames, fps):
    temp_dir = tempfile.mkdtemp()

    video_path = os.path.join(temp_dir, "output_video.mp4")

    width, height, _ = frames[0].shape

    save_video(video_path, fps, width, height, frames)

    return video_path

def remove_tmp_dir(video_path):
    folder_path = os.path.dirname(video_path)
    system_temp_dir = tempfile.gettempdir()

    if os.path.commonpath([system_temp_dir, folder_path]) == system_temp_dir:
        shutil.rmtree(folder_path)

def get_extra_kwargs_scheduler(generator, eta, noise_scheduler):
    extra_step_kwargs = {}

    scheduler_params = set(inspect.signature(noise_scheduler.step).parameters.keys())
    if "eta" in scheduler_params:
        extra_step_kwargs["eta"] = eta

    if "generator" in scheduler_params:
        extra_step_kwargs["generator"] = generator

    return extra_step_kwargs