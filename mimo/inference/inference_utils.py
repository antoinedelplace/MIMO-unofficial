import sys
sys.path.append(".")

import tempfile
import shutil
import os, cv2

def create_video_from_frames(frames, fps):
    temp_dir = tempfile.mkdtemp()

    video_path = os.path.join(temp_dir, "output_video.mp4")

    width, height, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    return video_path

def remove_tmp_dir(video_path):
    folder_path = os.path.dirname(video_path)
    system_temp_dir = tempfile.gettempdir()

    if os.path.commonpath([system_temp_dir, folder_path]) == system_temp_dir:
        shutil.rmtree(folder_path)