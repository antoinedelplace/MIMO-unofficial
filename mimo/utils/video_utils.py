import cv2
import hashlib

def hash_file(file_path, hash_algo='md5'):
    """Compute the hash of a file."""
    hash_func = hashlib.new(hash_algo)
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def frame_gen_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def frame_from_video(video, index):
    video.set(cv2.CAP_PROP_POS_FRAMES, index)  # Set to the required frame
    success, frame = video.read()
    if success:
        return frame
    else:
        raise ValueError(f"Could not read frame {index} from video.")

def save_video(video_path, fps, width, height, frames):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()