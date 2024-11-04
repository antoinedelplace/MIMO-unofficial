import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist
from mimo.utils.vae_encoding_utils import download_vae, VaeBatchPredictor

from mimo.configs.paths import FILLED_SCENE_FOLDER, OCCLUSION_FOLDER, ENCODED_OCCLUSION_SCENE_FOLDER, RESIZED_FOLDER, APOSE_REF_FOLDER, RAW_FOLDER


def visualize(vae, latent, video, input_path, output_folder):
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

    for frames in vae.decode(latent):
        for frame in frames:
            output_file.write(frame)

    output_file.release()

def get_original_width_height(basename, raw_input_folder):
    video_path = assert_file_exist(raw_input_folder, basename)
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    return width, height

def remove_mirror_occlusion_for_vertical_videos(frame_gen, ori_width, ori_height, input_net_size):
    if ori_width < ori_height:
        scale = input_net_size/ori_height
        for frame in frame_gen:
            frame[:, :int(scale*(ori_height-ori_width)//2), :] = 0
            frame[:, -int(scale*(ori_height-ori_width+1)//2):, :] = 0
            yield frame
    else:
        for frame in frame_gen:
            yield frame

def inpaint_mirror_for_vertical_videos(frame_gen, ori_width, ori_height, input_net_size, input_path):
    if ori_width < ori_height:
        scale = input_net_size/ori_height

        assert_file_exist(input_path)
        video_scene = cv2.VideoCapture(input_path)
        frame_gen_scene2 = frame_gen_from_video(video_scene)

        for frame, frame_scene in zip(frame_gen, frame_gen_scene2):
            left_pad = int(scale*(ori_height-ori_width)//2)
            right_pad = int(scale*(ori_height-ori_width+1)//2)
            frame[:, :left_pad, :] = frame_scene[:, :left_pad, :]
            frame[:, -right_pad:, :] = frame_scene[:, -right_pad:, :]
            yield frame
    else:
        for frame in frame_gen:
            yield frame

def get_latent_scene(input_path, vae, output_folder=None):
    assert_file_exist(input_path)
    video_scene = cv2.VideoCapture(input_path)
    frame_gen_scene = np.array(list(frame_gen_from_video(video_scene)))
    print("np.shape(frame_gen_scene)", np.shape(frame_gen_scene))

    latent_scene = np.concatenate(list(vae.encode(frame_gen_scene)))
    print("np.shape(latent_scene)", np.shape(latent_scene))
    if output_folder is not None:
        visualize(vae, latent_scene, video_scene, input_path, output_folder)

    video_scene.release()

    return latent_scene

def get_latent_occlusion(basename, occlusion_input_folder, ori_width, ori_height, vae, output_folder=None):
    video_path = assert_file_exist(occlusion_input_folder, basename)
    video_occlusion = cv2.VideoCapture(video_path)
    frame_gen_occlusion = frame_gen_from_video(video_occlusion)

    frame_gen_occlusion = np.array(list(frame_gen_occlusion))
    print("np.shape(frame_gen_occlusion)", np.shape(frame_gen_occlusion))

    input_net_size = int(video_occlusion.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_gen_occlusion_filtered = remove_mirror_occlusion_for_vertical_videos(frame_gen_occlusion, ori_width, ori_height, input_net_size)

    frame_gen_occlusion_filtered = np.array(list(frame_gen_occlusion_filtered))
    print("np.shape(frame_gen_occlusion_filtered)", np.shape(frame_gen_occlusion_filtered))

    latent_occlusion = np.concatenate(list(vae.encode(frame_gen_occlusion_filtered)))
    print("np.shape(latent_occlusion)", np.shape(latent_occlusion))
    if output_folder is not None:
        visualize(vae, latent_occlusion, video_occlusion, video_path, output_folder)
    
    video_occlusion.release()

    return latent_occlusion

def get_latent_video(input_path, basename, resized_folder, ori_width, ori_height, vae, output_folder=None):
    video_path = assert_file_exist(resized_folder, basename)
    video_ori = cv2.VideoCapture(video_path)
    frame_gen_video = frame_gen_from_video(video_ori)

    frame_gen_video = np.array(list(frame_gen_video))
    print("np.shape(frame_gen_video)", np.shape(frame_gen_video))

    input_net_size = int(video_ori.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_gen_video_filtered = inpaint_mirror_for_vertical_videos(frame_gen_video, ori_width, ori_height, input_net_size, input_path)

    frame_gen_video_filtered = np.array(list(frame_gen_video_filtered))
    print("np.shape(frame_gen_video_filtered)", np.shape(frame_gen_video_filtered))

    latent_video = np.concatenate(list(vae.encode(frame_gen_video_filtered)))
    print("np.shape(latent_video)", np.shape(latent_video))
    if output_folder is not None:
        visualize(vae, latent_video, video_ori, input_path, output_folder)
    
    video_ori.release()

    return latent_video

def get_latent_apose(basename, apose_ref_folder, vae):
    image_path = assert_file_exist(apose_ref_folder, basename.replace(".mp4", ".png"))
    a_pose = cv2.imread(image_path)
    latent_apose = np.concatenate(list(vae.encode([a_pose])))
    print("np.shape(latent_apose)", np.shape(latent_apose))

    return latent_apose

def run_on_video(input_path, occlusion_input_folder, resized_folder, apose_ref_folder, raw_input_folder, vae, output_folder):
    basename = os.path.basename(input_path)

    ori_width, ori_height = get_original_width_height(basename, raw_input_folder)

    latent_scene = get_latent_scene(input_path, vae)

    latent_occlusion = get_latent_occlusion(basename, occlusion_input_folder, ori_width, ori_height, vae)

    latent_video = get_latent_video(input_path, basename, resized_folder, ori_width, ori_height, vae)

    latent_apose = get_latent_apose(basename, apose_ref_folder, vae)
    
    output_path = os.path.join(output_folder, basename.replace(".mp4", ".npz"))
    np.savez_compressed(output_path, 
                        latent_scene=latent_scene, 
                        latent_occlusion=latent_occlusion,
                        latent_video=latent_video,
                        latent_apose=latent_apose)

def main(
        scene_input_folder=FILLED_SCENE_FOLDER,
        occlusion_input_folder=OCCLUSION_FOLDER,
        resized_folder=RESIZED_FOLDER,
        apose_ref_folder=APOSE_REF_FOLDER,
        raw_input_folder=RAW_FOLDER,
        output_folder=ENCODED_OCCLUSION_SCENE_FOLDER,
        batch_size=16,
        workers=8,
        cpu_memory_limit_gb=60
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    download_vae()

    set_memory_limit(cpu_memory_limit_gb)

    vae = VaeBatchPredictor(batch_size, workers)

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(scene_input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(scene_input_folder, filename)
        try_wrapper(lambda: run_on_video(input_path, occlusion_input_folder, resized_folder, apose_ref_folder, raw_input_folder, vae, output_folder), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/vae_encoding.py