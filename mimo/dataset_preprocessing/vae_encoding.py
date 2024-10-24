import sys
sys.path.append(".")

import os, cv2, tqdm
import numpy as np

from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args
from mimo.utils.vae_encoding_utils import download_vae, VaeBatchPredictor

from mimo.configs.paths import FILLED_SCENE_FOLDER, OCCLUSION_FOLDER, ENCODED_OCCLUSION_SCENE_FOLDER, RESIZED_FOLDER, APOSE_REF_FOLDER


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

def run_on_video(input_path, occlusion_input_folder, resized_folder, apose_ref_folder, vae, output_folder):
    basename = os.path.basename(input_path)

    video = cv2.VideoCapture(input_path)
    frame_gen = frame_gen_from_video(video)

    latent_scene = np.concatenate(list(vae.encode(frame_gen)))
    print("np.shape(latent_scene)", np.shape(latent_scene))
    # visualize(latent_scene, video, input_path)

    video.release()

    video = cv2.VideoCapture(os.path.join(occlusion_input_folder, basename))
    frame_gen = frame_gen_from_video(video)

    latent_occlusion = np.concatenate(list(vae.encode(frame_gen)))
    print("np.shape(latent_occlusion)", np.shape(latent_occlusion))
    # visualize(vae, latent_occlusion, video, input_path, output_folder)
    
    video.release()

    video = cv2.VideoCapture(os.path.join(resized_folder, basename))
    frame_gen = frame_gen_from_video(video)

    latent_video = np.concatenate(list(vae.encode(frame_gen)))
    print("np.shape(latent_video)", np.shape(latent_video))
    # visualize(vae, latent_video, video, input_path, output_folder)
    
    video.release()

    a_pose = cv2.imread(os.path.join(apose_ref_folder, basename).replace(".mp4", ".png"))
    latent_apose = np.concatenate(list(vae.encode([a_pose])))
    print("np.shape(latent_apose)", np.shape(latent_apose))
    
    output_path = os.path.join(output_folder, basename).replace(".mp4", ".npz")
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
        try_wrapper(lambda: run_on_video(input_path, occlusion_input_folder, resized_folder, apose_ref_folder, vae, output_folder), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/vae_encoding.py