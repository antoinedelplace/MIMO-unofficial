import sys
sys.path.append(".")

import os
import numpy as np

from mimo.utils.general_utils import set_memory_limit, parse_args, assert_file_exist
from mimo.utils.vae_encoding_utils import download_vae, VaeBatchPredictor

from mimo.dataset_preprocessing.vae_encoding import visualize_image, visualize_video


def main(
        vae_encoding_path,
        output_folder=None,
        batch_size=16,
        workers=8,
        cpu_memory_limit_gb=60
        ):
    assert_file_exist(vae_encoding_path)

    download_vae()

    set_memory_limit(cpu_memory_limit_gb)

    vae = VaeBatchPredictor(batch_size, workers)

    basename_wo_ext = os.path.splitext(os.path.basename(vae_encoding_path))[0]

    encoded_frames = dict(np.load(vae_encoding_path))

    if output_folder is None:
        output_folder = os.path.dirname(vae_encoding_path)
    
    width = 768
    height = 768
    fps = 24

    visualize_video(vae, encoded_frames["latent_scene"], f"{basename_wo_ext}_scene.mp4", width, height, fps, output_folder)
    visualize_video(vae, encoded_frames["latent_occlusion"], f"{basename_wo_ext}_occlusion.mp4", width, height, fps, output_folder)
    visualize_video(vae, encoded_frames["latent_video"], f"{basename_wo_ext}_video.mp4", width, height, fps, output_folder)
    visualize_image(vae, encoded_frames["latent_apose"], f"{basename_wo_ext}_apose.png", output_folder)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/visualize_vae_encoding.py