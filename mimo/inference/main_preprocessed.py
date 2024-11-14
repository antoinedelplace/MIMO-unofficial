import sys
sys.path.append(".")

import cv2
import numpy as np

from mimo.utils.general_utils import set_memory_limit, parse_args, assert_file_exist
from mimo.utils.video_utils import frame_gen_from_video
from mimo.inference.inference_pipeline import InferencePipeline

def main(
        vae_encoding_path, 
        rast_2d_joints_path, 
        a_pose_clip_path,
        output_video_path=None, 
        seed=123456, 
        num_scheduler_steps=30,
        guidance_scale=3.5,
        weight_dtype="fp16", 
        num_workers=8, 
        cpu_memory_limit_gb=60,
        input_net_size=768,
        input_net_fps=24,
        neighbor_context_mimo=4,
        batch_size_mimo=20,
        batch_size_vae=12,
    ):
    set_memory_limit(cpu_memory_limit_gb)

    pipe = InferencePipeline(
        seed=seed, 
        num_scheduler_steps=num_scheduler_steps,
        guidance_scale=guidance_scale,
        weight_dtype=weight_dtype, 
        num_workers=num_workers,
        input_net_size=input_net_size,
        input_net_fps=input_net_fps,
        a_pose_raw_path=None,
        depth_anything_encoder=None,
        score_threshold_detectron2=None,
        neighbor_context_mimo=neighbor_context_mimo,
        batch_size_mimo=batch_size_mimo,
        batch_size_depth=None,
        batch_size_detectron2=None,
        batch_size_propainter=None,
        batch_size_reposer=None,
        batch_size_clip=None,
        batch_size_vae=batch_size_vae,
        batch_size_rasterizer=None
    )

    if output_video_path is None:
        output_video_path = vae_encoding_path.replace(".npz", "_output.mp4")
    
    rast_2d_joints_video = cv2.VideoCapture(assert_file_exist(rast_2d_joints_path))
    rast_2d_joints = np.array(list(frame_gen_from_video(rast_2d_joints_video)))
    rast_2d_joints_video.release()
    rast_2d_joints = rast_2d_joints[:, :, :, ::-1] / 255.0
    rast_2d_joints = rast_2d_joints.transpose(0, 3, 1, 2)

    a_pose_clip = dict(np.load(assert_file_exist(a_pose_clip_path)))

    encoded_frames = dict(np.load(assert_file_exist(vae_encoding_path)))

    # Fake frame_gen here
    n_frames = np.shape(rast_2d_joints)[0]
    frame_gen = np.zeros((n_frames, input_net_size, input_net_size, 3))

    pipe.infer_already_preprocessed(
        rast_2d_joints, 
        a_pose_clip["image_embeds"][0], 
        encoded_frames["latent_scene"], 
        encoded_frames["latent_occlusion"], 
        encoded_frames["latent_apose"][0], 
        frame_gen, 
        input_net_size, 
        input_net_size, 
        output_video_path
    )

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))


# accelerate config
#    - No distributed training
#    - numa efficiency
#    - fp16

# accelerate launch mimo/inference/main_preprocessed.py -i ../../data/mimo_video_cropped/demo_motion_parkour.mp4 -a ../../data/iron_man.jpg