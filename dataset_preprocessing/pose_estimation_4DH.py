import sys
sys.path.append(".")

import os, cv2, torch, tqdm
import numpy as np

from utils.video_utils import frame_gen_from_video
from utils.general_utils import try_wrapper, set_memory_limit
from utils.pose_4DH_utils import HMR2_4dhuman, Human4DConfig
from utils.skeleton_utils import Skeleton, SMPL_bones, SMPL_hierarchy, points_animation_linked_3d, get_chains_from_bones_hierarchy

input_folder = "../../data/human_data/"
output_folder = "../../data/poses_4DH_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

batch_size = 12
workers = 8
input_size = 768
set_memory_limit(60)

cfg = Human4DConfig()

def make_iterable(obj):
    def smpl_iter(self):
        for key, value in self.__dict__.items():
            yield (key, value)
    obj.__class__.__iter__ = smpl_iter
    return obj

cfg.SMPL = make_iterable(cfg.SMPL)
cfg.render.enable = False
cfg.video.extract_video = False
cfg.video.source = None
cfg.video.start_frame=None
cfg.video.end_frame=None
cfg.video.start_time=None
cfg.video.end_time=None
cfg.post_process.phalp_pkl_path = None

phalp_tracker = HMR2_4dhuman(cfg)

def run_on_video(input_path):
    # skeleton = Skeleton("skeleton")
    # skeleton.set_local_position(torch.Tensor([0, 1, 0]))

    # skeleton.construct_from_zero_pose(SMPL_bones, SMPL_hierarchy)

    # n_joints = len(SMPL_bones)
    # chains = get_chains_from_bones_hierarchy(SMPL_hierarchy)
    # print("chains", chains)

    # poses = torch.from_numpy(np.load("../../data/gWA_sBM_c01_d25_mWA2_ch01.npy"))
    # print("poses", poses)
    # print("np.shape(poses)", np.shape(poses))

    # points = np.zeros((len(poses), n_joints, 3))
    # for i in range(len(poses)):
    #     skeleton.set_pose_axis_angle(poses[i])
    #     points[i] = skeleton.get_global_position_joints().reshape(n_joints, 3).numpy()
    # print("points", points)
    # print("np.shape(points)", np.shape(points))

    # print("skeleton.get_bone2idx()", [(k.name, v) for k, v in skeleton.get_bone2idx().items()])

    # points_animation_linked_3d(points,
    #                            chains,
    #                            joint_labels=None,
    #                            fps=24,
    #                            show=False,
    #                            save_path=os.path.join(output_folder, "test.gif"))

    video = cv2.VideoCapture(input_path)

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

    phalp_tracker.io_manager.input_path = input_path
    final_visuals_dic, pkl_path = phalp_tracker.track()

    # print("final_visuals_dic", final_visuals_dic)
    print("len(final_visuals_dic)", len(final_visuals_dic))

    video.release()


input_files = ["../../data/human_data/03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
run_on_video(input_files[0])

# input_files = sorted(os.listdir(input_folder))
output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

for filename in tqdm.tqdm(input_files):
    basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
    if basename_wo_ext in output_files:
        continue

    input_path = os.path.join(input_folder, filename)
    # try_wrapper(lambda: run_on_video(input_path), filename, log_path)


# python dataset_preprocessing/pose_estimation_4DH.py



# Renderer needs to be removed to avoid OpenGL errors
# in 4D-Humans/hmr2/models/__init__.py line 84
# model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)