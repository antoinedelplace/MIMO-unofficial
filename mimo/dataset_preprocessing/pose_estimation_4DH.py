import sys
sys.path.append(".")

import os, cv2, torch, tqdm
import numpy as np

from mimo.utils.video_utils import frame_gen_from_video
from mimo.utils.general_utils import try_wrapper, set_memory_limit, parse_args, assert_file_exist
from mimo.utils.pose_4DH_utils import HMR2_4dhuman, Human4DConfig
from mimo.utils.skeleton_utils import Skeleton, SMPL_bones, SMPL_hierarchy, Openpose_25_hierarchy, points_animation_linked_3d, get_chains_from_bones_hierarchy
from mimo.utils.rotation_conversions_utils import matrix_to_axis_angle

from mimo.configs.paths import HUMAN_FOLDER, POSES_4DH_FOLDER

def get_cfg():
    cfg = Human4DConfig()

    cfg.SMPL = make_iterable(cfg.SMPL)
    cfg.render.enable = False
    cfg.video.extract_video = False
    cfg.video.source = None
    cfg.video.start_frame=None
    cfg.video.end_frame=None
    cfg.video.start_time=None
    cfg.video.end_time=None
    cfg.post_process.phalp_pkl_path = None

    return cfg

def make_iterable(obj):
    def smpl_iter(self):
        for key, value in self.__dict__.items():
            yield (key, value)
    obj.__class__.__iter__ = smpl_iter
    return obj

def visualize_poses(data_poses, input_path, output_folder):
    # TODO
    skeleton = Skeleton("skeleton")
    skeleton.set_local_position(torch.Tensor([0, 1, 0]))

    skeleton.construct_from_zero_pose(SMPL_bones, SMPL_hierarchy)

    n_joints = len(SMPL_bones)

    poses = torch.from_numpy(np.load("../../data/gWA_sBM_c01_d25_mWA2_ch01.npy"))
    print("poses", poses)
    print("np.shape(poses)", np.shape(poses))

    points = np.zeros((len(poses), n_joints, 3))
    for i in range(len(poses)):
        skeleton.set_pose_axis_angle(poses[i])
        points[i] = skeleton.get_global_position_joints().reshape(n_joints, 3).numpy()
    print("points", points)
    print("np.shape(points)", np.shape(points))

    print("skeleton.get_bone2idx()", [(k.name, v) for k, v in skeleton.get_bone2idx().items()])

    basename = os.path.basename(input_path)
    visualize_joints_3d(points, f"poses_{basename}", output_folder)

def visualize_joints_3d(data_joints_3d, input_path, output_folder):
    # TODO
    chains = get_chains_from_bones_hierarchy(SMPL_hierarchy)

    basename = os.path.basename(input_path)
    points_animation_linked_3d(data_joints_3d,
                               chains,
                               joint_labels=None,
                               fps=24,
                               show=False,
                               save_path=os.path.join(output_folder, f"joints_3d_{basename}"))

def visualize_joints_2d(data_joints_2d, input_path, output_folder, chains=None):
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

    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, f"joints_2d_{basename}"),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    frame_gen = frame_gen_from_video(video)

    data_joints_2d = data_joints_2d*width

    for frame, pred in tqdm.tqdm(zip(frame_gen, data_joints_2d)):
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if chains is not None:
            for chain in chains:
                for i in range(len(chain) - 1):  # Loop through each chain and draw lines
                    pt1 = (int(pred[chain[i], 0]), int(pred[chain[i], 1]))  # Starting point
                    pt2 = (int(pred[chain[i + 1], 0]), int(pred[chain[i + 1], 1]))  # Ending point
                    cv2.line(vis_frame, pt1, pt2, color=(0, 255, 0), thickness=2)  # Green lines connecting points

        for i in range(len(pred)):
            cv2.circle(vis_frame, (int(pred[i, 0]), int(pred[i, 1])), radius=5, color=(255, 0, 0), thickness=-1)

        # Converts Matplotlib RGB format to OpenCV BGR format
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        output_file.write(vis_frame)

    output_file.release()
    video.release()

def get_data_from_4DH(input_path, phalp_tracker):
    phalp_tracker.io_manager.input_path = input_path
    outputs, _ = phalp_tracker.track()

    n_frames = len(outputs)
    n_joints = len(SMPL_bones)
    n_betas = 10

    data_poses = np.zeros((n_frames, n_joints, 3))
    data_cam_trans = np.zeros((n_frames, 3))
    data_betas = np.zeros((n_frames, n_betas))
    data_joints_3d = np.zeros((n_frames, 45, 3))
    data_joints_2d = np.zeros((n_frames, 45, 2))

    # If no data detected, get the data at the previous frame
    smpl_temp = {
        "global_orient": np.zeros((3, 3)),
        "body_pose": np.zeros((n_joints-1, 3, 3)),
        "betas": np.zeros((n_betas)),
    }
    camera_temp = np.zeros((3))
    joints_3d_temp = np.zeros((45, 3))
    joints_2d_temp = np.zeros((45, 2))
    for i in range(len(outputs)):
        if len(outputs[i]["smpl"]) > 0:
            smpl_temp = outputs[i]["smpl"][0]

        data_poses[i, 0, :] = matrix_to_axis_angle(torch.from_numpy(smpl_temp["global_orient"])).numpy()
        data_poses[i, 1:, :] = matrix_to_axis_angle(torch.from_numpy(smpl_temp["body_pose"])).numpy()
        data_betas[i] = smpl_temp["betas"]

        if len(outputs[i]["camera"]) > 0:
            camera_temp = outputs[i]["camera"][0]
        data_cam_trans[i] = camera_temp

        if len(outputs[i]["3d_joints"]) > 0:
            joints_3d_temp = outputs[i]["3d_joints"][0]
        data_joints_3d[i] = joints_3d_temp

        if len(outputs[i]["2d_joints"]) > 0:
            joints_2d_temp = outputs[i]["2d_joints"][0].reshape(-1, 2)
        data_joints_2d[i] = joints_2d_temp
    
    return {"data_poses": data_poses,
            "data_cam_trans": data_cam_trans,
            "data_betas": data_betas,
            "data_joints_3d": data_joints_3d,
            "data_joints_2d": data_joints_2d}
    
def run_on_video(input_path, phalp_tracker, output_folder):
    assert_file_exist(input_path)
    basename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, basename.replace(".mp4", ".npz"))

    data_4DH = get_data_from_4DH(input_path, phalp_tracker)

    np.savez_compressed(output_path, **data_4DH)

    # outputs = dict(np.load(output_path))
    # data_joints_2d = outputs["data_joints_2d"]
    # data_joints_3d = outputs["data_joints_3d"]
    # data_poses = outputs["data_poses"]
    # openpose_chains = get_chains_from_bones_hierarchy(Openpose_25_hierarchy)
    #visualize_joints_2d(data_joints_2d[:, :25, :], input_path, openpose_chains)
    # visualize_joints_3d(data_joints_3d, input_path)
    # visualize_poses(data_poses, input_path)

def main(
        input_folder=HUMAN_FOLDER,
        output_folder=POSES_4DH_FOLDER,
        cpu_memory_limit_gb=60
        ):
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_log.txt")

    set_memory_limit(cpu_memory_limit_gb)

    phalp_tracker = HMR2_4dhuman(get_cfg())

    # input_files = ["03ecb2c8-7e3f-42df-96bc-9723335397d9-original.mp4"]
    input_files = sorted(os.listdir(input_folder))
    output_files = sorted([os.path.splitext(os.path.basename(file))[0] for file in os.listdir(output_folder)])

    for filename in tqdm.tqdm(input_files):
        basename_wo_ext = os.path.splitext(os.path.basename(filename))[0]
        if basename_wo_ext in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: run_on_video(input_path, phalp_tracker, output_folder), filename, log_path)


if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python mimo/dataset_preprocessing/pose_estimation_4DH.py



# Renderer needs to be removed to avoid OpenGL errors
# in 4D-Humans/hmr2/models/__init__.py line 84
# model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)

# Remove automatic saving files to speed up inference
# in PHALP/phalp/trackers/PHALP.py line 264
# Remove joblib.dump(final_visuals_dic, pkl_path, compress=3)