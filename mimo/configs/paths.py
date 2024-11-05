import os

DATA_FOLDER        = "../../data"
CHECKPOINTS_FOLDER = "../../checkpoints"

# Github Repo
ANIMATE_ANYONE_REPO = "../AnimateAnyone"
DEPTH_ANYTHING_REPO = "../Depth-Anything-V2"
DETECTRON2_REPO     = "../detectron2"
HUMANS_4D_REPO      = "../4D-Humans"
PHALP_REPO          = "../PHALP"
NVDIFFRAST_REPO     = "../nvdiffrast"
PROPAINTER_REPO     = "../ProPainter"
SAM2_REPO           = "../sam2"
MIMIC_MOTION_REPO   = "../MimicMotion"

# Data
RAW_FOLDER                     = os.path.join(DATA_FOLDER, "data")
RESIZED_FOLDER                 = os.path.join(DATA_FOLDER, "resized_data")
DEPTH_FOLDER                   = os.path.join(DATA_FOLDER, "depth_data")
DETECTRON2_FOLDER              = os.path.join(DATA_FOLDER, "detectron2_data")
POSES_4DH_FOLDER               = os.path.join(DATA_FOLDER, "poses_4DH_data")
HUMAN_FOLDER                   = os.path.join(DATA_FOLDER, "human_data")
OCCLUSION_FOLDER               = os.path.join(DATA_FOLDER, "occlusion_data")
SCENE_FOLDER                   = os.path.join(DATA_FOLDER, "scene_data")
RASTERIZED_2D_JOINTS_FOLDER    = os.path.join(DATA_FOLDER, "rasterized_2D_joints_data")
APOSE_REF_FOLDER               = os.path.join(DATA_FOLDER, "apose_ref_data")
APOSE_CLIP_EMBEDS_FOLDER       = os.path.join(DATA_FOLDER, "apose_clip_embeds_data")
FILLED_SCENE_FOLDER            = os.path.join(DATA_FOLDER, "filled_scene_data")
ENCODED_OCCLUSION_SCENE_FOLDER = os.path.join(DATA_FOLDER, "encoded_occlusion_scene_data")

# Checkpoints
IMAGE_ENCODER_FOLDER  = os.path.join(CHECKPOINTS_FOLDER, "sd-image-variations-diffusers")
BASE_MODEL_FOLDER     = os.path.join(CHECKPOINTS_FOLDER, "stable-diffusion-v1-5")
ANIMATE_ANYONE_FOLDER = os.path.join(CHECKPOINTS_FOLDER, "AnimateAnyone")
DWPOSE_FOLDER         = os.path.join(CHECKPOINTS_FOLDER, "DWPose")
VAE_FOLDER            = os.path.join(CHECKPOINTS_FOLDER, "sd-vae-ft-mse")

# Training
ML_RUNS = "./mlruns"
TRAIN_OUTPUTS = "./train_outputs"