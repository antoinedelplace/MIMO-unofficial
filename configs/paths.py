import os

DATA_FOLDER = "../../data"
CHECKPOINTS_FOLDER = "../../checkpoints"

# Data
APOSE_REF_FOLDER = os.path.join(DATA_FOLDER, "apose_ref_data")
APOSE_CLIP_EMBEDS_FOLDER = os.path.join(DATA_FOLDER, "apose_clip_embeds_data")

# Checkpoints
IMAGE_ENCODER_FOLDER = os.path.join(CHECKPOINTS_FOLDER, "sd-image-variations-diffusers")