# MIMO-unofficial
Unofficial implementation of MIMO (MImicking anyone anywhere with complex Motions and Object interactions)

My blog post: [![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@delplaceantoine/mastering-mimo-mimicking-anyone-anywhere-with-complex-motions-and-object-interactions-e8598d9d97d6)
Original paper: [![arXiv](https://img.shields.io/badge/arXiv-2409.16160-00ff00.svg)](https://arxiv.org/abs/2409.16160)

## 🎯 Overview
This repository offers a comprehensive pipeline for training and inference to transform character appearances and motions in videos. As part of the video-to-video generation category, this framework enables dynamic character modification with optional inputs, including an avatar photo and/or 3D animations.

## ⚒️ Installation and Setup
Tests were made using:
```bash
Cuda 12.2
Python 3.10.12
torch 2.4.1
```

### Clone repository and Install requirements
```bash
git clone git@github.com:antoinedelplace/MIMO-unofficial.git
cd MIMO-unofficial/
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Folder architecture
See `configs/paths.py`
```bash
├── code
│   ├── MIMO-unofficial
│   ├── AnimateAnyone
│   ├── Depth-Anything-V2
│   ├── 4D-Humans
│   ├── PHALP
│   ├── detectron2
│   ├── sam2
│   ├── nvdiffrast
│   ├── MimicMotion
│   └── ProPainter
├── checkpoints
└── data
```

### Clone external repositories and Install requirements
- [AnimateAnyone](https://github.com/novitalabs/AnimateAnyone)
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [PHALP](https://github.com/brjathu/PHALP)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [sam2](https://github.com/facebookresearch/sam2)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [MimicMotion](https://github.com/Tencent/MimicMotion)
- [ProPainter](https://github.com/sczhou/ProPainter.git)

### Download checkpoints
- [depth_anything_v2_vitl.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
- [MimicMotion_1-1.pth](https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth)

### Extra steps
- For `mimo/dataset_preprocessing/pose_estimation_4DH.py`
    - Download SMPL model and put it in `MIMO-unofficial` and in `MIMO-unofficial/data`
    [basicModel_neutral_lbs_10_207_0_v1.0.0.pkl](https://huggingface.co/spaces/brjathu/HMR2.0/resolve/e5201da358ccbc04f4a5c4450a302fcb9de571dd/data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl)

    - Uninstall phalp and hmr2 packages to use the clone repositoy instead
    ```bash
    pip uninstall phalp hmr2
    ```

    - Renderer needs to be removed to avoid OpenGL errors
    in `4D-Humans/hmr2/models/__init__.py` line 84:
    ```bash
    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)
    ```

    - Remove automatic saving files to speed up inference
    in `PHALP/phalp/trackers/PHALP.py` line 264:
    Remove `joblib.dump(final_visuals_dic, pkl_path, compress=3)`

- For `mimo/dataset_preprocessing/get_apose_ref.py`
    - If you need DWPose to extract 2D pose from an image
    ```bash
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.1"
    mim install "mmdet>=3.1.0"
    mim install "mmpose>=1.1.0"
    ```

    - in `AnimateAnyone/src/models/unet_2d_blocks.py` line 9:
    ```bash
    from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
    ```

- For `mimo/training/main.py`
    - in `AnimateAnyone/src/models/mutual_self_attention.py` line 48:
    ```bash
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            batch_size=batch_size,
            fusion_blocks=fusion_blocks,
        )
    ```

    - in `AnimateAnyone/src/models/unet_2d_blocks.py` line 9:
    ```bash
    from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
    ```


## 🚀 Run scripts
### Dataset Preprocessing
1. `mimo/dataset_preprocessing/video_sampling_resizing.py`
1. `mimo/dataset_preprocessing/remove_duplicate_videos.py`
1. `mimo/dataset_preprocessing/human_detection_detectron2.py`
1. `mimo/dataset_preprocessing/depth_estimation.py`
1. `mimo/dataset_preprocessing/video_tracking_sam2.py`
1. `mimo/dataset_preprocessing/video_inpainting.py`
1. `mimo/dataset_preprocessing/get_apose_ref.py`
1. `mimo/dataset_preprocessing/vae_encoding.py`
1. `mimo/dataset_preprocessing/clip_embedding.py`
1. `mimo/dataset_preprocessing/pose_estimation_4DH.py`
1. `mimo/dataset_preprocessing/rasterizer_2d_joints.py`

### Inference
```bash
accelerate config
    - No distributed training
    - numa efficiency
    - fp16

accelerate launch mimo/inference/main.py -i input_video.mp4
```

### Training
```bash
accelerate config
   - multi-GPU
   - numa efficiency
   - fp16

accelerate launch mimo/training/main.py -c 1540
```

## 🙏🏻 Acknowledgements
This project is based on [novitalabs/AnimateAnyone](https://github.com/novitalabs/AnimateAnyone) and [MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) which is licensed under the Apache License 2.0. We thank to the authors of [MIMO](https://menyifang.github.io/projects/MIMO/index.html), [novitalabs/AnimateAnyone](https://github.com/novitalabs/AnimateAnyone) and [MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), for their open research and exploration.