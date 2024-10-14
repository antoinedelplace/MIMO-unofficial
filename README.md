# MIMO-unofficial
Unofficial implementation of MIMO (MImicking anyone anywhere with complex Motions and Object interactions)

My blog post: [![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@delplaceantoine/mastering-mimo-mimicking-anyone-anywhere-with-complex-motions-and-object-interactions-e8598d9d97d6)
Original paper: [![arXiv](https://img.shields.io/badge/arXiv-2409.16160-00ff00.svg)](https://arxiv.org/abs/2409.16160)

## Installation and Setup
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
```bash
├── code
│   ├── MIMO-unofficial
│   ├── Depth-Anything-V2
│   ├── 4D-Humans
│   ├── PHALP
│   ├── detectron2
│   ├── sam2
│   └── nvdiffrast
├── checkpoints
└── data
```

### Clone external repositories and Install requirements
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)
- [PHALP](https://github.com/brjathu/PHALP)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [sam2](https://github.com/facebookresearch/sam2)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)

### Download checkpoints
- [depth_anything_v2_vitl.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

### Extra steps
- For `dataset_preprocessing/pose_estimation_4DH.py`
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

## Note
- The 45 2D joints are composed of 25+20 joints corresponding to 25 openpose joints and 20 other joints (part of Opentrack?)