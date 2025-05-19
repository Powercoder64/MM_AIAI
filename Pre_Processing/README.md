# I3D-Based Video Feature Pre-Processing 🚀
Convert raw **`.mp4` videos → NumPy feature tensors** ready for multimodal / action‑recognition pipelines.  
This module lives in **`MM_AIAI/Pre_Processing/`** and extracts **RGB** and **optical‑flow** embeddings with an *Inflated‑3‑D (I3D)* backbone, using **PWC‑Net** for dense flow.

---

## 📑 Table of Contents
1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Outputs](#outputs)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Dependencies](#dependencies)
7. [Pre‑Trained Weights](#pre-trained-weights)

---

## ✨ Features
- **One‑liner inference:** `python Pre_Processing.py filename=<video.mp4>`
- **Dual‑stream I3D** (RGB + Flow) with PWC‑Net flow estimation  
- Customisable **input / output folders**, FPS, clip length, device IDs  
- Produces four artefacts; only `*_rgb.npy` and `*_flow.npy` are mandatory for the subsequent multimodal model

---

## 🚀 Quick Start

### 1 · Clone the parent project
```bash
git clone https://github.com/Powercoder64/MM_AIAI.git
cd MM_AIAI
```

### 2 · Build the Docker image *(from the sub‑folder)*
```bash
docker build -t pre-process \
             -f Pre_Processing/Dockerfile \
             Pre_Processing
```

### 3 · Run the extractor
```bash
# Prepare folders & copy a video
mkdir -p Pre_Processing/video Pre_Processing/output
cp /path/to/my_clip.mp4 Pre_Processing/video/

docker run --gpus all \
           -v $(pwd)/Pre_Processing/video:/app/video \
           -v $(pwd)/Pre_Processing/output:/app/output \
           pre-process \
           python Pre_Processing.py filename=my_clip.mp4
```

### 4 · Run locally (conda / pip)
```bash
cd Pre_Processing
conda env create -f conda_env_torch_zoo.yml
conda activate torch_zoo

python Pre_Processing.py filename=my_clip.mp4
```

---

## 📦 Outputs
| File (for `my_clip`)        | Tensor shape | Description                                |
|-----------------------------|--------------|--------------------------------------------|
| `my_clip_rgb.npy`           | `(T, 1024)`  | **_*feed to multimodal model*_**           |
| `my_clip_flow.npy`          | `(T, 1024)`  | **_*feed to multimodal model*_**           |
| `my_clip_rgb_frames.npy`    | `(N,H,W,3)`  | (optional) decoded RGB frames              |
| `my_clip_flow_frames.npy`   | `(N,H,W,2)`  | (optional) raw optical flow (PWC‑Net)      |

Only the first two files are required by the next multimodal stage.

---

## 🗂 Project Structure
```text
MM_AIAI/
└── Pre_Processing/
    ├── Pre_Processing.py               # entry point
    │
    ├── Dockerfile                      # Docker build file
    │
    ├── utils/
    │   └── utils.py                    # path_list = './video/' + args.filename
    │
    ├── models/
    │   ├── i3d/
    │   │   ├── extract_i3d.py          # self.output_path = './output/'
    │   │   └── checkpoints/
    │   │       ├── i3d_rgb.pt
    │   │       └── i3d_flow.pt
    │   └── pwc/
    │       ├── extract_pwc.py
    │       └── checkpoints/
    │           └── pwc_net_sintel.pt
    │
    └── configs/                        # frame‑rate & clip‑length YAMLs
```
> **Keep the checkpoint folders exactly as shown** so the loader can find the `.pt` files.

---

## ⚙️ Configuration
| Parameter         | Default     | How to change                                                                                   |
|-------------------|-------------|-------------------------------------------------------------------------------------------------|
| Video folder      | `./video/`  | Edit **`utils/utils.py`** → `path_list = './video/' + args.filename`                            |
| Output folder     | `./output/` | Edit **`models/i3d/extract_i3d.py`** → `self.output_path = './output/'`                         |
| FPS / clip length | see YAMLs   | Tweak files in **`configs/`**                                                                   |

---

## 📚 Dependencies
*(Full list in the Dockerfile & Conda YAML)*
| Package                        | Purpose                                |
|--------------------------------|----------------------------------------|
| **PyTorch ≥ 1.7 + CUDA**       | Deep learning & GPU compute            |
| **torchvision**                | I3D layers & video transforms          |
| **opencv-python**              | Video decoding                         |
| **ffmpeg / av**                | Extra container formats                |
| **numpy**, **pandas**          | Tensor & log handling                  |
| **scikit-learn**               | Evaluation utilities                   |

---

## 📥 Pre‑Trained Weights
```bash
unzip pp_models.zip -d tmp_pp_models

# move the files into place
mv tmp_pp_models/i3d_rgb.pt   Pre_Processing/models/i3d/checkpoints/
mv tmp_pp_models/i3d_flow.pt  Pre_Processing/models/i3d/checkpoints/
mv tmp_pp_models/pwc_net_sintel.pt Pre_Processing/models/pwc/checkpoints/

# clean up
rm -r tmp_pp_models
```
Resulting tree:
```text
models/
├── i3d/checkpoints/i3d_rgb.pt
├── i3d/checkpoints/i3d_flow.pt
└── pwc/checkpoints/pwc_net_sintel.pt
```
