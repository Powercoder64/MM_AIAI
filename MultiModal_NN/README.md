# Multimodal_NN — Video + Transcript Fusion 🧩
Classroom‑activity detection from **pre‑extracted I3D features** *and* raw discourse transcripts.

> This module lives in **`MM_AIAI/Multimodal_NN/`**  

---

## 📑 Table of Contents
1. [Features](#features)  
2. [Quick Start](#quick-start)  
3. [Inputs](#inputs)  
4. [Outputs](#outputs)  
5. [Project Structure](#project-structure)  
6. [Configuration](#configuration)  
7. [Dependencies](#dependencies)  
8. [Pre-Trained Weights](#pre-trained-weights)

---

<a id="features"></a>
## ✨ Features
* **One‑liner inference**

  ```bash
  python Neural_Network.py --filename=my_clip.mp4
  ```

* **Multimodal fusion** of  
  • RGB embeddings + Flow embeddings + 20‑class transcript one‑hots  
* **CSV matrix output** with **24 activity classes** (rows) × **seconds** (columns)  
* Docker container & Conda recipe included  
* All data stays inside **`./data/`** (override via `options.py`)

---

<a id="quick-start"></a>
## 🚀 Quick Start

### 1 · Clone the parent project
```bash
git clone https://github.com/Powercoder64/MM_AIAI.git
cd MM_AIAI/Multimodal_NN
```

### 2 · Build the Docker image
```bash
docker build -t multimodal-nn -f Dockerfile .
```

### 3 · Run the model inside Docker
```bash
# 1) drop your files in the right sub‑folders (see below)
# 2) launch the container
docker run --gpus all                                              -v $(pwd)/data:/app/data                                 -v $(pwd)/models:/app/models                             multimodal-nn                                            python Neural_Network.py --filename=my_clip.mp4
```

### 4 · Local run (conda / pip)
```bash
conda env create -f conda_env_mm_nn.yml
conda activate mm_nn

python Neural_Network.py --filename=my_clip.mp4
```

---

<a id="inputs"></a>
## 📥 Inputs

```text
data/
├── video/              # raw classroom clips               (*.mp4)
│   └── my_clip.mp4
├── features/           # I3D outputs from Pre_Processing
│   ├── my_clip_rgb.npy
│   └── my_clip_flow.npy
└── transcripts/        # annotated discourse spreadsheets
    └── my_clip.xlsx    # sheet “Coding Labels”
```
*The default root (`./data`) can be changed in **`options.py`**.*

---

<a id="outputs"></a>
## 📦 Outputs

After a run you will find:

| File (for `my_clip`) | Shape / Type | Description |
|----------------------|--------------|-------------|
| `my_clip_MATRIX.csv` | 24 × *S*     | Binary **activity matrix**<br>rows = classes, cols = seconds |

<details>
<summary>CSV preview (first 5 s, 4 classes)</summary>

```text
,0001,0002,0003,0004,0005
Whole_Class_Activity,0,1,1,1,0
Individual_Activity ,0,0,1,0,0
Small_Group_Activity,0,0,0,0,0
Book-Using_or_Holding,0,0,1,1,1
...
```
*0 = class absent, 1 = class present (per second).*
</details>

---

<a id="project-structure"></a>
## 🗂 Project Structure

```text
Multimodal_NN/
├── Neural_Network.py          # entry point (arg --filename)
├── options.py                 # paths & hyper‑params
├── models/
│   ├── BaSNet_model_best.pkl  # ⇦ download & place here
│   └── MM_Transformer.pth     # ⇦ provided in repo
├── data/                      # default data root
│   ├── video/
│   ├── features/
│   └── transcripts/
├── utils/                     # data loaders, metrics, logging
├── Dockerfile
└── conda_env_mm_nn.yml
```

---

<a id="configuration"></a>
## ⚙️ Configuration

| Parameter        | Default      | Where to change                                     |
|------------------|--------------|-----------------------------------------------------|
| Data root        | `./data/`    | `options.py → DATA_PATH`                            |
| Output directory | `./outputs/` | `options.py → OUT_DIR`                              |
| Device           | CUDA if available | `--device cpu|cuda`                          |
| Checkpoint path  | `models/MM_Transformer.pth` | fusion network (default)            |

---

<a id="dependencies"></a>
## 📚 Dependencies
*(full list in Dockerfile & Conda YAML)*

| Package                 | Purpose                                  |
|-------------------------|------------------------------------------|
| **PyTorch ≥ 2.2 + CUDA**| Backbone & fusion network                |
| **numpy**, **pandas**   | Feature I/O, CSV/XLSX handling           |
| **openpyxl**            | Read annotated Excel transcripts         |
| **scikit‑learn**        | F1 / metric utilities                    |
| **opencv‑python**       | Video post-processing     |

---

<a id="pre-trained-weights"></a>
## 📥 Pre-Trained Weights

| File | Purpose |
|------|---------|
| **MM_Transformer.pth** | Multimodal fusion checkpoint. **Already included** in `models/`; no download needed. |
| **BaSNet_model_best.pkl** | Video baseline model. Download and place in `models/`; can also be tested for video baseline. |

Download BaSNet:

```bash
wget -O BaSNet_model_best.pkl   'https://drive.google.com/uc?export=download&id=1d0qPeMQSjOrllvrjKdMqkhC5gf0hEREt'

mkdir -p models
mv BaSNet_model_best.pkl models/
```

Resulting tree:
```text
models/
├── BaSNet_model_best.pkl
└── MM_Transformer.pth
```

---

### ▶️ Full example
```bash
python Neural_Network.py --filename=my_clip.mp4
# →  outputs/my_clip_MATRIX.csv
```

Enjoy second‑wise classroom‑activity matrices for your downstream analyses! 🎉
