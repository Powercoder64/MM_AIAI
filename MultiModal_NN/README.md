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
* **inference**

  ```bash
  python Neural_Network.py --filename=my_clip.mp4  #runs the multimodal model and produces the audio-enhanced video labels
  ```
**For Debugging purposes, several output JSON files can be created:**

 ```bash
  python convert_JSON_video.py file_name.csv  #converts the output video csv file to video JSON file
 ```


* **Multimodal fusion** of  
  • RGB embeddings + Flow embeddings + 20‑class transcript one‑hots  
* **JSON matrix output** with **24 activity classes** + **audio/transcript labels**  
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
    └── my_clip.xlsx    # audio labels obtained by the Audio Model”
```
*The default root (`./data`) can be changed in **`options.py`**.*

---

<a id="outputs"></a>

---

<a id="project-structure"></a>
## 🗂 Project Structure

```text
Multimodal_NN/
├── Neural_Network.py          # entry point (arg --filename)
├── Merge_JASON.py             # merge audio and video labels as a single JSON
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


