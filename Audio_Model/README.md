# Audio Labeling Script

This script processes Excel-format transcript data (e.g., classroom audio transcripts) and generates labeled samples based on a specified field (e.g., `Feedback2`). The output can be used for training audio classification or sequence models.
---

## Environment Setup

Make sure you are using the following Python version and dependencies:

### Python Version
- Python `3.10.17`

### Python Packages (recommended to install in a virtual environment)
```bash
pip install torch==2.2.2+cu118 \
            transformers==4.51.3 \
            scikit-learn==1.6.1 \
            pandas==2.2.3 \
            numpy==1.24.4 \
            matplotlib==3.10.3
```
### HPC Cluster Module Setup
```bash
module purge
module load gcc/11.4.0
module load cuda/11.8.0
```
### Pretrained Models

Please ensure that the following pretrained models are placed under the `model/` directory **before running the code**:

```
model/
├── audio_class_model/
├── CogDem_Ncon_clear_model/
├── ExJust_Ncon_clear_model/
├── Feedback1_Ncon_clear_model/
├── Feedback2_Ncon_clear_model/
├── Questions_Ncon_clear_model/
└── Uptake_Ncon_clear_model/
```
link for the model: [Download Link](https://drive.google.com/file/d/11LSGXlFkFIhGZo-Oi59mTj57vVmcGB58/view?usp=sharing)
### How to Run

Option 1: Using .xlsx input: (example)

```bash
python audio_label.py \
  --video_transcript ./210.041_MATH2_20180320.xlsx \
  --output_xlsx ./output.xlsx
```

Option 2: Using .json input
```bash
python audio_label.py \
  --transcript_json ./transcript.json \
  --output_xlsx ./output.xlsx
```

# How to Run the Audio Labeling Docker Container

After building the Docker image (e.g., with `docker build -t audio_label_gpu .`), you can run the audio labeling script using the following command:

```bash
docker run --rm --gpus all \
  -v "$PWD":/home/ubuntu/audio_labeling \
  -v "$PWD/model":/home/ubuntu/audio_labeling/model \
  audio_label_gpu \
  --video_transcript /home/ubuntu/audio_labeling/input.xlsx \
  --output_xlsx /home/ubuntu/audio_labeling/output.xlsx
```

## Notes:
- `--gpus all` enables GPU support (requires NVIDIA driver and `nvidia-docker`).
- `-v "$PWD":/home/ubuntu/audio_labeling` mounts your current directory into the container.
- `-v "$PWD/model":/home/ubuntu/audio_labeling/model` ensures the pretrained models are accessible.
- Replace `input.xlsx` with your own Excel transcript file.
- The output Excel file will be written to the current directory as `output.xlsx`.
