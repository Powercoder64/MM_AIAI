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

- `audio_class_model`
- `CogDem_Ncon_clear_model`
- `ExJust_Ncon_clear_model`
- `Feedback1_Ncon_clear_model`
- `Feedback2_Ncon_clear_model`
- `Questions_Ncon_clear_model`
- `Uptake_Ncon_clear_model`

### How to Run
Run the script with the following command: (example)
```bash
python audio_label.py \
  --video_transcript /home/ekn8kz/Audio/210.041_MATH2_20180320.xlsx \
  --output_xlsx /home/ekn8kz/Audio/output.xlsx
```