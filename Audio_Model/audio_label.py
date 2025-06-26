import argparse
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import sigmoid
import re
from io import BytesIO
import json
import pandas as pd

MMIO_TO_MODEL_SUBLABEL = {
    "teacherExplainsThinking": ("CogDem", "Analysis_Give"),
    "askingForThinking": ("CogDem", "Analysis_Request"),
    "teacherGivesInformation": ("CogDem", "Report_Give"),
    "askingForRecall": ("CogDem", "Report_Request"),
    "openEnded": ("Questions", "Open"),
    "closedEnded": ("Questions", "Closed"),
    "taskRelatedPrompt": ("Questions", "Prompt"),
    "studentGivesExplanation": ("ExJust", "Student_Give"),
    "studentAsksForExplanation": ("ExJust", "Student_Request"),
    "teacherGivesExplanation": ("ExJust", "Teacher_Give"),
    "teacherAsksForExplanation": ("ExJust", "Teacher_Request"),
    "positiveFeedback": ("Feedback1", "Affirming"),
    "correctiveFeedback": ("Feedback1", "Disconfirming"),
    "neutralFeedback": ("Feedback1", "Neutral"),
    "elaboratedFeedback": ("Feedback2", "Elaborated"),
    "unelaboratedFeedback": ("Feedback2", "Unelaborated"),
    "buildsOnStudentIdea": ("Uptake", "Building"),
    "pushesStudentThinkingFurther": ("Uptake", "Exploring"),
    "repeatsStudentIdea": ("Uptake", "Restating")
}

def ms_to_timestamp(ms):
    seconds = ms // 1000
    return f"{seconds//60:02d}:{seconds%60:02d}"

def convert_json_to_virtual_xlsx(transcript_path, output_path=None):
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)

    rows = []
    id_to_index = {}
    for i, entry in enumerate(transcript_data):
        who = "Teacher" if entry["isPrimary"] else f"Speaker {entry['speaker']}"
        timestamp = ms_to_timestamp(entry["startMilliseconds"])
        transcript_str = f"{who} ({timestamp}): {entry['text']}"
        row = {
            "Transcript": transcript_str,
            "CogDem": "", "Questions": "", "ExJust": "",
            "Feedback1": "", "Feedback2": "", "Uptake": ""
        }
        rows.append(row)
        id_to_index[entry["id"]] = i

    
    if output_path:
        with open(output_path, 'r', encoding='utf-8') as f:
            mmio_data = json.load(f)

        for field, entries in mmio_data.items():
            for item in entries:
                ids = item.get("ids", [])
                labels = item.get("labels", [])
                for label in labels:
                    if label not in MMIO_TO_MODEL_SUBLABEL:
                        continue
                    target_col, model_label = MMIO_TO_MODEL_SUBLABEL[label]
                    for sid in ids:
                        if sid in id_to_index:
                            idx = id_to_index[sid]
                            current = rows[idx][target_col]
                            if model_label not in current:
                                rows[idx][target_col] = (current + "; " + model_label).strip("; ")

    df = pd.DataFrame(rows)
    virtual_excel = BytesIO()
    df.to_excel(virtual_excel, index=False)
    virtual_excel.seek(0)
    return virtual_excel



MAIN_LABELS = ["CogDem", "Questions", "ExJust", "Feedback1", "Feedback2", "Uptake"]
ALL_LABELS = MAIN_LABELS + ["None"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_MODEL_PATH = os.path.join(SCRIPT_DIR, "model/audio_class_model")

main_tokenizer = BertTokenizer.from_pretrained(MAIN_MODEL_PATH, local_files_only=True)
main_model = BertForSequenceClassification.from_pretrained(MAIN_MODEL_PATH, local_files_only=True)
main_model.eval()
main_model.to("cuda" if torch.cuda.is_available() else "cpu")

SUBMODEL_DIR = {
    label: os.path.join(SCRIPT_DIR, f"model/{label}_Ncon_clear_model") for label in MAIN_LABELS
}
submodels = {}
sub_tokenizers = {}
sub_idx2label = {}
for label in MAIN_LABELS:
    submodels[label] = BertForSequenceClassification.from_pretrained(SUBMODEL_DIR[label], local_files_only=True)
    submodels[label].eval()
    submodels[label].to("cuda" if torch.cuda.is_available() else "cpu")
    sub_tokenizers[label] = BertTokenizer.from_pretrained(SUBMODEL_DIR[label], local_files_only=True)

    label_map_path = os.path.join(SUBMODEL_DIR[label], "label_map.txt")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            idx2label = {}
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    idx, label_name = int(parts[0]), parts[1]
                    idx2label[idx] = label_name
                else:
                    idx = len(idx2label)
                    idx2label[idx] = line.strip()
        sub_idx2label[label] = idx2label
    else:
        sub_idx2label[label] = {0: "Label_0", 1: "Label_1"}

def predict_main_labels(text, threshold=0.5):
    inputs = main_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(main_model.device)
    logits = main_model(**inputs).logits
    probs = sigmoid(logits).squeeze().detach().cpu().numpy()
    preds = [int(p > threshold) for p in probs]
    return {label: preds[i] for i, label in enumerate(ALL_LABELS)}

def predict_sub_label(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(model.device)
    logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    return pred_id

def extract_timestamp(text):
    match = re.search(r"\((\d{2}:\d{2})\)", text)
    return match.group(1) if match else ""

def process_transcripts(input_path):
    df = pd.read_excel(input_path)
    valid_indices = []
    timestamps = []
    for i, row in df.iterrows():
        text = str(row["Transcript"]).strip()
        timestamp = extract_timestamp(text)
        if timestamp:
            valid_indices.append(i)
            timestamps.append(timestamp)

    column_headers = ["AudioLabel"] + timestamps
    binary_results = {}
    for label in MAIN_LABELS:
        for sublabel in sub_idx2label[label].values():
            binary_results[f"{label} {sublabel}".strip()] = []

    no_label_row = []

    for i in valid_indices:
        text = str(df.loc[i, "Transcript"]).strip()
        multi_label_result = predict_main_labels(text)

        for label in MAIN_LABELS:
            for sublabel in sub_idx2label[label].values():
                binary_results[f"{label} {sublabel}".strip()].append(0)

        has_any_label = False
        for label in MAIN_LABELS:
            if multi_label_result[label] == 1:
                has_any_label = True
                submodel = submodels[label]
                subtokenizer = sub_tokenizers[label]
                pred_id = predict_sub_label(text, submodel, subtokenizer)
                label_name = sub_idx2label[label].get(pred_id, f"Label_{pred_id}")
                binary_results[f"{label} {label_name}".strip()][-1] = 1

        no_label_row.append(1 if not has_any_label else 0)

    output_rows = []
    for k, v in binary_results.items():
        output_rows.append([k] + v)
    output_rows.append(["NoLabel"] + no_label_row)

    result_df = pd.DataFrame(output_rows, columns=column_headers)
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_transcript", type=str, help="Path to input .XLSX file")
    group.add_argument("--transcript_json", type=str, help="Path to transcript.json")

    parser.add_argument("--output_xlsx", type=str, default="predictions.xlsx", help="Path to output XLSX file")

    args = parser.parse_args()

    if args.transcript_json:
        virtual_excel = convert_json_to_virtual_xlsx(args.transcript_json)  # 🚫 no output_json now
        df_result = process_transcripts(virtual_excel)
    else:
        df_result = process_transcripts(args.video_transcript)

    df_result.to_excel(args.output_xlsx, index=False)
    print(f" Predictions saved to {args.output_xlsx}")


