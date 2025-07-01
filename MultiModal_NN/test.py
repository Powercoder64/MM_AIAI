import torch
import torch.nn as nn
import numpy as np
import utils
import os
import json
from tqdm import tqdm
from sklearn import metrics
from scipy.interpolate import interp1d
import cv2
import pandas as pd
import re
from pathlib import Path
import math
import warnings


warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning
)


class_dict = {0: 'Whole_Class_Activity',
              1: 'Individual_Activity',
              2: 'Small_Group_Activity',
              3: 'Book-Using_or_Holding',
              4: 'Instructional_Tool-Using_or_Holding',
              5: 'Student_Writing',
              6: 'Teacher_Writing',
              7: 'Raising_Hand',
              8: 'Presentation_with_Technology',
              9: 'Individual_Technology',
              10: 'Worksheet-Using_or_Holding',
              11: 'Notebook-Using_or_Holding',
              12: 'Student(s)_Carpet_or_Floor-Sitting',
              13: 'Student(s)_Desks-Sitting',
              14: 'Student(s)_Group_Tables-Sitting',
              15: 'Student(s)_Standing_or_Walking',
              16: 'Teacher_Sitting',
              17: 'Teacher_Standing_(T)',
              18: 'Teacher_Walking',
              19: 'Teacher_Supporting_One_Student',
              20: 'Teacher_Supporting_Multiple_with_SS_Interaction',
              21: 'Teacher_Supporting_Multiple_without_SS_Interaction',
              22: 'On_Task_Student_Talking_with_Student',
              23: 'Transition'}


def check_file_with_retries(filename, max_retries=5, delay_seconds=30):
    attempt = 0
    while attempt < max_retries:
        if os.path.exists(filename):
            print(f"File found: {filename}")
            return True
        else:
            print(f"File not found: {filename}. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(delay_seconds)
            attempt += 1
    
    raise FileNotFoundError(f"File not found after {max_retries} retries: {filename}")


def results_process(results, output_path, filename, data_path):
    confi_thresh = np.array([
    0.0002, 0.0003, 0.00016, 0.0004, 0.00015, 0.0002, 0.00042, 0.0005, 0.00018,
    0.00023, 0.00012, 0.0004, 0.00017, 0.00019, 0.0002, 0.00025, 0.00016, 0.00015,
    0.00042, 0.00035, 0.0002, 0.00038, 0.00036, 0.00054])

    results_all = []

    for i in range(0, 24):
    
        #print(results.shape)   
        results_i = results.reshape(results.shape[0] * results.shape[1], results.shape[2])[:, i]
        results_i[np.where(results_i < confi_thresh[i])] = 0
        results_i[np.where(results_i >= confi_thresh[i])] = 1
        results_i = results_i.reshape(results.shape[0], results.shape[1], 1)
        results_all.append(results_i)

    results_all = np.concatenate(results_all, axis=2).squeeze()
    #print(results_all.shape)
  
   
    def scale_binary_2d(video_sample, scale_factor):
        num_frames, num_classes = video_sample.shape
        x = np.arange(num_frames)
        scaled_frames = int(num_frames * scale_factor)
        xnew = np.linspace(0, num_frames - 1, num=scaled_frames)

        # Initialize the scaled video array
        scaled_video = np.zeros((scaled_frames, num_classes), dtype=int)

        # Apply the scaling function to each class (column) independently
        for i in range(num_classes):
            f = interp1d(x, video_sample[:, i], kind='nearest')
            scaled_video[:, i] = np.round(f(xnew)).astype(int)

        return scaled_video
    


    
    path_vid  = data_path + '/' + 'video/'
    vid_names = filename
    #print(vid_names)
    num_frames=[]
    lines =  vid_names

    cap = cv2.VideoCapture(path_vid + vid_names)
    print(path_vid + vid_names)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    num_frames = length
    
    sample_size = results_all.shape[0]

    output_dir = output_path
    scale_factor = num_frames / (sample_size * 30)
    scaled_results = scale_binary_2d(results_all, scale_factor)
    #print(scaled_results.shape)
    
    n_secs   = scaled_results.shape[0]    
                    # seconds in video
    col_names = [f"{s+1:04d}" for s in range(n_secs)] 
    #print(n_secs)
    
    df = pd.DataFrame(
    scaled_results.T,                                 # rows == activities
    index=[class_dict[i] for i in range(24)],        # activity labels
    columns=col_names
)


    temp_path   = './tmp/temp_scores.xlsx'
    df.to_excel(temp_path, sheet_name="scores")
    
############################################################################################



    RAW_FILE      = data_path + '/audio/' + filename[0:-4] + '.xlsx'        # raw “transcript + labels”
    OUTPUT_FILE   = './tmp/temp_tran.xlsx'
    #TEMPLATE_FILE = None            # keep row order from a template, or None to build fresh

    # ── 19 canonical rows (order matters) ──────────────────────────────────────
    ROWS = [
        "CogDem Analysis_Give",
        "CogDem Analysis_Request",
        "CogDem Report_Give",
        "CogDem Report_Request",
        "ExJust Student_Give",
        "ExJust Student_Request",
        "ExJust Teacher_Give",
        "ExJust Teacher_Request",
        "Feedback2 Elaborated",
        "Feedback2 Unelaborated",
        "Feedback1 Affirming",
        "Feedback1 Disconfirming",
        "Feedback1 Neutral",
        "Questions Closed",
        "Questions Open",
        "Uptake Building",
        "Uptake Exploring",
        "Uptake Restating",
        "NoLabel",
    ]

    TS_RX = re.compile(r"^(\d{1,2}):(\d{2})")  # matches MM:SS or MM:SS.xxx


    def ts_to_sec(ts: str) -> int:
        """'12:34' → 754 seconds."""
        m = TS_RX.match(str(ts))
        if not m:
            raise ValueError(f"Bad timestamp: {ts!r}")
        mm, ss = map(int, m.groups())
        return mm * 60 + ss


    def convert(in_path: str, out_path: str) -> None:
        df = pd.read_excel(in_path)

        # ── collect start times & active codes ─────────────────────────────────
        ts_cols = [c for c in df.columns if c != "AudioLabel"]
        if not ts_cols:
            raise RuntimeError("No timestamp columns found.")

        start_secs = sorted(ts_to_sec(c) for c in ts_cols)
        utter_map = {
            ts_to_sec(c): df.loc[df[c] == 1, "AudioLabel"].astype(str).tolist()
            for c in ts_cols
        }

        # ── build utterance spans (start … next-1) ─────────────────────────────
        utterances = []
        for i, s in enumerate(start_secs):
            nxt = start_secs[i + 1] if i + 1 < len(start_secs) else None
            end_sec = (nxt - 1) if nxt is not None else s
            utterances.append((s, end_sec, utter_map.get(s, [])))

        first_sec, last_sec = start_secs[0], utterances[-1][1]
        time_axis = list(range(first_sec, last_sec + 1))
        headers = [f"{t - first_sec:04d}" for t in time_axis]


        zero_block = pd.DataFrame(
            np.zeros((len(ROWS), len(headers)), dtype=np.int8),  # tiny & fast
            columns=headers,
        )
        grid = pd.concat([pd.DataFrame({"AudioLabel": ROWS}), zero_block], axis=1)
        idx = {lbl: i for i, lbl in enumerate(ROWS)}

        # ── populate grid ──────────────────────────────────────────────────────
        for start, end, codes in utterances:
            rel_start, rel_end = start - first_sec, end - first_sec

            # keep only recognised labels
            codes = [c for c in codes if c in idx]

            if not codes:
                grid.iloc[idx["NoLabel"], 1 + rel_start : 1 + rel_end + 1] = 1
                continue

            for code in codes:
                grid.iloc[idx[code], 1 + rel_start : 1 + rel_end + 1] = 1

        # ── sanity-check row order then save ───────────────────────────────────
        grid = grid.set_index("AudioLabel").reindex(ROWS).reset_index()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        grid.to_excel(out_path, index=False)
        print("Saved:", Path(out_path).resolve())

    convert(RAW_FILE, OUTPUT_FILE)

    #TARGET_FILE = './tmp/temp_tran.xlsx'      
    CKPT_PATH   =  './models/MM_Transformer.pth'           # saved checkpoint from training
    OUTPUT_DIR  =  "Enhanced_Labels"                       # where CSV will be written
    # ---------------------------------------------------------------- #

    #DIR_PRED = base_path / "Final_Labels_Video"
    #DIR_TRAN = base_path / "Final_Labels_Tran"
    #OUTPUT_DIR.mkdir(exist_ok=True)

    TRANSCRIPT_CLASSES = [
        "CogDem Analysis_Give", "CogDem Analysis_Request", "CogDem Report_Give",
        "CogDem Report_Request", "ExJust Student_Give", "ExJust Student_Request",
        "ExJust Teacher_Give", "ExJust Teacher_Request", "Feedback2 Elaborated",
        "Feedback2 Unelaborated", "Feedback1 Affirming", "Feedback1 Disconfirming",
        "Feedback1 Neutral", "NoLabel", "Questions Closed", "Questions Open",
        "Questions Prompt", "Uptake Building", "Uptake Exploring", "Uptake Restating"
    ]

    CONTEXT   = 5                     # ±5 s → 11-frame window
    SEQ_LEN   = CONTEXT * 2 + 1       # 11
    INPUT_DIM = 44                    # 24 + 20
    HIDDEN    = 128
    OUTPUT_DIM= 24
    DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
    THRESHOLD = 0.5

    # ---------------------------------------------------------------- #
    # 2.  Helpers
    # ---------------------------------------------------------------- #
    def open_xlsx_df(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path)
        df.set_index(df.columns[0], inplace=True)
        return df

    def build_full_transcript(df_partial: pd.DataFrame, template_cols):
        full = pd.DataFrame(0, index=TRANSCRIPT_CLASSES, columns=template_cols)
        if "0000" in df_partial.columns:
            df_partial = df_partial.drop(columns="0000")
        for cls in df_partial.index.intersection(full.index):
            full.loc[cls, df_partial.columns] = df_partial.loc[cls].replace(-1, 0)
        return full

    # ---------------------------------------------------------------- #
    # 3.  Load ONE pair of XLSX files
    # ---------------------------------------------------------------- #
    pred_df = open_xlsx_df('./tmp/temp_scores.xlsx')          # 24 × T scores
    tran_df = open_xlsx_df('./tmp/temp_tran.xlsx')          # partial transcript
    full_tran = build_full_transcript(tran_df, pred_df.columns)

    class_names = pred_df.index.tolist()
    n_frames    = pred_df.shape[1]

    # Build per-frame 44-D feature matrix
    if pred_df.values.T.shape[0] < full_tran.values.T.shape[0]:
        feat_frames = np.concatenate(
            [pred_df.values.T, full_tran.values.T[0: pred_df.values.T.shape[0]]], axis=1      # (T, 44)
        ).astype(np.float32)

    if pred_df.values.T.shape[0] > full_tran.values.T.shape[0]:
        
                feat_frames = np.concatenate(
            [pred_df.values.T[0: full_tran.values.T.shape[0]], full_tran.values.T], axis=1      # (T, 44)
        ).astype(np.float32)
    # ---------------------------------------------------------------- #
    # 4.  Build every ±5-s sequence (centre = frames 5 … T-6)
    # ---------------------------------------------------------------- #
    seq_list   = []
    centres    = []                 # remember which frame each seq targets
    for centre in range(CONTEXT, n_frames - CONTEXT):
        left, right = centre - CONTEXT, centre + CONTEXT + 1
        seq_list.append(feat_frames[left:right])
        centres.append(centre)

    X_seq = torch.tensor(seq_list, dtype=torch.float32, device=DEVICE)  # (N_seq, 11, 44)

    # ---------------------------------------------------------------- #
    # 5.  Define model and load checkpoint
    # ---------------------------------------------------------------- #
    class CentreFrameTransformer(nn.Module):
        def __init__(self,
                    in_dim: int = INPUT_DIM,
                    d_model: int = HIDDEN,
                    nhead: int = 4,
                    num_layers: int = 2,
                    ff_mult: int = 4,
                    out_dim: int = OUTPUT_DIM,
                    dropout: float = 0.1):
            super().__init__()
            self.proj = nn.Linear(in_dim, d_model)
            self.pos  = nn.Parameter(torch.randn(SEQ_LEN, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * ff_mult,
                dropout=dropout,
                batch_first=True)
            self.enc  = nn.TransformerEncoder(enc_layer, num_layers)
            self.head = nn.Linear(d_model, out_dim)

        def forward(self, x):                # x: (B, 11, 44)
            x = self.proj(x) + self.pos
            x = self.enc(x)
            centre_vec = x[:, SEQ_LEN // 2, :]
            return self.head(centre_vec)     # (B, 24 logits)

    model = CentreFrameTransformer().to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---------------------------------------------------------------- #
    # 6.  Inference
    # ---------------------------------------------------------------- #
    with torch.no_grad():
        logits = model(X_seq).cpu()
    probs = torch.sigmoid(logits)           # (N_seq, 24)
    preds_bin = (probs >= THRESHOLD).int()

    # ---------------------------------------------------------------- #
    # 7.  Assemble full-length enhanced label matrix
    #     – centre frames get model output
    #     – first/last 5 frames fall back to thresholded video scores
    # ---------------------------------------------------------------- #
    enhanced = torch.zeros((OUTPUT_DIM, n_frames), dtype=torch.int)   # (24, T)

    # … the three assignment blocks remain identical, because they all
    #     work on tensors now:
    # model-refined frames
    for seq_i, frame_i in enumerate(centres):
        enhanced[:, frame_i] = preds_bin[seq_i]

    # boundary frames → baseline threshold on raw video scores
    # boundary frames → baseline threshold on raw video predictions
    baseline = torch.tensor(
        (pred_df >= THRESHOLD).astype(int).values.tolist(),  # plain list-of-lists
        dtype=torch.int
    )                           # (24, T) – no NumPy handed to PyTorch

    enhanced[:, :CONTEXT]  = baseline[:, :CONTEXT]
    enhanced[:, -CONTEXT:] = baseline[:, -CONTEXT:]

    # --- write CSV (convert to list-of-lists → DataFrame) -------------
    out_df = pd.DataFrame(enhanced.tolist(),      # no NumPy needed
                        index=class_names,
                        columns=pred_df.columns)
    csv_path = output_path + '/' + filename[0:-4] + '_MATRIX' + '.csv'
    out_df.to_csv(csv_path)
    print(f"Enhanced  Multimodal labels written to: {csv_path}")

def test(net, config, logger, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()

        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}


        load_iter = iter(test_loader)
        score_np_list = []
        label_np_list = []
        print('the feature file is being processed: ' + str(config.filename))
        for i in range(len(test_loader.dataset)):
            _data, vid_name, vid_num_seg = next(load_iter)

            

            _data = _data.cuda()
            #print(_data.shape)
            #_label = _label.cuda()

            _, cas_base, score_supp, cas_supp, fore_weights = net(_data)

            #label_np = _label.cpu().numpy()
            score_np = score_supp.cpu().data.numpy()


            score_np = score_np[:, :, 0:-1].squeeze()
            #label_np = label_np.squeeze()

            score_np_list.append(score_np)
    print('the NN model processed the feature files')
            #label_np_list.append(label_np)
    results_process(np.array(score_np_list), config.output_path, config.filename, config.data_path)
    print('the output matrix file is generated')
