import argparse
import json
import math
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────── VIDEO ROW → FIELD/LABEL MAP ──────────────────── #
ROW_NAME_MAP = {
    # Instructional format
    "Whole_Class_Activity":               ("instructionalFormat", "wholeClass"),
    "Individual_Activity":                ("instructionalFormat", "individual"),
    "Small_Group_Activity":               ("instructionalFormat", "smallGroup"),
    "Transition":                         ("instructionalFormat", "transition"),

    # Proximity to students
    "Teacher_Supporting_Multiple_with_SS_Interaction":
                                           ("proximityToStudents", "multiple"),
    "Teacher_Supporting_Multiple_without_SS_Interaction":
                                           ("proximityToStudents", "multiple"),
    "Teacher_Supporting_One_Student":     ("proximityToStudents",
                                           "individualStudent"),

    # Teacher position
    "Teacher_Sitting":                    ("teacherPosition", "sitting"),
    "Teacher_Standing_(T)":               ("teacherPosition", "standing"),
    "Teacher_Walking":                    ("teacherPosition", "walking"),

    # Student position
    "Student(s)_Desks-Sitting":           ("studentPosition", "deskSitting"),
    "Student(s)_Group_Tables-Sitting":    ("studentPosition",
                                           "groupTableSitting"),
    "Student(s)_Standing_or_Walking":     ("studentPosition",
                                           "standingOrWalking"),
    "Student(s)_Carpet_or_Floor-Sitting": ("studentPosition",
                                           "carpetOrFloorSitting"),

    # Student participation
    "Raising_Hand":                       ("studentParticipation", "handRaising"),
    "On_Task_Student_Talking_with_Student":
                                           ("studentParticipation",
                                            "studentToStudent"),

    # Tools & representations
    "Student_Writing":                    ("toolsAndRepresentationsInUse",
                                           "studentWriting"),
    "Teacher_Writing":                    ("toolsAndRepresentationsInUse",
                                           "teacherWriting"),
    "Book-Using_or_Holding":              ("toolsAndRepresentationsInUse",
                                           "paperBasedMaterials"),
    "Instructional_Tool-Using_or_Holding":
                                           ("toolsAndRepresentationsInUse",
                                           "paperBasedMaterials"),
    "Worksheet-Using_or_Holding":         ("toolsAndRepresentationsInUse",
                                           "paperBasedMaterials"),
    "Notebook-Using_or_Holding":          ("toolsAndRepresentationsInUse",
                                           "paperBasedMaterials"),
    "Individual_Technology":              ("toolsAndRepresentationsInUse",
                                           "studentTechnologyUse"),
    "Presentation_with_Technology":       ("toolsAndRepresentationsInUse",
                                           "teacherTechnologyUse"),
}

VIDEO_MULTI_LABEL = {
    "studentPosition",
    "studentParticipation",
    "toolsAndRepresentationsInUse",
}

# ─────────────── AUDIO LABEL COLS → SPEC FIELD/LABEL MAP ──────────────── #
COLUMN_MAP = {
    "sheet":         0,        # XLSX sheet index
    "id":            "id",     # utterance ID
    "start_seconds": "start",  # start time (seconds) from XLSX
}

AUDIO_LABEL_COLUMNS = {
    # Cognitive demand
    "askingForRecall":          ("levelOfThinkingPromoted", "askingForRecall"),
    "askingForThinking":        ("levelOfThinkingPromoted", "askingForThinking"),
    "teacherExplainsThinking":  ("levelOfThinkingPromoted", "teacherExplainsThinking"),
    "teacherGivesInformation":  ("levelOfThinkingPromoted", "teacherGivesInformation"),

    # Question types
    "openEnded":        ("typesOfQuestionsAsked", "openEnded"),
    "closedEnded":      ("typesOfQuestionsAsked", "closedEnded"),
    "taskRelatedPrompt":("typesOfQuestionsAsked", "taskRelatedPrompt"),

    # Explanation / justification
    "studentGivesExplanation":  ("explainingAndJustifyingIdeas", "studentGivesExplanation"),
    "teacherGivesExplanation":  ("explainingAndJustifyingIdeas", "teacherGivesExplanation"),
    "studentAsksForExplanation":("explainingAndJustifyingIdeas", "studentAsksForExplanation"),
    "teacherAsksForExplanation":("explainingAndJustifyingIdeas", "teacherAsksForExplanation"),

    # Feedback
    "neutralFeedback":     ("feedbackToStudents", "neutralFeedback"),
    "positiveFeedback":    ("feedbackToStudents", "positiveFeedback"),
    "correctiveFeedback":  ("feedbackToStudents", "correctiveFeedback"),
    "elaboratedFeedback":  ("feedbackToStudents", "elaboratedFeedback"),
    "unelaboratedFeedback":("feedbackToStudents", "unelaboratedFeedback"),

    # Embracing student ideas
    "repeatsStudentIdea":        ("embracingStudentIdeas", "repeatsStudentIdea"),
    "buildsOnStudentIdea":       ("embracingStudentIdeas", "buildsOnStudentIdea"),
    "pushesStudentThinkingFurther":
                                 ("embracingStudentIdeas", "pushesStudentThinkingFurther"),

    # Talk-time
    "studentTalk": ("talkTime", "studentTalk"),
    "teacherTalk": ("talkTime", "teacherTalk"),

    # Academic language
    "mathVocabulary": ("academicLanguage", "mathVocabulary"),
}

AUDIO_MULTI_LABEL = {field for (_, (field, _)) in AUDIO_LABEL_COLUMNS.items()}


# ───────────────────────────── UTILITIES ──────────────────────────────── #
def _runs(indices, payload):
    """Yield (start_idx, end_idx, value) for blocks of identical value."""
    for val, grp in groupby(zip(indices, payload), key=lambda x: x[1]):
        grp = list(grp)
        if not val:                       # skip blanks / empty tuples
            continue
        yield grp[0][0], grp[-1][0], val


# ───────────────────────── VIDEO CONVERSION ──────────────────────────── #
def video_segments(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path).set_index("Unnamed: 0")
    df.index = df.index.str.strip()
    second_cols = [c for c in df.columns if c.isdigit()]
    if not second_cols:
        raise ValueError("CSV has no numeric second columns like '0001'.")
    sec_idx = np.array([int(c) - 1 for c in second_cols])   # '0001' → 0 s
    mat = df[second_cols].T
    mat.index = sec_idx
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    n_sec = len(mat.index)

    # Allocate per-second containers
    per_sec = {}
    for _, (field, _) in ROW_NAME_MAP.items():
        per_sec.setdefault(field, [])
    for field in per_sec:
        if field in VIDEO_MULTI_LABEL:
            per_sec[field] = [[] for _ in range(n_sec)]
        else:
            per_sec[field] = [None for _ in range(n_sec)]

    # Populate
    for row_name, (field, label) in ROW_NAME_MAP.items():
        if row_name not in mat.columns:
            continue
        active = mat[row_name].values.astype(bool)
        for i, act in enumerate(active):
            if not act:
                continue
            if field in VIDEO_MULTI_LABEL:
                per_sec[field][i].append(label)
            else:
                per_sec[field][i] = label

    # Build segments
    segments = defaultdict(list)
    for field, values in per_sec.items():
        ser = [tuple(sorted(v)) if isinstance(v, list) else v for v in values]
        for s, e, v in _runs(mat.index, ser):
            entry = {"start": int(s), "end": int(e)}
            entry["labels"] = list(v) if isinstance(v, tuple) else [v]
            segments[field].append(entry)
    return segments


# ───────────────────────── AUDIO CONVERSION ──────────────────────────── #
def audio_segments(xlsx_path: Path, transcript_path: Path) -> dict:
    # Transcript stats
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    id2words = {u["id"]: len(u["text"].split()) for u in transcript}
    total_words = sum(id2words.values())
    total_secs = sum(
        max(1, (u["endMilliseconds"] - u["startMilliseconds"]) / 1000)
        for u in transcript
    )
    wps = total_words / total_secs if total_secs else 2.0

    # XLSX
    df = pd.read_excel(xlsx_path, sheet_name=COLUMN_MAP["sheet"])
    df = df.dropna(subset=[COLUMN_MAP["id"], COLUMN_MAP["start_seconds"]]).reset_index(drop=True)
    n = len(df)

    lbls = {f: [set() if f in AUDIO_MULTI_LABEL else None for _ in range(n)]
            for f in AUDIO_MULTI_LABEL.union({f for (_, (f, _)) in AUDIO_LABEL_COLUMNS.items()})}
    ids = {f: [set() for _ in range(n)] for f in lbls}

    for i, row in df.iterrows():
        utt_id = int(row[COLUMN_MAP["id"]])
        for col, (field, label) in AUDIO_LABEL_COLUMNS.items():
            if pd.notna(row.get(col)) and int(row[col]) == 1:
                if field in AUDIO_MULTI_LABEL:
                    lbls[field][i].add(label)
                else:
                    lbls[field][i] = label
                ids[field][i].add(utt_id)

    starts = df[COLUMN_MAP["start_seconds"]].astype(int).tolist()
    ends = [
        int(starts[i] + math.ceil(id2words.get(int(df.loc[i, COLUMN_MAP['id']]), 0) / wps))
        for i in range(n)
    ]

    segs = defaultdict(list)
    for field in lbls:
        series = [tuple(sorted(v)) if isinstance(v, set) else v for v in lbls[field]]
        id_series = [tuple(sorted(v)) for v in ids[field]]
        for s, e, val in _runs(range(n), series):
            seg_ids = set()
            for k in range(s, e + 1):
                seg_ids.update(id_series[k])
            segs[field].append({
                "start": starts[s],
                "end":   ends[e],
                "labels": list(val) if isinstance(val, tuple) else [val],
                "ids":   sorted(seg_ids),
            })
    return segs


# ──────────────────────────────── MAIN ────────────────────────────────── #
def convert(base_name: str) -> None:
    out_dir = Path("./output")
    data_audio = Path("./data/audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    video_csv       = out_dir / f"{base_name}_MATRIX.csv"
    audio_xlsx      = data_audio / f"{base_name}.xlsx"
    transcript_json = data_audio / f"{base_name}.json"
    output_json     = out_dir / f"{base_name}_AUDIO_VIDEO.json"

    visual = video_segments(video_csv)
    audio  = audio_segments(audio_xlsx, transcript_json)

    merged = {**visual, **audio}
    output_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"✅  Combined MM-IO JSON saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-second video CSV + audio XLSX + transcript JSON "
                    "into a MM-IO JSON file."
    )
    parser.add_argument(
        "base_name",
        help="File base-name (no extension). "
             "Expecting files:\n"
             "  ./output/<base>.csv (video matrix)\n"
             "  ./data/audio/<base>.xlsx (audio labels)\n"
             "  ./data/audio/<base>.json (transcript)"
    )
    args = parser.parse_args()
    convert(args.base_name)


if __name__ == "__main__":
    main()
