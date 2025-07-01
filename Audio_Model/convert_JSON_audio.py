import argparse
import json
import math
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import pandas as pd


# ──────────────────────────────── CONFIG ─────────────────────────────── #
COLUMN_MAP = {
    "sheet":          0,          # change if labels live on another sheet
    "id":             "id",       # utterance ID (must match transcript JSON)
    "start_seconds":  "start",    # start time in seconds from the XLSX
}

# label-column  →  (spec field, spec label)
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

MULTI_LABEL_FIELDS = {field for (_, (field, _)) in AUDIO_LABEL_COLUMNS.items()}


# ─────────────────────────────── HELPERS ─────────────────────────────── #
def contiguous(idx_iterable, value_iterable):
    """Yield (start_idx, end_idx, value) for runs of identical `value`."""
    for val, grp in groupby(zip(idx_iterable, value_iterable), key=lambda x: x[1]):
        grp = list(grp)
        if not val:        # skip blanks / empty tuples
            continue
        yield grp[0][0], grp[-1][0], val


# ──────────────────────────── MAIN PIPELINE ──────────────────────────── #
def convert(base_name: str) -> None:
    # paths
    data_dir = Path("./data/audio")
    output_dir = Path("./output")
    xlsx_path  = data_dir / f"{base_name}.xlsx"
    tr_json    = data_dir / f"{base_name}.json"
    out_json   = output_dir / f"{base_name}_AUDIO.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) -------- transcripts → id→(text, words, duration) ---------------- #
    with tr_json.open(encoding="utf-8") as fh:
        transcript = json.load(fh)

    id2info, total_words, total_secs = {}, 0.0, 0.0
    for utt in transcript:
        words = len(utt["text"].split())
        secs  = max(1, (utt["endMilliseconds"] - utt["startMilliseconds"]) / 1000)
        id2info[utt["id"]] = {"text": utt["text"], "words": words}
        total_words += words
        total_secs  += secs

    WORDS_PER_SEC = total_words / total_secs if total_secs else 2.0

    # 2) ------------------- read XLSX labels ----------------------------- #
    df = pd.read_excel(xlsx_path, sheet_name=COLUMN_MAP["sheet"])
    df = df.dropna(subset=[COLUMN_MAP["id"], COLUMN_MAP["start_seconds"]]).reset_index(drop=True)

    # 3) -------- compute words & est. end for each utterance ------------- #
    est_end, utter_words = [], []
    for _, row in df.iterrows():
        utt_id = int(row[COLUMN_MAP["id"]])
        words  = id2info.get(utt_id, {}).get("words", 0)
        utter_words.append(words)
        est_dur = math.ceil(words / WORDS_PER_SEC) if words else 1
        est_end.append(int(row[COLUMN_MAP["start_seconds"]] + est_dur))

    df["est_end"] = est_end
    df["words"]   = utter_words

    # 4) -------- build per-row label & id sets --------------------------- #
    n = len(df)
    labels_per_field = {f: [set() for _ in range(n)] for f in MULTI_LABEL_FIELDS}
    ids_per_field    = {f: [set() for _ in range(n)] for f in MULTI_LABEL_FIELDS}

    single_fields = set(f for f in AUDIO_LABEL_COLUMNS.values()) - MULTI_LABEL_FIELDS
    for f in single_fields:
        labels_per_field[f] = [None] * n
        ids_per_field[f]    = [set() for _ in range(n)]

    for idx, row in df.iterrows():
        utt_id = int(row[COLUMN_MAP["id"]])
        for col, (field, label) in AUDIO_LABEL_COLUMNS.items():
            if pd.notna(row.get(col)) and int(row[col]) == 1:
                if field in MULTI_LABEL_FIELDS:
                    labels_per_field[field][idx].add(label)
                else:
                    labels_per_field[field][idx] = label
                ids_per_field[field][idx].add(utt_id)

    # 5) ---------------------- segment building -------------------------- #
    output = defaultdict(list)
    starts = df[COLUMN_MAP["start_seconds"]].astype(int).tolist()
    ends   = df["est_end"].tolist()

    for field in labels_per_field:
        lbl_series = [tuple(sorted(v)) if isinstance(v, set) else v
                      for v in labels_per_field[field]]
        id_series  = [tuple(sorted(v)) for v in ids_per_field[field]]

        for s_idx, e_idx, labels in contiguous(range(n), lbl_series):
            seg_ids = set()
            for i in range(s_idx, e_idx + 1):
                seg_ids.update(id_series[i])

            segment = {
                "start":  starts[s_idx],
                "end":    ends[e_idx],
                "labels": list(labels) if isinstance(labels, tuple) else [labels],
                "ids":    sorted(seg_ids),
            }
            output[field].append(segment)

    # 6) --------------------------- dump --------------------------------- #
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("✅  Audio JSON")


# ──────────────────────────────── CLI ─────────────────────────────────── #
def main():
    parser = argparse.ArgumentParser(
        description="Convert audio labels and transcript to combined JSON."
    )
    parser.add_argument(
        "base_name",
        help="File base-name (no extension). "
             "Script expects ./data/audio/<base_name>.xlsx and .json",
    )
    args = parser.parse_args()
    convert(args.base_name)


if __name__ == "__main__":
    main()
