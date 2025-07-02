import argparse
import json
import os
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1.  Label maps & constants  (unchanged)
# --------------------------------------------------------------------------- #
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

MULTI_LABEL_FIELDS = {
    "studentPosition",
    "studentParticipation",
    "toolsAndRepresentationsInUse",
}


# --------------------------------------------------------------------------- #
# 2.  Utilities
# --------------------------------------------------------------------------- #
def contiguous_runs(times, payloads):
    """Yield (start, end, value) for stretches with identical value."""
    for val, grp in groupby(zip(times, payloads), key=lambda x: x[1]):
        if not val:
            continue                      # skip empty tuple / None / 0
        grp = list(grp)
        yield grp[0][0], grp[-1][0], val


# --------------------------------------------------------------------------- #
# 3.  Main conversion
# --------------------------------------------------------------------------- #
def convert_matrix_csv(csv_path: str, json_path: str) -> None:
    # --- load & reshape ---------------------------------------------------- #
    df = pd.read_csv(csv_path).set_index("Unnamed: 0")
    df.index = df.index.str.strip()

    second_cols = [c for c in df.columns if c.isdigit()]
    if not second_cols:
        raise ValueError("No numeric second columns like '0001', '0002', …")

    sec_idx = np.array([int(c) - 1 for c in second_cols])  # '0001' → 0 s
    matrix  = df[second_cols].T
    matrix.index = sec_idx
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    n_sec = len(matrix.index)

    # --- 1. pre-allocate per-second containers ---------------------------- #
    per_sec = {}
    for _, (field, _) in ROW_NAME_MAP.items():
        if field in MULTI_LABEL_FIELDS:
            per_sec[field] = [[] for _ in range(n_sec)]
        else:
            per_sec[field] = [None for _ in range(n_sec)]

    # --- 2. fill them according to the CSV -------------------------------- #
    for row_name, (field, label) in ROW_NAME_MAP.items():
        if row_name not in matrix.columns:
            continue
        active = matrix[row_name].values.astype(bool)
        for i in range(n_sec):
            if not active[i]:
                continue
            if field in MULTI_LABEL_FIELDS:
                per_sec[field][i].append(label)
            else:
                per_sec[field][i] = label   # overwrites if two rows collide

    # --- 3. build segments ------------------------------------------------- #
    field_segments = defaultdict(list)
    for field, values in per_sec.items():
        # normalise payloads for grouping
        if field in MULTI_LABEL_FIELDS:
            series = [tuple(sorted(v)) for v in values]
        else:
            series = values

        for s, e, val in contiguous_runs(matrix.index, series):
            field_segments[field].append(
                {"start": int(s), "end": int(e),
                 "labels": list(val) if isinstance(val, tuple) else [val]}
            )

    # --- 4. raised-hands count ------------------------------------------- #
    if "Raising_Hand" in matrix.columns:
        rh = matrix["Raising_Hand"].values.astype(int)
        rh_segs = [
            {"start": int(s), "end": int(e), "count": int(c)}
            for s, e, c in contiguous_runs(matrix.index, rh)
        ]
        if rh_segs:
            field_segments["raisedHands"] = rh_segs

    # --- 5. dump ---------------------------------------------------------- #
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(field_segments, fh, indent=2)

    print("✅  JSON saved")


# --------------------------------------------------------------------------- #
# 4.  CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a per-second label matrix CSV to JSON."
    )
    parser.add_argument(
        "csv_file",
        help="Name of the CSV file inside ./output (with or without '.csv')."
    )
    args = parser.parse_args()

    # Ensure the extension & paths
    csv_name = args.csv_file + "_MATRIX"
    if not csv_name.lower().endswith(".csv"):
        csv_name += ".csv"

    input_csv  = os.path.join("output", csv_name)
    base_name  = Path(csv_name).stem
    output_json = os.path.join("./output", f"{base_name}_VIDEO.json")

    convert_matrix_csv(input_csv, output_json)
