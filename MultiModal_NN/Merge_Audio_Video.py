import argparse
import json
import os
from typing import Any, Dict, List

def _merge_lists(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Concatenate two lists of interval dictionaries and sort by start → end time."""
    merged = (a or []) + (b or [])
    return sorted(merged, key=lambda item: (item.get("start", 0), item.get("end", 0)))

def merge_json(audio_path: str, video_path: str, output_path: str) -> None:
    """Merge two JSON files (audio & video) into a unified JSON file."""

    # Load the two modality‑specific JSONs
    with open(audio_path, "r", encoding="utf-8") as f:
        audio_data = json.load(f)
    with open(video_path, "r", encoding="utf-8") as f:
        video_data = json.load(f)

    unified: Dict[str, Any] = {}

    # First copy video‑derived fields, then extend/override with audio‑derived ones
    for source in (video_data, audio_data):
        for key, value in source.items():
            if key not in unified:
                unified[key] = value
            else:
                # When the same field exists in both files, merge their interval lists
                if isinstance(unified[key], list) and isinstance(value, list):
                    unified[key] = _merge_lists(unified[key], value)
                else:
                    # Fallback: prefer the second (audio) value when merging scalars
                    unified[key] = value

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the unified JSON with pretty formatting for readability
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    print(f"Merged file written to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge audio/transcript and video JSON outputs into a single Multi‑modal I/O JSON file."
    )
    parser.add_argument(
        "basename",
        help="Base file name without extension. The script expects to find ./data/audio/<basename>.json and ./output/<basename>.json."
    )
    args = parser.parse_args()

    base = args.basename
    audio_path = os.path.join("data", "audio", f"{base}.json")
    video_path = os.path.join("output", f"{base}.json")
    output_path = os.path.join("output", f"{base}_unified.json")

    # Sanity checks
    for p in (audio_path, video_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required input file not found: {p}")

    merge_json(audio_path, video_path, output_path)

if __name__ == "__main__":
    main()

