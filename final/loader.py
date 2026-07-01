"""
Stage 0 — shared loader for the variance-technique study.

Reads every generated_data/*.json file, strips markdown ```json fences, and
returns a tidy DataFrame with one row per generated example, tagged with the
variance `technique` (task prefix) and `run` (per-technique file index).

Import:   from loader import load_generated, TECHNIQUES, LABELS
Run:      python loader.py        # prints a load report
"""

import re
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
DATA_DIR = BASE / "generated_data"

# The 8 valid labels, defined in dataGen_codes(thread+analyse)/prompt.txt
LABELS = [
    "slice_allocation",
    "slice_deallocation",
    "slice_list",
    "slice_ue_quota_update",
    "slice_pdu_session_quota_update",
    "slice_rb_update",
    "slice_rrc_con_update",
    "other",
]
LABEL_SET = set(LABELS)

# Maps the task_<id> prefix to a short technique name used everywhere downstream.
TECHNIQUES = {
    "task_1": "intent_classification_utility",
    "task_2": "natural_language_distribution_diversity",
    "task_3": "scenario-path_domain_coverage",
    "task_4": "taboo_opening_words",
    "task_5": "task-specific_hints",
    "task_6": "all_enhancements",
}

# filename: task_<id>_<name>_<run>.json   ->   capture task_<id> and the trailing run index
_FNAME_RE = re.compile(r"^(task_\d+)_.+_(\d+)\.json$")


def _strip_fences(text: str) -> str:
    """Remove a leading ```json (or ```) fence and a trailing ``` if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def load_json_file(path: Path):
    """Parse one generated file to a list of dicts, or None if it can't be read."""
    raw = path.read_text(encoding="utf-8")
    cleaned = _strip_fences(raw)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else [data]


def load_generated(data_dir: Path = DATA_DIR):
    """
    Load all generated_data files into a tidy DataFrame.

    Columns: intent, slicing_operation, technique, task, run, file, label_valid
    Returns: (df, report)  where report is a dict of load diagnostics.
    """
    rows, skipped_files, unknown_tasks = [], [], set()

    for path in sorted(data_dir.glob("*.json")):
        m = _FNAME_RE.match(path.name)
        if not m:
            skipped_files.append((path.name, "filename pattern mismatch"))
            continue
        task, run = m.group(1), int(m.group(2))
        technique = TECHNIQUES.get(task)
        if technique is None:
            unknown_tasks.add(task)
            technique = task  # fall back to the raw prefix rather than dropping data

        items = load_json_file(path)
        if items is None:
            skipped_files.append((path.name, "empty or invalid JSON"))
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            intent = (item.get("intent") or "").strip()
            op = (item.get("slicing_operation") or "").strip()
            rows.append({
                "intent": intent,
                "slicing_operation": op,
                "technique": technique,
                "task": task,
                "run": run,
                "file": path.name,
                "label_valid": op in LABEL_SET,
            })

    df = pd.DataFrame(rows)
    report = {
        "n_files_loaded": df["file"].nunique() if len(df) else 0,
        "n_files_skipped": len(skipped_files),
        "skipped_files": skipped_files,
        "n_examples": len(df),
        "n_invalid_labels": int((~df["label_valid"]).sum()) if len(df) else 0,
        "n_empty_intents": int((df["intent"] == "").sum()) if len(df) else 0,
        "unknown_tasks": sorted(unknown_tasks),
    }
    return df, report


def print_report(df: pd.DataFrame, report: dict) -> None:
    sep = "=" * 70
    print(sep)
    print("STAGE 0 - LOAD REPORT")
    print(sep)
    print(f"Files loaded        : {report['n_files_loaded']}")
    print(f"Files skipped       : {report['n_files_skipped']}")
    for name, why in report["skipped_files"]:
        print(f"    - {name}  ({why})")
    print(f"Total examples      : {report['n_examples']:,}")
    print(f"Invalid labels      : {report['n_invalid_labels']}")
    print(f"Empty intents       : {report['n_empty_intents']}")
    if report["unknown_tasks"]:
        print(f"Unknown task prefixes: {report['unknown_tasks']}")

    if not len(df):
        print("\nNo data loaded.")
        return

    print(f"\nExamples per technique:")
    per_tech = df.groupby("technique").agg(
        examples=("intent", "size"),
        runs=("run", "nunique"),
    ).sort_values("examples", ascending=False)
    for tech, row in per_tech.iterrows():
        print(f"  {tech:<42} {row['examples']:>5} examples  ({row['runs']} runs)")

    print(f"\nLabel distribution (overall):")
    for label, n in df["slicing_operation"].value_counts().items():
        flag = "" if label in LABEL_SET else "  <-- INVALID"
        print(f"  {label:<34} {n:>5}  ({n/len(df):.1%}){flag}")


if __name__ == "__main__":
    df, report = load_generated()
    print_report(df, report)
