import json
import re
from pathlib import Path
from collections import defaultdict, Counter


RESULTS_DIR = Path(__file__).parent / "results"


def load_json_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    # Strip markdown code fences if present
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    if text:
        return json.loads(text)

    return None

def analyse():
    json_files = sorted(RESULTS_DIR.glob("*.json"))

    total_files = len(json_files)
    total_items = 0
    operation_counts = defaultdict(int)  # per-operation totals
    per_task_operation_counts = defaultdict(lambda: defaultdict(int))

    all_intents = []  # (intent, file_name, index)

    for path in json_files:
        items = load_json_file(path)
        if items is None:
            print(f"Skipping file: {path}")


        total_items += len(items)
        for i, item in enumerate(items):
            op = item.get("slicing_operation", "UNKNOWN")
            intent = item.get("intent", "")
            operation_counts[op] += 1
            per_task_operation_counts[path.name][op] += 1
            all_intents.append((intent.strip(), path.name, i + 1))

    # Duplicate intent detection (exact match)
    intent_locations = defaultdict(list)
    for intent, fname, idx in all_intents:
        intent_locations[intent].append((fname, idx))

    duplicates = {k: v for k, v in intent_locations.items() if len(v) > 1}

    # ── Report ────────────────────────────────────────────────────────────────
    sep = "=" * 70

    print(sep)
    print("DATA GENERATION ANALYSIS REPORT")
    print(sep)

    print(f"\nTotal JSON files processed : {total_files}")
    print(f"Total items (intents)      : {total_items}")

    # Files grouped by task prefix
    task_prefixes = sorted({re.match(r"(task_\d+_[\w-]+)_\d+\.json", f.name).group(1)
                             for f in json_files
                             if re.match(r"(task_\d+_[\w-]+)_\d+\.json", f.name)})

    print(f"\nFiles per task:")
    for prefix in task_prefixes:
        count = sum(1 for f in json_files
                    if re.match(rf"{re.escape(prefix)}_\d+\.json", f.name))
        print(f"  {prefix:<50} {count} files")

    print(f"\nItems per slicing_operation (all files combined):")
    for op, count in sorted(operation_counts.items(), key=lambda x: -x[1]):
        pct = count / total_items * 100
        print(f"  {op:<40} {count:>5}  ({pct:.1f}%)")

    print(f"\nPer-task breakdown:")
    for prefix in task_prefixes:
        task_files = [f for f in json_files
                      if re.match(rf"{re.escape(prefix)}_\d+\.json", f.name)]
        task_total = sum(sum(per_task_operation_counts[f.name].values()) for f in task_files)
        print(f"\n  [{prefix}]  ({len(task_files)} files, {task_total} items)")
        combined = defaultdict(int)
        for f in task_files:
            for op, cnt in per_task_operation_counts[f.name].items():
                combined[op] += cnt
        for op, cnt in sorted(combined.items(), key=lambda x: -x[1]):
            print(f"    {op:<40} {cnt:>5}")

    print(f"\n{sep}")
    print("DUPLICATE INTENT CHECK")
    print(sep)

    if not duplicates:
        print("\nNo duplicate intents found across all files.")
    else:
        print(f"\nFound {len(duplicates)} duplicate intent(s):\n")
        for i, (intent, locations) in enumerate(sorted(duplicates.items()), 1):
            print(f"  [{i}] \"{intent[:100]}{'...' if len(intent) > 100 else ''}\"")
            for fname, idx in locations:
                print(f"       -> {fname}  (item #{idx})")
            print()

    print(sep)
    print(f"Total unique intents : {len(intent_locations)}")
    print(f"Total duplicates     : {len(duplicates)}")
    print(sep)


if __name__ == "__main__":
    analyse()
