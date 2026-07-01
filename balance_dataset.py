"""
Build a class- and technique-balanced copy of the corpus under final/generated_data.

Balancing rule: take exactly k examples for every (technique, label) pair, where
k = min count over all (technique, label) pairs. This makes every technique the
same size AND every label equally represented within each technique.

Run:  .venv/Scripts/python balance_dataset.py
"""

import json
import numpy as np
import pandas as pd

from loader import load_generated, LABELS, TECHNIQUES, BASE

SEED = 42
FINAL = BASE / "final"
OUT_DATA = FINAL / "generated_data"
OUT_DATA.mkdir(parents=True, exist_ok=True)

# technique -> task_<id> (reverse of TECHNIQUES)
TECH2TASK = {v: k for k, v in TECHNIQUES.items()}


def main():
    df, _ = load_generated()
    df = df[df["label_valid"] & (df["intent"] != "")].copy()

    # counts per (technique, label)
    ct = df.groupby(["technique", "slicing_operation"]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=LABELS, fill_value=0)
    print("Per-(technique,label) counts:")
    print(ct.to_string())
    k = int(ct.values.min())
    print(f"\nMinimum (technique,label) count -> k = {k}")
    print(f"Balanced size per technique = {k} x {len(LABELS)} = {k*len(LABELS)}")
    print(f"Balanced total = {k*len(LABELS)*df['technique'].nunique()}\n")

    rng = np.random.RandomState(SEED)
    written = []
    for tech in sorted(df["technique"].unique()):
        rows = []
        for lab in LABELS:
            pool = df[(df["technique"] == tech) & (df["slicing_operation"] == lab)]
            take = pool.sample(n=k, random_state=rng)
            rows.append(take)
        bal = pd.concat(rows).sample(frac=1, random_state=rng)  # shuffle within technique
        items = [{"intent": r.intent, "slicing_operation": r.slicing_operation}
                 for r in bal.itertuples()]
        task = TECH2TASK.get(tech, tech)
        fname = f"{task}_{tech}_1.json"
        (OUT_DATA / fname).write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        written.append((fname, len(items)))
        print(f"  wrote {fname}  ({len(items)} examples)")

    print(f"\nBalanced dataset written to {OUT_DATA}")


if __name__ == "__main__":
    main()
