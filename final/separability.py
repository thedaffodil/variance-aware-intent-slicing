"""
Stage 2 — separability / learnability per variance technique.

Adapts eda_old.py's fig-9 baseline (TF-IDF + Logistic Regression under
StratifiedKFold(5) cross_val_predict) and runs it SEPARATELY for each variance
technique. There is no held-out test set: every example is predicted exactly
once while it is out-of-fold.

This measures how learnable / class-separable each technique's generated data
is — a different axis from label *correctness* (Stage 1). High separability
just means the intent text strongly predicts its assigned label; it does not
prove the label is the right one.

Run:  .venv/Scripts/python separability.py
Out:  output/separability_summary.csv  +  printed report
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report

from loader import load_generated, LABELS, BASE

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

N_SPLITS = 5
MIN_PER_CLASS = N_SPLITS  # need >= n_splits samples in each class to stratify


def build_pipeline() -> Pipeline:
    """Same recipe as eda_old.py fig-9."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000,
                                  ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)),
    ])


def evaluate_subset(intents: np.ndarray, labels: np.ndarray):
    """
    Run stratified 5-fold cross_val_predict on one technique's data.
    Returns (metrics_dict, y_true, y_pred) or (None, ...) if not enough data.
    """
    # StratifiedKFold needs every retained class to have >= n_splits members.
    counts = pd.Series(labels).value_counts()
    keep = counts[counts >= MIN_PER_CLASS].index
    mask = np.isin(labels, keep)
    intents, labels = intents[mask], labels[mask]

    if len(intents) < N_SPLITS or pd.Series(labels).nunique() < 2:
        return None, None, None

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    pipe = build_pipeline()
    y_pred = cross_val_predict(pipe, intents, labels, cv=cv, n_jobs=-1)

    metrics = {
        "n_examples": int(len(labels)),
        "n_classes": int(pd.Series(labels).nunique()),
        "n_dropped_rare": int((~mask).sum()),
        "accuracy": accuracy_score(labels, y_pred),
        "macro_f1": f1_score(labels, y_pred, average="macro"),
        "weighted_f1": f1_score(labels, y_pred, average="weighted"),
    }
    return metrics, labels, y_pred


def main():
    df, report = load_generated()
    df = df[df["label_valid"] & (df["intent"] != "")].copy()
    print(f"Loaded {len(df):,} valid examples across {df['technique'].nunique()} techniques.\n")

    rows = []
    techniques = sorted(df["technique"].unique())

    # Per-technique evaluation
    for tech in techniques:
        sub = df[df["technique"] == tech]
        metrics, y_true, y_pred = evaluate_subset(
            sub["intent"].values, sub["slicing_operation"].values)
        if metrics is None:
            print(f"[skip] {tech}: not enough data to stratify")
            continue
        metrics["technique"] = tech
        rows.append(metrics)
        print(f"[{tech}]")
        print(f"    examples={metrics['n_examples']:>5}  classes={metrics['n_classes']}  "
              f"dropped_rare={metrics['n_dropped_rare']}")
        print(f"    accuracy={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}  "
              f"weighted_f1={metrics['weighted_f1']:.3f}\n")

    # Pooled baseline across all techniques, for reference
    pooled_metrics, _, _ = evaluate_subset(
        df["intent"].values, df["slicing_operation"].values)
    if pooled_metrics is not None:
        pooled_metrics["technique"] = "ALL_POOLED"
        rows.append(pooled_metrics)

    summary = pd.DataFrame(rows).set_index("technique")
    cols = ["n_examples", "n_classes", "n_dropped_rare", "accuracy", "macro_f1", "weighted_f1"]
    summary = summary[cols]

    out_csv = OUT / "separability_summary.csv"
    summary.to_csv(out_csv)

    sep = "=" * 78
    print(sep)
    print("STAGE 2 - SEPARABILITY SUMMARY  (TF-IDF + LogReg, StratifiedKFold-5)")
    print(sep)
    print(summary.round(3).to_string())
    print(f"\nSaved {out_csv}")
    print("\nNote: higher accuracy/F1 = more class-separable text, NOT proof of")
    print("label correctness. Compare against Stage 1 (LLM-judge) for quality.")


if __name__ == "__main__":
    main()
