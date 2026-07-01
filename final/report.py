"""
Stage 5 — reporting: merge the per-technique results from Stages 2-4 into one
master comparison table and render the headline figures for the paper.

Inputs (produced by the earlier stages):
  output/separability_summary.csv   (Stage 2)
  output/diversity_summary.csv      (Stage 3)
  output/consistency_summary.csv    (Stage 4)

Outputs:
  output/report_master.csv          raw merged metrics, one row per technique
  output/report_master_normalized.csv  min-max normalized, oriented so higher=better
  output/fig_heatmap.png            normalized metric heatmap (technique x metric)
  output/fig_tradeoff.png           diversity vs separability scatter
  output/fig_headline.png           4-panel headline bars

Run:  .venv/Scripts/python report.py
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loader import TECHNIQUES, BASE

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
sns.set_theme(style="whitegrid", font_scale=1.0)

OUT = BASE / "output"

# task_<id> order -> short technique name; used to order rows consistently.
TECH_ORDER = [TECHNIQUES[f"task_{i}"] for i in range(1, 7)]
SHORT = {
    "intent_classification_utility": "T1 intent_clf",
    "natural_language_distribution_diversity": "T2 nat_lang",
    "scenario-path_domain_coverage": "T3 scenario",
    "taboo_opening_words": "T4 taboo",
    "task-specific_hints": "T5 hints",
    "all_enhancements": "T6 all",
}

# (column, nice label, direction)  direction: +1 higher=better, -1 lower=better
METRIC_SPEC = [
    ("accuracy",            "Separability acc",  +1),
    ("macro_f1",            "Separability F1",   +1),
    ("separation_gap",      "Semantic sep",      +1),
    ("mattr",               "MATTR",             +1),
    ("distinct_2",          "Distinct-2",        +1),
    ("mean_edit_dist",      "Edit dist",         +1),
    ("self_bleu",           "Self-BLEU",         -1),
    ("mean_pairwise_cos",   "Semantic div",      -1),
    ("label_entropy",       "Label entropy",     +1),
    ("mean_js_label",       "Label JS",          -1),
    ("cross_run_dup_rate",  "Cross-run dup",     -1),
    ("near_dup_cross_run",  "Near dup",          -1),
    ("centroid_drift",      "Semantic drift",    -1),
]


def load_master() -> pd.DataFrame:
    sep = pd.read_csv(OUT / "separability_summary.csv", index_col=0)
    div = pd.read_csv(OUT / "diversity_summary.csv", index_col=0)

    sep = sep.drop(index="ALL_POOLED", errors="ignore")  # keep only the 6 techniques

    blocks = [
        sep[["accuracy", "macro_f1"]],
        div[["mattr", "distinct_1", "distinct_2", "self_bleu",
             "mean_edit_dist", "label_entropy"]],
    ]
    # Consistency (Stage 4) is out of scope for this paper — include only if present.
    con_path = OUT / "consistency_summary.csv"
    if con_path.exists():
        con = pd.read_csv(con_path, index_col=0)
        blocks.append(con[["mean_js_label", "label_prop_std", "cross_run_dup_rate", "near_dup_cross_run"]])
    # Semantic (Stage 3b) is optional — include only if embeddings were computed.
    sem_path, drift_path = OUT / "semantic_summary.csv", OUT / "semantic_centroid_drift.csv"
    if sem_path.exists():
        sem = pd.read_csv(sem_path, index_col=0)
        blocks.append(sem[["mean_pairwise_cos", "centroid_dispersion",
                           "intra_label_cos", "inter_label_cos", "separation_gap"]])
    if drift_path.exists():
        blocks.append(pd.read_csv(drift_path, index_col=0)[["centroid_drift", "run_centroid_spread"]])

    master = pd.concat(blocks, axis=1)

    master = master.reindex([t for t in TECH_ORDER if t in master.index])
    return master


def normalize_oriented(master: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize each metric to [0,1] and orient so higher = better."""
    norm = pd.DataFrame(index=master.index)
    for col, label, direction in METRIC_SPEC:
        if col not in master.columns:
            continue
        v = master[col].astype(float)
        rng = v.max() - v.min()
        scaled = (v - v.min()) / rng if rng else pd.Series(0.5, index=v.index)
        norm[label] = scaled if direction > 0 else (1 - scaled)
    return norm


def fig_heatmap(norm: pd.DataFrame):
    disp = norm.rename(index=SHORT)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(disp, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, cbar_kws={"label": "normalized (1 = best across techniques)"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    plt.setp(ax.get_xticklabels(), ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "fig_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  saved fig_heatmap.png")


def fig_tradeoff(master: pd.DataFrame):
    x = 1 - master["self_bleu"]          # higher = more lexically diverse
    y = master["accuracy"]               # higher = more separable
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("Set2", len(master))
    for i, tech in enumerate(master.index):
        hl = tech == "all_enhancements"
        ax.scatter(x[tech], y[tech], s=320 if hl else 200,
                   color=colors[i], edgecolor="black",
                   linewidth=1.8 if hl else 0.8, zorder=3)
        ax.annotate(SHORT[tech], (x[tech], y[tech]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold" if hl else "normal")
    ax.set_xlabel("Lexical diversity  (1 - Self-BLEU),  higher = more diverse", fontsize=11)
    ax.set_ylabel("Separability  (5-fold accuracy),  higher = more learnable", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_tradeoff.png", bbox_inches="tight")
    plt.close()
    print("  saved fig_tradeoff.png")


def fig_headline(master: pd.DataFrame):
    panels = [
        ("accuracy",           "Separability (accuracy)",      "higher = more separable", False),
        ("self_bleu",          "Diversity (Self-BLEU)",        "lower = more diverse",    True),
        ("mean_js_label",      "Label-mix stability (JS)",     "lower = more stable",     True),
        ("cross_run_dup_rate", "Cross-run redundancy",         "lower = more novel",      True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    labels = [SHORT[t] for t in master.index]
    for ax, (col, title, sub, lower_better) in zip(axes.flatten(), panels):
        vals = master[col].values
        best = vals.min() if lower_better else vals.max()
        colors = ["#2ca02c" if v == best else "#7f9fbf" for v in vals]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(f"{title}\n({sub})", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right", fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.15)
    plt.suptitle("Variance Techniques — Headline Metrics (green = best)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig_headline.png", bbox_inches="tight")
    plt.close()
    print("  saved fig_headline.png")


def main():
    master = load_master()
    norm = normalize_oriented(master)

    master.to_csv(OUT / "report_master.csv")
    norm.to_csv(OUT / "report_master_normalized.csv")

    sep = "=" * 100
    print(sep)
    print("STAGE 5 - MASTER COMPARISON TABLE (raw)")
    print(sep)
    print(master.round(3).to_string())

    print(f"\nOverall 'goodness' (mean of normalized, oriented metrics):")
    overall = norm.mean(axis=1).sort_values(ascending=False)
    for tech, score in overall.items():
        print(f"  {SHORT[tech]:<18} {score:.3f}")

    print("\nRendering figures...")
    fig_heatmap(norm)
    fig_tradeoff(master)
    fig_headline(master)
    print(f"\nSaved master tables + 3 figures to {OUT}/")


if __name__ == "__main__":
    main()
