"""
Stage 3b — SEMANTIC variance per variance technique (BERT embeddings).

Complements the lexical diversity of Stage 3 with semantic-space measures using
all-MiniLM-L6-v2 embeddings (normalized, so dot == cosine).

Per-technique metrics (output/semantic_summary.csv):
  - mean_pairwise_cos   : mean cosine over sampled intent pairs (LOWER = more diverse)
  - centroid_dispersion : mean (1 - cos(x, centroid)) = embedding radius (HIGHER = more diverse)
  - intra_label_cos     : mean within-label cosine (how tight each class cluster is)
  - inter_label_cos     : mean between-label cosine (how close different classes are)
  - separation_gap      : intra_label_cos - inter_label_cos (HIGHER = classes better separated)

Also writes, across the whole corpus:
  - output/fig_semantic_diversity.png   per-technique semantic diversity bars
  - output/fig_op_similarity.png        label x label mean cosine heatmap
  - output/semantic_centroid_drift.csv  per-technique across-run centroid drift (consistency)

Run:  .venv/Scripts/python semantic.py
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loader import LABELS, TECHNIQUES, BASE
from embed import load_embeddings, MODEL_NAME

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
sns.set_theme(style="whitegrid", font_scale=1.0)

OUT = BASE / "output"
PAIR_SAMPLE = 1200          # sampled rows for O(n^2) pairwise cosine, per technique
SEED = 42
TECH_ORDER = [TECHNIQUES[f"task_{i}"] for i in range(1, 7)]
SHORT = {
    "intent_classification_utility": "T1 intent_clf",
    "natural_language_distribution_diversity": "T2 nat_lang",
    "scenario-path_domain_coverage": "T3 scenario",
    "taboo_opening_words": "T4 taboo",
    "task-specific_hints": "T5 hints",
    "all_enhancements": "T6 all",
}


def mean_pairwise_cos(emb, rng):
    """Mean off-diagonal cosine on a random sample (normalized emb -> dot=cos)."""
    if len(emb) > PAIR_SAMPLE:
        emb = emb[rng.choice(len(emb), PAIR_SAMPLE, replace=False)]
    sim = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    return float(sim[iu].mean())


def centroid_dispersion(emb):
    """Mean (1 - cos to centroid). Higher = more spread in semantic space."""
    centroid = emb.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-12)
    cos_to_c = emb @ centroid
    return float((1 - cos_to_c).mean())


def intra_inter_label(emb, labels, rng):
    """Mean within-label and between-label cosine on a sample (for separation_gap)."""
    if len(emb) > PAIR_SAMPLE:
        idx = rng.choice(len(emb), PAIR_SAMPLE, replace=False)
        emb, labels = emb[idx], labels[idx]
    sim = emb @ emb.T
    same = labels[:, None] == labels[None, :]
    iu = np.triu_indices(len(emb), k=1)
    s, d = sim[iu], same[iu]
    intra = float(s[d].mean()) if d.any() else np.nan
    inter = float(s[~d].mean()) if (~d).any() else np.nan
    return intra, inter


def per_technique(emb, meta):
    rows = []
    for tech in TECH_ORDER:
        m = meta["technique"] == tech
        if not m.any():
            continue
        e = emb[m.values]
        labels = meta.loc[m, "slicing_operation"].to_numpy()
        rng = np.random.RandomState(SEED)
        mpc = mean_pairwise_cos(e, rng)
        disp = centroid_dispersion(e)
        intra, inter = intra_inter_label(e, labels, np.random.RandomState(SEED))
        rows.append({
            "technique": tech,
            "n_examples": int(m.sum()),
            "mean_pairwise_cos": mpc,
            "centroid_dispersion": disp,
            "intra_label_cos": intra,
            "inter_label_cos": inter,
            "separation_gap": intra - inter,
        })
    return pd.DataFrame(rows).set_index("technique")


def centroid_drift(emb, meta):
    """Across-run semantic stability: mean distance of each run centroid to the
    technique centroid (LOWER = runs land in the same semantic region)."""
    rows = []
    for tech in TECH_ORDER:
        m = (meta["technique"] == tech).values
        if not m.any():
            continue
        e = emb[m]
        runs = meta.loc[m, "run"].to_numpy()
        tech_c = e.mean(axis=0)
        tech_c /= (np.linalg.norm(tech_c) + 1e-12)
        run_cs = []
        for r in np.unique(runs):
            rc = e[runs == r].mean(axis=0)
            rc /= (np.linalg.norm(rc) + 1e-12)
            run_cs.append(rc)
        run_cs = np.array(run_cs)
        drift = float((1 - run_cs @ tech_c).mean())          # mean run->tech cosine distance
        spread = float(np.mean([1 - a @ b for a, b in combinations(run_cs, 2)])) \
            if len(run_cs) > 1 else np.nan                    # mean pairwise run-centroid distance
        rows.append({"technique": tech, "n_runs": len(run_cs),
                     "centroid_drift": drift, "run_centroid_spread": spread})
    return pd.DataFrame(rows).set_index("technique")


def fig_semantic_diversity(summary):
    disp = summary.rename(index=SHORT)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    specs = [("mean_pairwise_cos", "Mean pairwise cosine", "lower = more diverse", True),
             ("centroid_dispersion", "Centroid dispersion", "higher = more diverse", False),
             ("separation_gap", "Class separation gap", "higher = better separated", False)]
    for ax, (col, title, sub, lower_better) in zip(axes, specs):
        vals = disp[col].values
        best = vals.min() if lower_better else vals.max()
        colors = ["#2ca02c" if v == best else "#7f9fbf" for v in vals]
        bars = ax.bar(disp.index, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(f"{title}\n({sub})", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right", fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(OUT / "fig_semantic_diversity.png", bbox_inches="tight")
    plt.close()
    print("  saved fig_semantic_diversity.png")


def fig_op_similarity(emb, meta):
    """Label x label mean cosine heatmap over the whole corpus (sampled per label)."""
    rng = np.random.RandomState(SEED)
    cap = 500
    idx = []
    for lab in LABELS:
        pos = np.where(meta["slicing_operation"].values == lab)[0]
        if len(pos) > cap:
            pos = rng.choice(pos, cap, replace=False)
        idx.append(pos)
    idx = np.concatenate(idx)
    e, labs = emb[idx], meta["slicing_operation"].values[idx]

    n = len(LABELS)
    mat = np.zeros((n, n))
    for i, a in enumerate(LABELS):
        ea = e[labs == a]
        for j, b in enumerate(LABELS):
            eb = e[labs == b]
            if len(ea) == 0 or len(eb) == 0:
                continue
            sim = ea @ eb.T
            if i == j:
                m = np.triu(np.ones_like(sim, dtype=bool), k=1)
                mat[i, j] = sim[m].mean() if m.any() else 1.0
            else:
                mat[i, j] = sim.mean()

    short = [l.replace("slice_", "") for l in LABELS]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(mat, xticklabels=short, yticklabels=short, annot=True, fmt=".2f",
                cmap="RdYlGn_r", vmin=0.1, vmax=0.7, linewidths=0.4, ax=ax,
                cbar_kws={"label": "mean cosine similarity"})
    ax.tick_params(axis="x", rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "fig_op_similarity.png", bbox_inches="tight")
    plt.close()
    print("  saved fig_op_similarity.png")


def main():
    emb, meta = load_embeddings()
    print(f"Embeddings: {emb.shape}  meta: {len(meta):,} rows\n")

    summary = per_technique(emb, meta)
    drift = centroid_drift(emb, meta)
    summary.to_csv(OUT / "semantic_summary.csv")
    drift.to_csv(OUT / "semantic_centroid_drift.csv")

    sep = "=" * 96
    print(sep)
    print(f"STAGE 3b - SEMANTIC VARIANCE SUMMARY ({MODEL_NAME})")
    print(sep)
    print(summary.round(3).to_string())
    print("\nAcross-run centroid drift (consistency in semantic space, lower = stabler):")
    print(drift.round(4).to_string())

    print("\nRendering figures...")
    fig_semantic_diversity(summary)
    fig_op_similarity(emb, meta)
    print(f"\nSaved semantic tables + figures to {OUT}/")


if __name__ == "__main__":
    main()
