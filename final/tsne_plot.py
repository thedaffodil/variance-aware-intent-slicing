"""
2D embedding visualization (PCA(50) -> t-SNE), colored by label and by technique.

Mirrors eda_old.py fig-6 but for the two-field schema and per-technique grouping.

Run:  .venv/Scripts/python tsne_plot.py
Out:  output/fig_tsne.png
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from loader import LABELS, TECHNIQUES, BASE
from embed import load_embeddings, MODEL_NAME

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
sns.set_theme(style="whitegrid", font_scale=1.0)

OUT = BASE / "output"
SEED = 42
TSNE_SAMPLE = 4000      # cap for a readable/fast t-SNE
TECH_ORDER = [TECHNIQUES[f"task_{i}"] for i in range(1, 7)]
SHORT = {
    "intent_classification_utility": "T1 intent_clf",
    "natural_language_distribution_diversity": "T2 nat_lang",
    "scenario-path_domain_coverage": "T3 scenario",
    "taboo_opening_words": "T4 taboo",
    "task-specific_hints": "T5 hints",
    "all_enhancements": "T6 all",
}


def scatter(ax, xy, categories, cats_order, title, palette):
    colors = sns.color_palette(palette, len(cats_order))
    cmap = {c: colors[i] for i, c in enumerate(cats_order)}
    for cat in cats_order:
        m = categories == cat
        label = cat.replace("slice_", "") if cat in LABELS else SHORT.get(cat, cat)
        ax.scatter(xy[m, 0], xy[m, 1], s=7, alpha=0.6, color=[cmap[cat]], label=label)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2.5, fontsize=8, framealpha=0.85, loc="best")
    ax.grid(True, alpha=0.2)


# Three techniques chosen to span the corners of the diversity-separability space:
#   scenario  -> most separable / most stable (separability corner)
#   all       -> highest lexical diversity / zero cross-run repetition (diversity corner)
#   intent..  -> highest semantic diversity but LOWEST class separation (overlap corner)
# Three techniques spanning the corners of the diversity-separability space,
# each written as a SEPARATE title-less panel -> LaTeX arranges them as (a)(b)(c).
SELECTED = [("scenario-path_domain_coverage", "a"),
            ("all_enhancements", "b"),
            ("intent_classification_utility", "c")]


def fig_per_technique(xy, meta):
    """Save one title-less PNG per selected technique (its points colored by label,
    the other settings greyed), on shared t-SNE coordinates."""
    techs = meta["technique"].to_numpy()
    labs = meta["slicing_operation"].to_numpy()
    colors = sns.color_palette("tab10", len(LABELS))
    cmap = {l: colors[i] for i, l in enumerate(LABELS)}

    for tech, tag in SELECTED:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        sel = techs == tech
        ax.scatter(xy[~sel, 0], xy[~sel, 1], s=5, alpha=0.10, color="lightgray", zorder=1)
        for lab in LABELS:
            m = sel & (labs == lab)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.75,
                           color=[cmap[lab]], label=lab.replace("slice_", ""), zorder=2)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.2)
        ax.legend(markerscale=1.5, fontsize=7, framealpha=0.9, loc="best", ncol=2)
        plt.tight_layout()
        plt.savefig(OUT / f"fig_tsne_{tag}.png", bbox_inches="tight")
        plt.close()
        print(f"  saved fig_tsne_{tag}.png")


def main():
    emb, meta = load_embeddings()
    rng = np.random.RandomState(SEED)
    if len(emb) > TSNE_SAMPLE:
        idx = rng.choice(len(emb), TSNE_SAMPLE, replace=False)
        emb, meta = emb[idx], meta.iloc[idx].reset_index(drop=True)
    print(f"t-SNE on {len(emb):,} embeddings...")

    pca = PCA(n_components=50, random_state=SEED)
    emb_pca = pca.fit_transform(emb)
    print(f"  PCA(50) variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    xy = TSNE(n_components=2, perplexity=40, max_iter=1000,
              random_state=SEED, init="pca").fit_transform(emb_pca)

    # Label-only overview (single panel; the by-technique panel was dropped as too cluttered)
    fig, ax = plt.subplots(figsize=(10, 9))
    scatter(ax, xy, meta["slicing_operation"].to_numpy(), LABELS,
            "By Slicing Operation (label)", "tab10")
    plt.suptitle(f"t-SNE — {MODEL_NAME} -> PCA(50) -> t-SNE",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "fig_tsne.png", bbox_inches="tight")
    plt.close()
    print("  saved fig_tsne.png")

    # Per-technique panels for the three selected techniques
    fig_per_technique(xy, meta)


if __name__ == "__main__":
    main()
