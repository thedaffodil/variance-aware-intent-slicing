"""
Axis-weighted goodness score for the general report (Separability + Diversity only;
consistency is excluded by design here).

Why a weighted (not flat) score?
  Flat per-metric averaging silently weights an axis by HOW MANY metrics it has.
  Diversity has more metrics than separability, so a flat mean would over-weight
  diversity. We instead give the two axes EQUAL weight, and inside each axis give
  its sub-components equal weight, so metric count never decides the ranking.

Hierarchy (each level split equally):
  Goodness = 1/2 Separability + 1/2 Diversity
    Separability = 1/2 Learnability + 1/2 SemanticSeparation
        Learnability       = mean(accuracy, macro_f1)        # correlated -> grouped
        SemanticSeparation = separation_gap                  # mpnet
    Diversity = 1/2 Lexical + 1/2 Semantic
        Lexical  = mean(MATTR, Distinct-2, 1-Self-BLEU)
        Semantic = mean(SemDiv=1-pairwise_cos, centroid_dispersion)   # mpnet

Reads output/report_master.csv (must be the mpnet run).
Run:  .venv/Scripts/python goodness_weighted.py
"""

import pandas as pd
from loader import BASE, TECHNIQUES

OUT = BASE / "output"
TECH_ORDER = [TECHNIQUES[f"task_{i}"] for i in range(1, 7)]
SHORT = {
    "intent_classification_utility": "T1 intent_clf",
    "natural_language_distribution_diversity": "T2 nat_lang",
    "scenario-path_domain_coverage": "T3 scenario",
    "taboo_opening_words": "T4 taboo",
    "task-specific_hints": "T5 hints",
    "all_enhancements": "T6 all",
}


def minmax(s, higher_better=True):
    rng = s.max() - s.min()
    z = (s - s.min()) / rng if rng else pd.Series(0.5, index=s.index)
    return z if higher_better else 1 - z


def main():
    m = pd.read_csv(OUT / "report_master.csv", index_col=0)

    # normalized, oriented so higher = better
    n_acc = minmax(m["accuracy"])
    n_mf1 = minmax(m["macro_f1"])
    n_sep = minmax(m["separation_gap"])
    n_mattr = minmax(m["mattr"])
    n_d2 = minmax(m["distinct_2"])
    n_sbleu = minmax(m["self_bleu"], higher_better=False)
    n_semdiv = minmax(m["mean_pairwise_cos"], higher_better=False)  # 1 - cos
    n_disp = minmax(m["centroid_dispersion"])

    learnability = (n_acc + n_mf1) / 2
    semantic_sep = n_sep
    separability = (learnability + semantic_sep) / 2

    lexical = (n_mattr + n_d2 + n_sbleu) / 3
    semantic = (n_semdiv + n_disp) / 2
    diversity = (lexical + semantic) / 2

    goodness = (separability + diversity) / 2

    out = pd.DataFrame({
        "Learnability": learnability, "Sem.Separation": semantic_sep,
        "Separability": separability, "Lexical": lexical, "Sem.Diversity": semantic,
        "Diversity": diversity, "GOODNESS": goodness,
    }).reindex(TECH_ORDER).round(3)
    out.index = [SHORT[t] for t in out.index]
    out = out.sort_values("GOODNESS", ascending=False)
    out.to_csv(OUT / "goodness_weighted.csv")

    print("=" * 78)
    print("AXIS-WEIGHTED GOODNESS  (Separability 50% + Diversity 50%)")
    print("=" * 78)
    print(out.to_string())
    print(f"\nSaved {OUT / 'goodness_weighted.csv'}")


if __name__ == "__main__":
    main()
