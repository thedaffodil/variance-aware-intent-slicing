"""
Stage 3 — diversity per variance technique.

Measures how lexically / structurally varied each technique's generated intents
are. Higher diversity = the technique produces less repetitive text.

Metrics (per technique):
  - unique_intent_ratio : 1 - exact-duplicate rate (corpus level)
  - mattr               : moving-average type-token ratio, window=50 (length-robust vocab richness)
  - distinct_1          : unique unigrams / total unigrams
  - distinct_2          : unique bigrams  / total bigrams
  - self_bleu           : mean BLEU-4 of each intent vs the others  (LOWER = more diverse)
  - mean_edit_dist      : mean char Levenshtein over sampled intent pairs (HIGHER = more diverse)
  - label_entropy       : Shannon entropy of the 8-label distribution, normalized to [0,1]

Fairness: size-sensitive metrics are computed on a fixed random sub-sample so
techniques with more files are not unfairly penalized/rewarded. The O(n^2)
metrics (self-BLEU, edit distance) use a smaller pair sample.

Run:  .venv/Scripts/python diversity.py
Out:  output/diversity_summary.csv  +  printed report
"""

import math
import warnings

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from loader import load_generated, LABELS, BASE

warnings.filterwarnings("ignore")

OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

LEX_SAMPLE = 1500   # for size-sensitive corpus metrics (TTR, distinct-n)
PAIR_SAMPLE = 400   # for O(n^2) metrics (self-BLEU, pairwise edit distance)
MATTR_WINDOW = 50
SEED = 42
_SMOOTH = SmoothingFunction().method1


def _tokens(texts):
    """Flat lowercase whitespace token list across a list of strings."""
    toks = []
    for t in texts:
        toks.extend(t.lower().split())
    return toks


def mattr(texts, window=MATTR_WINDOW):
    """Moving-average type-token ratio — length-robust vocabulary richness."""
    toks = _tokens(texts)
    if len(toks) < window:
        return len(set(toks)) / max(len(toks), 1)
    ratios = []
    for i in range(len(toks) - window + 1):
        win = toks[i:i + window]
        ratios.append(len(set(win)) / window)
    return float(np.mean(ratios))


def distinct_n(texts, n):
    """Unique n-grams / total n-grams across the corpus."""
    total, seen = 0, set()
    for t in texts:
        toks = t.lower().split()
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


def self_bleu(texts, rng):
    """
    Mean BLEU-4 of each sampled intent against all other sampled intents as
    references. Lower = more diverse (less n-gram overlap between intents).
    """
    sample = texts if len(texts) <= PAIR_SAMPLE else \
        [texts[i] for i in rng.choice(len(texts), PAIR_SAMPLE, replace=False)]
    tok = [s.lower().split() for s in sample]
    scores = []
    for i in range(len(tok)):
        refs = tok[:i] + tok[i + 1:]
        if not tok[i]:
            continue
        scores.append(sentence_bleu(refs, tok[i], smoothing_function=_SMOOTH))
    return float(np.mean(scores)) if scores else 0.0


def mean_edit_distance(texts, rng):
    """Mean character Levenshtein distance over sampled intent pairs."""
    sample = texts if len(texts) <= PAIR_SAMPLE else \
        [texts[i] for i in rng.choice(len(texts), PAIR_SAMPLE, replace=False)]
    dists = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dists.append(Levenshtein.distance(sample[i], sample[j]))
    return float(np.mean(dists)) if dists else 0.0


def label_entropy(labels):
    """Shannon entropy of the label distribution, normalized to [0,1] over 8 labels."""
    counts = pd.Series(labels).value_counts()
    p = counts / counts.sum()
    h = -(p * np.log2(p)).sum()
    return float(h / math.log2(len(LABELS)))


def evaluate_technique(intents, labels, rng):
    # Fixed sub-sample for size-sensitive corpus metrics
    if len(intents) > LEX_SAMPLE:
        idx = rng.choice(len(intents), LEX_SAMPLE, replace=False)
        lex = [intents[i] for i in idx]
    else:
        lex = list(intents)

    return {
        "n_examples": int(len(intents)),
        "unique_intent_ratio": len(set(intents)) / len(intents),
        "mattr": mattr(lex),
        "distinct_1": distinct_n(lex, 1),
        "distinct_2": distinct_n(lex, 2),
        "self_bleu": self_bleu(lex, rng),
        "mean_edit_dist": mean_edit_distance(lex, rng),
        "label_entropy": label_entropy(labels),
    }


def main():
    df, _ = load_generated()
    df = df[df["label_valid"] & (df["intent"] != "")].copy()
    print(f"Loaded {len(df):,} valid examples across {df['technique'].nunique()} techniques.")
    print(f"(lexical sample={LEX_SAMPLE}, pair sample={PAIR_SAMPLE} per technique)\n")

    rows = []
    for tech in sorted(df["technique"].unique()):
        sub = df[df["technique"] == tech]
        rng = np.random.RandomState(SEED)  # independent, reproducible per technique
        m = evaluate_technique(sub["intent"].values,
                               sub["slicing_operation"].values, rng)
        m["technique"] = tech
        rows.append(m)
        print(f"[{tech}]  n={m['n_examples']}")
        print(f"    unique={m['unique_intent_ratio']:.3f}  mattr={m['mattr']:.3f}  "
              f"distinct1={m['distinct_1']:.3f}  distinct2={m['distinct_2']:.3f}")
        print(f"    self_bleu={m['self_bleu']:.3f} (lower=diverse)  "
              f"edit_dist={m['mean_edit_dist']:.1f}  label_entropy={m['label_entropy']:.3f}\n")

    cols = ["n_examples", "unique_intent_ratio", "mattr", "distinct_1", "distinct_2",
            "self_bleu", "mean_edit_dist", "label_entropy"]
    summary = pd.DataFrame(rows).set_index("technique")[cols]
    out_csv = OUT / "diversity_summary.csv"
    summary.to_csv(out_csv)

    sep = "=" * 100
    print(sep)
    print("STAGE 3 - DIVERSITY SUMMARY  (higher=more diverse, EXCEPT self_bleu where lower=more diverse)")
    print(sep)
    print(summary.round(3).to_string())
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
