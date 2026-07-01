"""
Embedding cache — encode every intent once with a sentence-transformer and
persist to disk so the semantic-variance stages don't re-encode on every run.

The model is set by MODEL_NAME (default all-mpnet-base-v2). Cache files are
named per model, so multiple models can coexist on disk. Embeddings are
normalized so the dot product equals cosine similarity.

Run:    .venv/Scripts/python embed.py
Out:    output/embeddings_<model>.npy        float32 [n, dim], row-aligned to ...
        output/embeddings_meta_<model>.csv   intent, slicing_operation, technique, task, run, file
Reuse:  from embed import load_embeddings   ->  (emb, meta_df)
"""

try:  # use the OS (Windows) trust store so corporate SSL interception works
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import pandas as pd

from loader import load_generated, BASE

OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

MODEL_NAME = "all-mpnet-base-v2"   # primary model for the general report
_SLUG = MODEL_NAME.replace("/", "__")
EMB_PATH = OUT / f"embeddings_{_SLUG}.npy"
META_PATH = OUT / f"embeddings_meta_{_SLUG}.csv"


def build(force: bool = False):
    """Encode all valid intents and cache embeddings + aligned metadata."""
    df, _ = load_generated()
    df = df[df["label_valid"] & (df["intent"] != "")].reset_index(drop=True)

    if EMB_PATH.exists() and META_PATH.exists() and not force:
        emb = np.load(EMB_PATH)
        if emb.shape[0] == len(df):
            print(f"Cache hit: {EMB_PATH.name} ({emb.shape}) — use force=True to rebuild.")
            return emb, pd.read_csv(META_PATH)

    from sentence_transformers import SentenceTransformer
    print(f"Encoding {len(df):,} intents with {MODEL_NAME} (CPU)...")
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(df["intent"].tolist(), batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True).astype(np.float32)

    meta = df[["intent", "slicing_operation", "technique", "task", "run", "file"]].copy()
    np.save(EMB_PATH, emb)
    meta.to_csv(META_PATH, index=False)
    print(f"Saved {EMB_PATH.name} {emb.shape} and {META_PATH.name}")
    return emb, meta


def load_embeddings():
    """Load cached embeddings + metadata, building them if missing."""
    if EMB_PATH.exists() and META_PATH.exists():
        return np.load(EMB_PATH), pd.read_csv(META_PATH)
    return build()


if __name__ == "__main__":
    build()
