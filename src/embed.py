"""Embedding-based retrieval over memory turns.

Uses all-MiniLM-L6-v2 -- the same embedder LoCoMo-Conv unified on, and cheap
enough to run on a CPU. Embeddings are cached to disk (graphs/embeddings) so the
index is built once and reused by every experiment.

This module is intentionally the ONLY place retrieval is implemented, because H1
must hold retrieval fixed while varying structure, and H3 must measure the graph
built on top of these embeddings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMB_CACHE = ROOT / "graphs" / "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"


def _key(texts: list[str], model: str) -> Path:
    h = hashlib.sha256(model.encode())
    for t in texts[:50]:
        h.update(t.encode("utf-8", "ignore"))
    h.update(str(len(texts)).encode())
    return EMB_CACHE / f"{h.hexdigest()[:24]}.npz"


def embed(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Return L2-normalized embeddings (n, d). Cached to graphs/embeddings/."""
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = _key(texts, model_name)
    if cpath.exists():
        return np.load(cpath)["emb"]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 200,
                       batch_size=64)
    emb = np.asarray(emb, dtype=np.float32)
    np.savez_compressed(cpath, emb=emb)
    (cpath.with_suffix(".meta.json")).write_text(json.dumps(
        {"model": model_name, "n": len(texts)}, indent=2))
    return emb


def top_k(query: str, texts: list[str], k: int = 10,
          model_name: str = MODEL_NAME) -> list[tuple[int, float]]:
    """Return [(index, cosine_similarity), ...] for the top-k most similar texts."""
    if not texts:
        return []
    emb = embed(texts, model_name)
    from sentence_transformers import SentenceTransformer

    q = SentenceTransformer(model_name).encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)[0]
    sims = emb @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]
