"""Embedding-based retrieval over memory turns.

Uses all-MiniLM-L6-v2 -- the same embedder LoCoMo-Conv unified on, and cheap
enough to run on a CPU. Embeddings are cached to disk (graphs/embeddings) so the
index is built once and reused by every experiment.

This module is intentionally the ONLY place retrieval is implemented, because H1
must hold retrieval fixed while varying structure, and H3 must measure the graph
built on top of these embeddings.

Device: the encoder is pinned to CUDA when present and loaded ONCE per process.
The on-disk index is keyed by model name only, deliberately -- one index, built
once, reused by every run regardless of which device built it, which is what
keeps retrieval comparable across machines. The building device and dtype land in
the sibling .meta.json so a run manifest can state them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMB_CACHE = ROOT / "graphs" / "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

# One loaded model per (model_name, device), for the life of the process.
# Rebuilding it per query was the original bug: one full model load per question.
_MODELS: dict[str, object] = {}


def device() -> str:
    """CUDA when available, else CPU. Never raises -- a missing torch is fine."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 -- CPU inference still works without torch
        pass
    return "cpu"


def _key(texts: list[str], model: str) -> Path:
    h = hashlib.sha256(model.encode())
    for t in texts[:50]:
        h.update(t.encode("utf-8", "ignore"))
    h.update(str(len(texts)).encode())
    return EMB_CACHE / f"{h.hexdigest()[:24]}.npz"


def _model(model_name: str = MODEL_NAME):
    key = f"{model_name}|{device()}"
    if key not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS[key] = SentenceTransformer(model_name, device=device())
    return _MODELS[key]


def encode(texts: list[str], model_name: str = MODEL_NAME,
           batch_size: int = 64) -> np.ndarray:
    """L2-normalized embeddings (n, d) for `texts`, on the pinned device."""
    return np.asarray(
        _model(model_name).encode(  # type: ignore[attr-defined]
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200,
        ),
        dtype=np.float32,
    )


def embed(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Return L2-normalized embeddings (n, d). Cached to graphs/embeddings/."""
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = _key(texts, model_name)
    if cpath.exists():
        return np.load(cpath)["emb"]

    emb = encode(texts, model_name)
    np.savez_compressed(cpath, emb=emb)
    (cpath.with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "model": model_name,
                "n": len(texts),
                "dim": int(emb.shape[1]),
                "device": device(),
                "dtype": str(emb.dtype),
            },
            indent=2,
        )
    )
    return emb


def top_k(query: str, texts: list[str], k: int = 10,
          model_name: str = MODEL_NAME) -> list[tuple[int, float]]:
    """Return [(index, cosine_similarity), ...] for the top-k most similar texts."""
    if not texts:
        return []
    emb = embed(texts, model_name)
    q = encode([query], model_name)[0]
    sims = emb @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]
