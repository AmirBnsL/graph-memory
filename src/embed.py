"""Embedding-based retrieval over memory turns.

RETRIEVER SPECS
---------------
A retriever is a *spec* (a plain dict from conf/retriever/*.yaml), not a bare
model name -- because the strong retrieval models are instruction- and
pooling-sensitive, and getting those wrong silently degrades them to MiniLM
performance while looking like a null result:

    name               HF id (or ST shorthand). e.g. all-MiniLM-L6-v2
    query_prefix       raw text PREPENDED to each query before encoding.
                       Qwen3-Embedding: "Instruct: {task}\nQuery:" -- the exact
                       format from the model card's reference implementation.
                       Must be config-visible so the manifest hash covers it.
    passage_prefix     raw text prepended to each passage (usually "").
    passage_template   how a Turn becomes the text that gets EMBEDDED.
                       "{text}" = bare turn text (what every existing result
                       used). The LoCoMo paper's dialog arm embeds
                       '{speaker} said, "{text}"' (data/locomo task_eval/gpt_utils.py
                       prepare_for_rag) -- speaker attribution matters on a
                       benchmark full of "what did X say" questions.
    batch_size         encoder batch size.
    trust_remote_code  forwarded to SentenceTransformer.

This module is intentionally the ONLY place retrieval is implemented, because H1
must hold retrieval fixed while varying structure, and H3 must measure the graph
built on top of these embeddings.

POOLING
-------
Each model's pooling lives in its own ST checkpoint (MiniLM: mean+normalize;
Qwen3-Embedding: lasttoken+normalize -- verified in the snapshot's
modules.json/1_Pooling). We deliberately do NOT pass prompt_name= to
SentenceTransformer and instead prepend query_prefix ourselves: ST's lasttoken
pooling is attention-mask aware, so it pools the true last token under either
padding side, and a raw prefix has none of prompt-string placeholder semantics
to get wrong.

CACHING
-------
Passage embeddings are cached to graphs/embeddings/, keyed by a hash of the
spec's *passage-side* fields PLUS EVERY TEXT (not texts[:50] -- that was a
cache-collision hole: two corpora sharing their first 50 texts and their length
silently shared one index). Query-side fields are deliberately NOT in the key:
they cannot change passage vectors, and including them would force a rebuild
whenever the instruction wording changed.

Device: the encoder is pinned to CUDA when present and loaded ONCE per process.
The on-disk index is one artifact regardless of which device built it, which is
what keeps retrieval comparable across machines; the building device and dtype
land in the sibling .meta.json so a run manifest can state them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMB_CACHE = ROOT / "graphs" / "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

# Bumped whenever the cache-key schema changes, so old files can never be read
# under new semantics. v1 = model + texts[:50] + len (the collision hole).
CACHE_SCHEMA = "emb-v2"

# One loaded model per (name, trust_remote_code, device), for the life of the
# process. Rebuilding it per query was the original bug: one full model load
# per question.
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


def spec(retriever: dict | str | None = None) -> dict:
    """Normalize a retriever to a full spec dict.

    Accepts None (the built-in default), a bare model name (back-compat with
    the pre-config era), or a conf/retriever/*.yaml group.
    """
    if retriever is None:
        retriever = {"name": MODEL_NAME}
    if isinstance(retriever, str):
        retriever = {"name": retriever}
    s = dict(retriever)
    s.setdefault("name", MODEL_NAME)
    s.setdefault("query_prefix", "")
    s.setdefault("passage_prefix", "")
    s.setdefault("passage_template", "{text}")
    s.setdefault("batch_size", 64)
    s.setdefault("trust_remote_code", False)
    return s


def _cache_fields(s: dict) -> dict:
    """The spec fields that CAN change passage vectors. Query-side fields are
    excluded on purpose (see module docstring)."""
    return {
        "schema": CACHE_SCHEMA,
        "name": s["name"],
        "passage_prefix": s["passage_prefix"],
        "trust_remote_code": bool(s["trust_remote_code"]),
    }


def _key(texts: list[str], s: dict) -> Path:
    h = hashlib.sha256()
    h.update(json.dumps(_cache_fields(s), sort_keys=True).encode())
    for t in texts:  # EVERY text -- the [:50] shortcut was a collision hole
        h.update(t.encode("utf-8", "ignore"))
    h.update(str(len(texts)).encode())
    return EMB_CACHE / f"{h.hexdigest()[:24]}.npz"


def _model(s: dict):
    key = f"{s['name']}|{s['trust_remote_code']}|{device()}"
    if key not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS[key] = SentenceTransformer(
            s["name"], device=device(),
            trust_remote_code=bool(s["trust_remote_code"]),
        )
    return _MODELS[key]


def passage_texts(turns, s: dict | None = None) -> list[str]:
    """Turn objects -> the strings that get EMBEDDED (spec.passage_template).

    This is retrieval input, not prompt rendering: eval._render (what the READER
    sees) is deliberately separate and unchanged.
    """
    s = spec(s)
    tmpl = s["passage_template"]
    if tmpl == "{text}":
        return [t.text for t in turns]
    return [
        tmpl.format(
            speaker=t.speaker, text=t.text,
            date_time=t.date_time, dia_id=t.dia_id,
        )
        for t in turns
    ]


def encode(texts: list[str], s: dict | None = None,
           batch_size: int | None = None) -> np.ndarray:
    """L2-normalized embeddings (n, d) for `texts`, on the pinned device.

    (L2 normalize + dot == cosine, which is what every spec here wants. The
    normalize module inside each ST checkpoint would do it anyway; this makes
    it explicit and model-independent.)
    """
    s = spec(s)
    return np.asarray(
        _model(s).encode(  # type: ignore[attr-defined]
            texts,
            normalize_embeddings=True,
            batch_size=batch_size or s["batch_size"],
            show_progress_bar=len(texts) > 200,
        ),
        dtype=np.float32,
    )


def embed(texts: list[str], s: dict | None = None) -> np.ndarray:
    """Passage embeddings (n, d), cached to graphs/embeddings/."""
    s = spec(s)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = _key(texts, s)
    if cpath.exists():
        return np.load(cpath)["emb"]

    emb = encode(texts, s)
    np.savez_compressed(cpath, emb=emb)
    (cpath.with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                **_cache_fields(s),
                "n": len(texts),
                "dim": int(emb.shape[1]),
                "device": device(),
                "dtype": str(emb.dtype),
            },
            indent=2,
        )
    )
    return emb


_QUERY_CACHE: dict[tuple, np.ndarray] = {}


def encode_query(query: str, s: dict | None = None) -> np.ndarray:
    """Query embedding (d,). Cached in-memory and on-disk across k-sweeps."""
    s = spec(s)
    key = (s["name"], s["query_prefix"], bool(s["trust_remote_code"]), query)
    if key in _QUERY_CACHE:
        return _QUERY_CACHE[key]

    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    h.update(f"{s['name']}|{s['query_prefix']}|{bool(s['trust_remote_code'])}".encode())
    h.update(query.encode("utf-8", "ignore"))
    qpath = EMB_CACHE / f"q_{h.hexdigest()[:24]}.npz"
    if qpath.exists():
        vec = np.load(qpath)["vec"]
        _QUERY_CACHE[key] = vec
        return vec

    vec = encode([s["query_prefix"] + query], s)[0]
    np.savez_compressed(qpath, vec=vec)
    _QUERY_CACHE[key] = vec
    return vec


def top_k(query: str, texts: list[str], k: int = 10,
          retriever: dict | str | None = None) -> list[tuple[int, float]]:
    """Return [(index, cosine_similarity), ...] for the top-k most similar texts."""
    if not texts:
        return []
    s = spec(retriever)
    emb = embed(texts, s)
    q = encode_query(query, s)
    sims = emb @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


class Embedder:
    """Wrapper class providing object-oriented access to embedding models."""

    def __init__(self, retriever: dict | str | None = None):
        self.s = spec(retriever)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = encode(texts, self.s)
        return arr.tolist()

