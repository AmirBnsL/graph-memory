"""Retrieval modes + answering.

The harness keeps RETRIEVAL as the single variable. H1 swaps *structure* while
holding this fixed; H3 measures the graph that this retrieval walks over.

Modes
    full    stuff the entire conversation into the prompt (no retrieval)
    vector  top-k turns by embedding similarity (all-MiniLM-L6-v2)

Planned (only after Week 2 builds a graph):
    graph   turns selected by graph traversal (PPR / BFS over edges)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import embed
import llm
from load import Conversation, QA, Turn

SYSTEM = (
    "You answer questions about a long conversation between two people. "
    "Use ONLY the provided context. Answer concisely, with the shortest correct "
    "answer (a name, date, or short phrase). If the context does not contain the "
    "answer, reply exactly: No information available."
)

# Kept as a named constant so a run manifest can hash the exact template used.
PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"


@dataclass
class AnswerResult:
    answer: str
    context_ids: list[str] = field(default_factory=list)
    prompt_chars: int = 0
    latency_s: float = 0.0


def _render(turns: list[Turn]) -> str:
    return "\n".join(f"[{t.dia_id}] {t.date_time} | {t.speaker}: {t.text}" for t in turns)


def _ask(question: str, context: str, reader: dict | None = None) -> str:
    r = reader or {}
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return llm.complete(prompt, system=SYSTEM,
                        max_tokens=int(r.get("max_tokens", 64)),
                        temperature=float(r.get("temperature", 0.0)),
                        overrides=r)


def answer_full(conv: Conversation, qa: QA, k: int = 10,
                reader: dict | None = None,
                retriever: dict | str | None = None) -> AnswerResult:
    """Stuff the whole conversation in. NOT runnable until num_ctx is explicit
    (provenance/serving.json -> effective_context_tokens): a whole conversation
    is ~15k tokens and would be truncated from the front. Kept for the H1
    upper-bound arm only; k and retriever are accepted and ignored so the
    mode registry stays uniform."""
    t0 = time.time()
    ctx = _render(conv.turns)
    out = _ask(qa.question, ctx, reader)
    return AnswerResult(out, [t.dia_id for t in conv.turns], len(ctx), time.time() - t0)


def answer_vector(conv: Conversation, qa: QA, k: int = 10,
                  reader: dict | None = None,
                  retriever: dict | str | None = None) -> AnswerResult:
    t0 = time.time()
    s = embed.spec(retriever)
    texts = embed.passage_texts(conv.turns, s)
    hits = embed.top_k(qa.question, texts, k=k, retriever=s)
    picked = [conv.turns[i] for i, _ in hits]
    ctx = _render(picked)
    out = _ask(qa.question, ctx, reader)
    return AnswerResult(out, [t.dia_id for t in picked], len(ctx), time.time() - t0)


_CONV_MEMORY_CACHE: dict[tuple[str, str, bool, float], Any] = {}


def get_cached_memories() -> dict[tuple[str, str, bool, float], Any]:
    return dict(_CONV_MEMORY_CACHE)


def clear_memory_cache() -> None:
    _CONV_MEMORY_CACHE.clear()



def answer_memory(conv: Conversation, qa: QA, k: int = 10,
                  reader: dict | None = None,
                  retriever: dict | str | None = None,
                  mode_cfg: dict | None = None) -> AnswerResult:
    """Answer using one of the pluggable memory backends (H1 decomposition)."""
    import extract
    from memory import FlatMemory, HippoRAGMemory, BiTemporalMemory

    t0 = time.time()
    mcfg = mode_cfg or {}
    backend = mcfg.get("backend", "flat")
    flatten = bool(mcfg.get("flatten_structure", False))
    syn_thresh = float(mcfg.get("synonym_threshold", 0.85))

    s = embed.spec(retriever)
    cache_key = (conv.sample_id, backend, flatten, syn_thresh, s["name"], s.get("query_prefix", ""))
    if cache_key in _CONV_MEMORY_CACHE:
        mem = _CONV_MEMORY_CACHE[cache_key]
    else:
        emb = embed.Embedder(s)
        if backend == "flat":
            mem = FlatMemory(embedder=emb)
        elif backend == "hipporag":
            mem = HippoRAGMemory(embedder=emb, synonym_threshold=syn_thresh, flatten_structure=flatten)
        elif backend in ("temporal", "temporal_bteg"):
            mem = BiTemporalMemory(embedder=emb)
        else:
            raise ValueError(f"unknown memory backend {backend!r}; choose flat | hipporag | temporal_bteg")

        records = extract.extract_conversation(conv, overrides=reader)
        mem.add_records(records)
        _CONV_MEMORY_CACHE[cache_key] = mem

    query_ts = getattr(qa, "timestamp", None) or (conv.turns[-1].date_time if conv.turns else None)
    retrieved = mem.retrieve(query=qa.question, k=k, timestamp=query_ts)
    ctx = "\n".join(f"[{item.id}] {item.text}" for item in retrieved)
    context_ids = [str(item.turn_id) for item in retrieved if item.turn_id is not None]

    out = _ask(qa.question, ctx, reader)
    return AnswerResult(out, context_ids, len(ctx), time.time() - t0)


MODES = {"full": answer_full, "vector": answer_vector, "memory": answer_memory}


def answer(conv: Conversation, qa: QA, mode: str = "vector", k: int = 10,
           reader: dict | None = None,
           retriever: dict | str | None = None,
           mode_cfg: dict | None = None) -> AnswerResult:
    fn = MODES.get(mode)
    if fn is None:
        raise ValueError(f"unknown mode {mode!r}; have {sorted(MODES)}")
    if mode == "memory":
        return fn(conv, qa, k=k, reader=reader, retriever=retriever, mode_cfg=mode_cfg)
    return fn(conv, qa, k=k, reader=reader, retriever=retriever)
