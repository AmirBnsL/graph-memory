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


@dataclass
class AnswerResult:
    answer: str
    context_ids: list[str] = field(default_factory=list)
    prompt_chars: int = 0
    latency_s: float = 0.0


def _render(turns: list[Turn]) -> str:
    return "\n".join(f"[{t.dia_id}] {t.date_time} | {t.speaker}: {t.text}" for t in turns)


def _ask(question: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return llm.complete(prompt, system=SYSTEM, max_tokens=64)


def answer_full(conv: Conversation, qa: QA) -> AnswerResult:
    t0 = time.time()
    ctx = _render(conv.turns)
    out = _ask(qa.question, ctx)
    return AnswerResult(out, [t.dia_id for t in conv.turns], len(ctx), time.time() - t0)


def answer_vector(conv: Conversation, qa: QA, k: int = 10) -> AnswerResult:
    t0 = time.time()
    texts = [t.text for t in conv.turns]
    hits = embed.top_k(qa.question, texts, k=k)
    picked = [conv.turns[i] for i, _ in hits]
    ctx = _render(picked)
    out = _ask(qa.question, ctx)
    return AnswerResult(out, [t.dia_id for t in picked], len(ctx), time.time() - t0)


MODES = {"full": answer_full, "vector": answer_vector}


def answer(conv: Conversation, qa: QA, mode: str = "vector", k: int = 10) -> AnswerResult:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; have {sorted(MODES)}")
    if mode == "vector":
        return answer_vector(conv, qa, k=k)
    return MODES[mode](conv, qa)
