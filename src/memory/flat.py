"""Backend A: Flat Extracted Facts Memory (The H1 Non-Graph Control).

Dense vector similarity search over atomic extracted propositions.
Zero graph structure, zero relational traversal.
Serves as the baseline to measure the pure contribution of LLM fact extraction.
"""
from __future__ import annotations

import math
from typing import Any
import networkx as nx

from extract import MemoryRecord
from memory.base import BaseMemory, RetrievedItem


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class FlatMemory(BaseMemory):
    def __init__(self, embedder: Any = None):
        self.embedder = embedder
        self.records: list[MemoryRecord] = []
        self._vectors: list[list[float]] = []

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            import embed
            self.embedder = embed.Embedder()

    def add_records(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        self._ensure_embedder()
        self.records.extend(records)
        new_texts = [r.fact_text for r in records]
        new_vecs = self.embedder.embed_texts(new_texts)
        self._vectors.extend(new_vecs)

    def retrieve(self, query: str, k: int, timestamp: str | None = None) -> list[RetrievedItem]:
        if not self.records or not self._vectors:
            return []
        self._ensure_embedder()
        q_vec = self.embedder.embed_texts([query])[0]
        
        scored: list[tuple[float, MemoryRecord]] = []
        for vec, rec in zip(self._vectors, self.records):
            sim = _cosine(q_vec, vec)
            scored.append((sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        return [
            RetrievedItem(
                id=rec.id,
                text=rec.fact_text,
                score=score,
                turn_id=rec.turn_id,
                metadata={"timestamp": rec.timestamp, "speaker": rec.speaker, "sample_id": rec.sample_id},
            )
            for score, rec in top
        ]

    def export_graph(self) -> nx.Graph | None:
        return None

    def clear(self) -> None:
        self.records.clear()
        self._vectors.clear()
