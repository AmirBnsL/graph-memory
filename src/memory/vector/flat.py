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
        if hasattr(self.embedder, "embed_query"):
            q_vec = self.embedder.embed_query(query)
        else:
            q_vec = self.embedder.embed_texts([query])[0]
        
        scored: list[tuple[float, MemoryRecord]] = []
        for vec, rec in zip(self._vectors, self.records):
            sim = _cosine(q_vec, vec)
            scored.append((sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Turn diversification: max 2 propositions per turn in pass 1, backfill in pass 2
        picked: list[tuple[float, MemoryRecord]] = []
        turn_counts: dict[str, int] = {}
        deferred: list[tuple[float, MemoryRecord]] = []

        for score, rec in scored:
            t_id = str(rec.turn_id)
            if turn_counts.get(t_id, 0) < 2:
                picked.append((score, rec))
                turn_counts[t_id] = turn_counts.get(t_id, 0) + 1
                if len(picked) == k:
                    break
            else:
                deferred.append((score, rec))

        if len(picked) < k:
            for score, rec in deferred:
                picked.append((score, rec))
                if len(picked) == k:
                    break

        return [
            RetrievedItem(
                id=rec.id,
                text=rec.fact_text,
                score=score,
                turn_id=rec.turn_id,
                metadata={"timestamp": rec.timestamp, "speaker": rec.speaker, "sample_id": rec.sample_id},
            )
            for score, rec in picked
        ]

    def export_graph(self) -> nx.Graph | None:
        return None

    def clear(self) -> None:
        self.records.clear()
        self._vectors.clear()
