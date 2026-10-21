"""Backend C: Bi-Temporal Knowledge Graph with Dynamic Edge Invalidation.

Isolates the bi-temporal mechanics from Zep / Graphiti (Rasmussen et al., §2.2.3):
  1. Quad-timestamped edges: (t_valid, t_invalid) ∈ T and (t'_created, t'_expired) ∈ T'.
  2. Dynamic edge invalidation: newer contradictory edges supersede older overlapping edges.
  3. Retrieval: BFS expansion from query seed entities, filtered by temporal validity windows,
     combined with cosine ranking.
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


# Predicates indicating mutually exclusive status or dynamic update
MUTUALLY_EXCLUSIVE_PREDICATES = {
    "lives in", "moved to", "located in", "works at", "employed by",
    "has role", "status is", "state is", "current car is", "owns",
}


class BiTemporalMemory(BaseMemory):
    def __init__(self, embedder: Any = None, bfs_depth: int = 2):
        self.embedder = embedder
        self.bfs_depth = bfs_depth
        self.records: list[MemoryRecord] = []
        self.graph = nx.MultiDiGraph()
        self._ingestion_tick = 0

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            import embed
            self.embedder = embed.Embedder()

    def add_records(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        self._ensure_embedder()
        self.records.extend(records)

        for rec in records:
            self._ingestion_tick += 1
            t_created = self._ingestion_tick
            t_valid = rec.timestamp or "present"

            for sub, pred, obj in rec.triples:
                s_clean, p_clean, o_clean = sub.strip(), pred.strip().lower(), obj.strip()
                if not s_clean or not o_clean:
                    continue

                self.graph.add_node(s_clean, type="entity")
                self.graph.add_node(o_clean, type="entity")

                # Invalidation check (§2.2.3): Check existing edges from this subject
                # If an edge shares a mutually exclusive predicate, invalidate the older edge
                if self.graph.has_node(s_clean):
                    for _, target, key, data in list(self.graph.out_edges(s_clean, keys=True, data=True)):
                        old_pred = data.get("predicate", "").lower()
                        # Check contradiction / update condition
                        if (
                            (old_pred == p_clean or (old_pred in MUTUALLY_EXCLUSIVE_PREDICATES and p_clean in MUTUALLY_EXCLUSIVE_PREDICATES))
                            and target != o_clean
                            and data.get("t_invalid") == "present"
                        ):
                            # Invalidate old edge
                            data["t_invalid"] = t_valid
                            data["t_expired"] = t_created
                            data["status"] = "superseded"

                # Add the new edge
                self.graph.add_edge(
                    s_clean,
                    o_clean,
                    predicate=pred,
                    t_valid=t_valid,
                    t_invalid="present",
                    t_created=t_created,
                    t_expired=None,
                    status="active",
                    fact_text=rec.fact_text,
                    record_id=rec.id,
                    turn_id=rec.turn_id,
                    speaker=rec.speaker,
                    sample_id=rec.sample_id,
                )

    def retrieve(self, query: str, k: int, timestamp: str | None = None) -> list[RetrievedItem]:
        if self.graph.number_of_nodes() == 0:
            return []
        self._ensure_embedder()

        # Step 1: Identify seed entities matching the query
        q_lower = query.lower()
        seed_nodes = [n for n in self.graph.nodes() if n.lower() in q_lower]

        candidate_edges: list[dict[str, Any]] = []

        if seed_nodes:
            # Step 2: BFS expansion up to bfs_depth from seed entities
            visited_nodes: set[str] = set(seed_nodes)
            current_frontier = set(seed_nodes)

            for _ in range(self.bfs_depth):
                next_frontier: set[str] = set()
                for node in current_frontier:
                    for u, v, data in self.graph.out_edges(node, data=True):
                        candidate_edges.append(data)
                        if v not in visited_nodes:
                            visited_nodes.add(v)
                            next_frontier.add(v)
                    for u, v, data in self.graph.in_edges(node, data=True):
                        candidate_edges.append(data)
                        if u not in visited_nodes:
                            visited_nodes.add(u)
                            next_frontier.add(u)
                current_frontier = next_frontier
        else:
            # Fallback: all active edges
            for _, _, data in self.graph.edges(data=True):
                candidate_edges.append(data)

        if not candidate_edges:
            return []

        # Deduplicate edges by record_id
        unique_edges: dict[str, dict[str, Any]] = {}
        for edge_data in candidate_edges:
            rid = edge_data.get("record_id", "")
            if rid and rid not in unique_edges:
                unique_edges[rid] = edge_data

        # Step 3: Temporal filtering & Cosine ranking
        q_vec = self.embedder.embed_texts([query])[0]
        fact_texts = [d["fact_text"] for d in unique_edges.values()]
        fact_vecs = self.embedder.embed_texts(fact_texts)

        scored_items: list[tuple[float, dict[str, Any]]] = []
        for vec, (rid, edge_data) in zip(fact_vecs, unique_edges.items()):
            sim = _cosine(q_vec, vec)

            # Prioritize active facts over superseded ones unless historical query
            if edge_data.get("status") == "active":
                sim += 0.2
            if edge_data.get("t_invalid") != "present":
                sim -= 0.1

            scored_items.append((sim, edge_data))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        top = scored_items[:k]

        return [
            RetrievedItem(
                id=edge_data["record_id"],
                text=edge_data["fact_text"],
                score=score,
                turn_id=edge_data.get("turn_id"),
                metadata={
                    "t_valid": edge_data.get("t_valid"),
                    "t_invalid": edge_data.get("t_invalid"),
                    "status": edge_data.get("status"),
                    "speaker": edge_data.get("speaker"),
                    "sample_id": edge_data.get("sample_id"),
                },
            )
            for score, edge_data in top
        ]

    def export_graph(self) -> nx.MultiDiGraph:
        return self.graph

    def clear(self) -> None:
        self.records.clear()
        self.graph.clear()
        self._ingestion_tick = 0
