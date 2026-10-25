"""Backend C: Bi-Temporal Knowledge Graph with Dynamic Edge Invalidation.

Isolates the bi-temporal mechanics from Zep / Graphiti (Rasmussen et al., §2.2.3):
  1. Quad-timestamped edges: (t_valid, t_invalid) ∈ T and (t'_created, t'_expired) ∈ T'.
  2. Dynamic edge invalidation: newer contradictory edges supersede older overlapping edges.
  3. Retrieval: BFS expansion from query seed entities, filtered by temporal validity windows,
     combined with cosine ranking.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any
import networkx as nx

try:
    from graph_memory.pipeline.extract import MemoryRecord
    from graph_memory.memory.base import BaseMemory, RetrievedItem
except ImportError:
    from extract import MemoryRecord
    from memory.base import BaseMemory, RetrievedItem


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _norm(e: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]", " ", e.lower())
    return " ".join(cleaned.split())


STOP_NODES = {
    "user", "assistant", "ai", "speaker", "human", "i", "me", "my",
    "you", "your", "he", "she", "they", "we", "it", "someone", "person",
}

# Grouped predicates indicating mutually exclusive status or dynamic update.
# Invalidation occurs only within the SAME semantic group (Zep/Graphiti §2.2.3).
CONFLICT_PREDICATE_GROUPS = [
    {"lives in", "moved to", "located in", "resides in", "relocated to", "living in"},
    {"works at", "employed by", "worked at", "job at", "works for"},
    {"has role", "role is", "position is", "title is", "promoted to"},
    {"status is", "state is", "current status is"},
    {"current car is", "drives", "owns car", "car is", "bought car"},
]


def _predicates_conflict(p1: str, p2: str) -> bool:
    if p1 == p2:
        return True
    for group in CONFLICT_PREDICATE_GROUPS:
        if p1 in group and p2 in group:
            return True
    return False


HISTORICAL_QUERY_TERMS = {
    "first", "before", "previously", "used to", "originally",
    "earlier", "prior", "initially", "former", "past",
}


class BiTemporalMemory(BaseMemory):
    def __init__(self, embedder: Any = None, bfs_depth: int = 2):
        self.embedder = embedder
        self.bfs_depth = bfs_depth
        self.records: list[MemoryRecord] = []
        self.graph = nx.MultiDiGraph()
        self.entity_embeddings: dict[str, list[float]] = {}
        self._ingestion_tick = 0

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            try:
                from graph_memory.core import embed
            except ImportError:
                import embed
            self.embedder = embed.Embedder()

    def add_records(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        self._ensure_embedder()
        self.records.extend(records)

        # Track added (s, p, o, rec.id) to prevent duplicate edge insertion
        seen_turn_triples: set[tuple[str, str, str, str]] = set()
        new_nodes: set[str] = set()

        for rec in records:
            self._ingestion_tick += 1
            t_created = self._ingestion_tick
            t_valid = rec.timestamp or "present"
            t_id = str(rec.turn_id)
            rec_text_lower = rec.fact_text.lower()

            # Register entities from rec
            for ent in rec.entities:
                en = _norm(ent)
                if en and en not in STOP_NODES:
                    new_nodes.add(en)
                    self.graph.add_node(en, type="entity")

            candidate_triples = []
            for sub, pred, obj in rec.triples:
                s_clean = _norm(sub)
                p_clean = pred.strip().lower()
                o_clean = _norm(obj)
                if not s_clean or not o_clean or s_clean in STOP_NODES or o_clean in STOP_NODES:
                    continue
                if s_clean in rec_text_lower or o_clean in rec_text_lower:
                    candidate_triples.append((sub, pred, obj, s_clean, p_clean, o_clean))

            if not candidate_triples and rec.triples:
                for sub, pred, obj in rec.triples:
                    s_clean = _norm(sub)
                    p_clean = pred.strip().lower()
                    o_clean = _norm(obj)
                    if s_clean and o_clean and s_clean not in STOP_NODES and o_clean not in STOP_NODES:
                        candidate_triples.append((sub, pred, obj, s_clean, p_clean, o_clean))

            for sub, pred, obj, s_clean, p_clean, o_clean in candidate_triples:
                new_nodes.add(s_clean)
                new_nodes.add(o_clean)
                self.graph.add_node(s_clean, type="entity")
                self.graph.add_node(o_clean, type="entity")

                # Deduplicate triples strictly per turn (prevent intra-turn proposition inflation)
                triple_key = (s_clean, p_clean, o_clean, str(rec.turn_id))
                if triple_key in seen_turn_triples:
                    continue
                seen_turn_triples.add(triple_key)

                # Invalidation check (§2.2.3): Check existing edges from this subject
                # If an edge shares a conflicting predicate, invalidate the older edge
                # Require different turn_id so intra-turn propositions do not invalidate each other
                if self.graph.has_node(s_clean):
                    for _, target, key, data in list(self.graph.out_edges(s_clean, keys=True, data=True)):
                        old_pred = data.get("predicate", "").lower()
                        if (
                            str(data.get("turn_id")) != str(rec.turn_id)
                            and _predicates_conflict(old_pred, p_clean)
                            and target != o_clean
                            and data.get("t_invalid") == "present"
                        ):
                            data["t_invalid"] = t_valid
                            data["t_expired"] = t_created
                            data["status"] = "superseded"

                # Add the new edge
                edge_id = f"{rec.sample_id}_{rec.turn_id}_{s_clean}_{p_clean}_{o_clean}"
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
                    edge_id=edge_id,
                    turn_id=rec.turn_id,
                    speaker=rec.speaker,
                    sample_id=rec.sample_id,
                )

        # Precompute embeddings for new nodes to enable semantic query seeding
        nodes_to_embed = [n for n in new_nodes if n not in self.entity_embeddings]
        if nodes_to_embed:
            vecs = self.embedder.embed_texts(nodes_to_embed)
            for n, v in zip(nodes_to_embed, vecs):
                self.entity_embeddings[n] = v

    def retrieve(self, query: str, k: int, timestamp: str | None = None) -> list[RetrievedItem]:
        if self.graph.number_of_nodes() == 0:
            return []
        self._ensure_embedder()

        # Step 1: Semantic + Lexical seed entity selection
        if hasattr(self.embedder, "embed_query"):
            q_vec = self.embedder.embed_query(query)
        else:
            q_vec = self.embedder.embed_texts([query])[0]

        q_lower = query.lower()
        scored_seeds: list[tuple[float, str]] = []

        for node in self.graph.nodes():
            score = 0.0
            if node in self.entity_embeddings:
                score += max(0.0, _cosine(q_vec, self.entity_embeddings[node]))
            if len(node) >= 3 and re.search(r"\b" + re.escape(node) + r"\b", q_lower):
                score += 0.5
            if score > 0.35:
                scored_seeds.append((score, node))

        scored_seeds.sort(key=lambda x: x[0], reverse=True)
        seed_nodes = [n for _, n in scored_seeds[:5]]

        candidate_edges: list[dict[str, Any]] = []
        bfs_edge_rids: set[str] = set()

        if seed_nodes:
            # Step 2: BFS expansion up to bfs_depth from seed entities
            visited_nodes: set[str] = set(seed_nodes)
            current_frontier = set(seed_nodes)

            for _ in range(self.bfs_depth):
                next_frontier: set[str] = set()
                for node in current_frontier:
                    for u, v, data in self.graph.out_edges(node, data=True):
                        candidate_edges.append(data)
                        if data.get("record_id"):
                            bfs_edge_rids.add(data["record_id"])
                        if v not in visited_nodes:
                            visited_nodes.add(v)
                            next_frontier.add(v)
                    for u, v, data in self.graph.in_edges(node, data=True):
                        candidate_edges.append(data)
                        if data.get("record_id"):
                            bfs_edge_rids.add(data["record_id"])
                        if u not in visited_nodes:
                            visited_nodes.add(u)
                            next_frontier.add(u)
                current_frontier = next_frontier

        # Graphiti Hybrid Union (§3.1): always include all active graph edges as candidates
        # so ego networks never starve candidate generation
        for _, _, data in self.graph.edges(data=True):
            candidate_edges.append(data)

        if not candidate_edges:
            return []

        # Deduplicate edges by edge_id (or record_id fallback)
        unique_edges: dict[str, dict[str, Any]] = {}
        for edge_data in candidate_edges:
            ekey = edge_data.get("edge_id") or edge_data.get("record_id", "")
            if ekey and ekey not in unique_edges:
                unique_edges[ekey] = edge_data

        # Step 3: Query-aware temporal filtering & Cosine ranking
        is_historical = any(re.search(r"\b" + w + r"\b", q_lower) for w in HISTORICAL_QUERY_TERMS)

        fact_texts = [d["fact_text"] for d in unique_edges.values()]
        fact_vecs = self.embedder.embed_texts(fact_texts)

        scored_items: list[tuple[float, dict[str, Any]]] = []
        for vec, (rid, edge_data) in zip(fact_vecs, unique_edges.items()):
            sim = _cosine(q_vec, vec)
            if rid in bfs_edge_rids:
                sim += 0.15  # Graph topological proximity boost from BFS (§3.1)

            if not is_historical:
                # Normal queries prefer current active state
                if edge_data.get("status") == "active":
                    sim += 0.2
                if edge_data.get("t_invalid") != "present":
                    sim -= 0.1
            else:
                # Historical queries ("what was first", "where did she live before")
                # give fair weight to superseded historical facts
                if edge_data.get("t_invalid") != "present":
                    sim += 0.15

            scored_items.append((sim, edge_data))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        # Step 4: Turn diversification
        diversified: list[tuple[float, dict[str, Any]]] = []
        turn_counts: dict[str, int] = defaultdict(int)
        remaining: list[tuple[float, dict[str, Any]]] = []

        for sim, edge_data in scored_items:
            t_id = str(edge_data.get("turn_id", ""))
            if turn_counts[t_id] < 2:
                diversified.append((sim, edge_data))
                turn_counts[t_id] += 1
                if len(diversified) >= k:
                    break
            else:
                remaining.append((sim, edge_data))

        if len(diversified) < k:
            for sim, edge_data in remaining:
                diversified.append((sim, edge_data))
                if len(diversified) >= k:
                    break

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
            for score, edge_data in diversified
        ]

    def export_graph(self) -> nx.MultiDiGraph:
        return self.graph

    def clear(self) -> None:
        self.records.clear()
        self.graph.clear()
        self.entity_embeddings.clear()
        self._ingestion_tick = 0
