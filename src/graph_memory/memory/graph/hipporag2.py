"""Bipartite Concept Graph memory backend (HippoRAG 2).

Integrates dialogue turns and extracted entities as co-equal vertices in a bipartite
concept graph traversed via bi-directional Personalized PageRank.

Graph Schema:
  - Turn vertices: node id 'turn:<turn_id>', attribute type='turn', turn_id=turn_id, text=turn_text.
  - Entity vertices: node id 'entity:<norm_entity>', attribute type='entity', name=norm_entity.
  - Bipartite occurrence edges: between turn and entity nodes with weight 1.0.
  - Relational edges: between entities from relation triples with weight 1.0.
  - Synonym edges: between entities with embedding cosine similarity >= synonym_threshold.

Personalized PageRank Traversal:
  - Seed entities matched from query (or cosine similarity >= seed_threshold, default 0.80).
  - Restart mass distributed across seed entity nodes (specificity-weighted: s_i = 1 / |turns(i)|).
  - Transition split:
      * Entity vertex: 50% distributed equally to incident turns, 50% to entity neighbors (100% if one empty).
      * Turn vertex: 100% distributed equally to constituent entities.
  - Damping factor alpha = 0.5.
  - Candidate turns ranked directly by final stationary probability.
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


class HippoRAG2Memory(BaseMemory):
    """Bipartite Concept Graph memory backend with bi-directional PPR."""

    def __init__(
        self,
        embedder: Any = None,
        synonym_threshold: float = 0.85,
        ppr_damping: float = 0.5,
        seed_threshold: float = 0.80,
    ):
        self.embedder = embedder
        self.synonym_threshold = synonym_threshold
        self.ppr_damping = ppr_damping
        self.seed_threshold = seed_threshold

        self.records: list[MemoryRecord] = []
        self.graph: nx.Graph = nx.Graph()
        self.entity_embeddings: dict[str, list[float]] = {}
        self.turn_texts: dict[str, list[str]] = defaultdict(list)
        self.turn_metadata: dict[str, dict[str, Any]] = {}

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

        new_entities: set[str] = set()
        for rec in records:
            tid = rec.turn_id
            t_key = str(tid)
            t_node = f"turn:{tid}"

            # Accumulate proposition text for the turn
            if rec.fact_text and rec.fact_text not in self.turn_texts[t_key]:
                self.turn_texts[t_key].append(rec.fact_text)
            turn_text = " ".join(self.turn_texts[t_key])

            if t_node not in self.graph:
                self.graph.add_node(
                    t_node,
                    type="turn",
                    turn_id=tid,
                    text=turn_text,
                    speaker=rec.speaker,
                    timestamp=rec.timestamp,
                )
                self.turn_metadata[t_key] = {
                    "speaker": rec.speaker,
                    "timestamp": rec.timestamp,
                    "sample_id": rec.sample_id,
                }
            else:
                self.graph.nodes[t_node]["text"] = turn_text
                if not self.graph.nodes[t_node].get("speaker") and rec.speaker:
                    self.graph.nodes[t_node]["speaker"] = rec.speaker
                    self.turn_metadata[t_key]["speaker"] = rec.speaker
                if not self.graph.nodes[t_node].get("timestamp") and rec.timestamp:
                    self.graph.nodes[t_node]["timestamp"] = rec.timestamp
                    self.turn_metadata[t_key]["timestamp"] = rec.timestamp

            # Extract normalized entities from entities and relation triples
            rec_ents: set[str] = set()
            for ent in rec.entities:
                en = _norm(ent)
                if en and en not in STOP_NODES:
                    rec_ents.add(en)

            for sub, pred, obj in rec.triples:
                s_norm, o_norm = _norm(sub), _norm(obj)
                if s_norm and s_norm not in STOP_NODES:
                    rec_ents.add(s_norm)
                if o_norm and o_norm not in STOP_NODES:
                    rec_ents.add(o_norm)

            # Add entity vertices and bipartite occurrence edges
            for en in rec_ents:
                new_entities.add(en)
                e_node = f"entity:{en}"
                if e_node not in self.graph:
                    self.graph.add_node(e_node, type="entity", name=en)
                self.graph.add_edge(t_node, e_node, weight=1.0, type="occurrence")

            # Add entity-entity relational edges
            for sub, pred, obj in rec.triples:
                s_norm, o_norm = _norm(sub), _norm(obj)
                if (
                    s_norm
                    and o_norm
                    and s_norm not in STOP_NODES
                    and o_norm not in STOP_NODES
                    and s_norm != o_norm
                ):
                    s_node = f"entity:{s_norm}"
                    o_node = f"entity:{o_norm}"
                    if s_node not in self.graph:
                        self.graph.add_node(s_node, type="entity", name=s_norm)
                    if o_node not in self.graph:
                        self.graph.add_node(o_node, type="entity", name=o_norm)
                    self.graph.add_edge(s_node, o_node, relation=pred.strip(), type="relation", weight=1.0)

        # Compute embeddings for new entity vertices
        ents_to_embed = [e for e in new_entities if e not in self.entity_embeddings]
        if ents_to_embed:
            vecs = self.embedder.embed_texts(ents_to_embed)
            for e, v in zip(ents_to_embed, vecs):
                self.entity_embeddings[e] = v

        # Add entity-entity synonym edges based on cosine similarity
        if len(self.entity_embeddings) > 1:
            all_ents = list(self.entity_embeddings.keys())
            for new_ent in ents_to_embed:
                v_new = self.entity_embeddings[new_ent]
                u_node = f"entity:{new_ent}"
                for other_ent in all_ents:
                    if new_ent != other_ent:
                        v_node = f"entity:{other_ent}"
                        if not self.graph.has_edge(u_node, v_node):
                            sim = _cosine(v_new, self.entity_embeddings[other_ent])
                            if sim >= self.synonym_threshold:
                                self.graph.add_edge(
                                    u_node,
                                    v_node,
                                    relation="synonym",
                                    type="synonym",
                                    weight=float(sim),
                                )

    def _build_transition_matrix(self) -> dict[str, dict[str, float]]:
        """Construct transition probabilities according to HippoRAG 2 bipartite rules.

        - For entity node u: 50% mass to incident turn nodes, 50% to entity neighbors (100% if one empty).
        - For turn node v: 100% mass distributed equally to constituent entity nodes.
        """
        transitions: dict[str, dict[str, float]] = {}
        for u in self.graph.nodes():
            node_type = self.graph.nodes[u].get("type")
            neighbors = list(self.graph.neighbors(u))
            if not neighbors:
                transitions[u] = {}
                continue

            if node_type == "turn":
                ent_nbrs = [v for v in neighbors if self.graph.nodes[v].get("type") == "entity"]
                targets = ent_nbrs if ent_nbrs else neighbors
                p = 1.0 / len(targets)
                transitions[u] = {v: p for v in targets}

            elif node_type == "entity":
                turn_nbrs = [v for v in neighbors if self.graph.nodes[v].get("type") == "turn"]
                ent_nbrs = [v for v in neighbors if self.graph.nodes[v].get("type") == "entity"]

                t_dict: dict[str, float] = {}
                if turn_nbrs and ent_nbrs:
                    p_turn = 0.5 / len(turn_nbrs)
                    p_ent = 0.5 / len(ent_nbrs)
                    for v in turn_nbrs:
                        t_dict[v] = p_turn
                    for v in ent_nbrs:
                        t_dict[v] = p_ent
                elif turn_nbrs:
                    p_turn = 1.0 / len(turn_nbrs)
                    for v in turn_nbrs:
                        t_dict[v] = p_turn
                elif ent_nbrs:
                    p_ent = 1.0 / len(ent_nbrs)
                    for v in ent_nbrs:
                        t_dict[v] = p_ent
                transitions[u] = t_dict

            else:
                p = 1.0 / len(neighbors)
                transitions[u] = {v: p for v in neighbors}

        return transitions

    def _power_iteration(
        self,
        p0: dict[str, float],
        transitions: dict[str, dict[str, float]],
        alpha: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """Compute stationary distribution via power iteration with damping factor alpha."""
        nodes = list(self.graph.nodes())
        if not nodes:
            return {}

        p = {u: p0.get(u, 0.0) for u in nodes}
        total_p0 = sum(p.values())
        if total_p0 > 0:
            for u in nodes:
                p[u] /= total_p0
        else:
            p = {u: 1.0 / len(nodes) for u in nodes}
        restart = dict(p)

        for _ in range(max_iter):
            p_next = {u: 0.0 for u in nodes}
            dangling_mass = 0.0
            for u in nodes:
                prob_u = p[u]
                if prob_u == 0.0:
                    continue
                t_u = transitions.get(u, {})
                if not t_u:
                    dangling_mass += prob_u
                else:
                    for v, w in t_u.items():
                        p_next[v] += prob_u * w

            teleport_mass = (1.0 - alpha) + alpha * dangling_mass
            for u in nodes:
                p_next[u] = alpha * p_next[u] + teleport_mass * restart[u]

            diff = sum(abs(p_next[u] - p[u]) for u in nodes)
            p = p_next
            if diff < tol:
                break

        return p

    def retrieve(self, query: str, k: int, timestamp: str | None = None) -> list[RetrievedItem]:
        if not self.records or self.graph.number_of_nodes() == 0:
            return []
        self._ensure_embedder()

        turn_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "turn"]
        if not turn_nodes:
            return []

        entity_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]

        if not entity_nodes:
            p0 = {t: 1.0 / len(turn_nodes) for t in turn_nodes}
        else:
            # Query embedding
            if hasattr(self.embedder, "embed_query"):
                q_vec = self.embedder.embed_query(query)
            else:
                q_vec = self.embedder.embed_texts([query])[0]

            q_lower = query.lower()
            seed_candidates: list[tuple[float, str]] = []

            for u in entity_nodes:
                en_name = self.graph.nodes[u].get("name", u.split(":", 1)[-1])
                sim = 0.0
                if en_name in self.entity_embeddings:
                    sim = max(0.0, _cosine(q_vec, self.entity_embeddings[en_name]))

                # Exact word-boundary match check
                exact_match = len(en_name) >= 3 and bool(
                    re.search(r"\b" + re.escape(en_name) + r"\b", q_lower)
                )
                if exact_match:
                    sim = max(sim, 1.0)

                if exact_match or sim >= self.seed_threshold:
                    seed_candidates.append((sim, u))

            if not seed_candidates:
                # Fallback to top cosine matching entities
                for u in entity_nodes:
                    en_name = self.graph.nodes[u].get("name", u.split(":", 1)[-1])
                    sim = 0.0
                    if en_name in self.entity_embeddings:
                        sim = max(0.0, _cosine(q_vec, self.entity_embeddings[en_name]))
                    seed_candidates.append((sim, u))
                seed_candidates.sort(key=lambda x: x[0], reverse=True)
                seed_candidates = seed_candidates[:5]

            # Calculate restart personalization vector with specificity weighting
            p0 = {}
            for score, u in seed_candidates:
                incident_turns = sum(
                    1 for v in self.graph.neighbors(u) if self.graph.nodes[v].get("type") == "turn"
                )
                specificity = 1.0 / max(1, incident_turns)
                p0[u] = score * specificity

            total_w = sum(p0.values())
            if total_w > 0:
                for u in p0:
                    p0[u] /= total_w
            else:
                for u in p0:
                    p0[u] = 1.0 / len(p0)

        # Personalized PageRank traversal
        transitions = self._build_transition_matrix()
        stationary_probs = self._power_iteration(
            p0, transitions, alpha=self.ppr_damping
        )

        # Candidate turn scoring: rank directly by final stationary probability
        turn_scores = [(stationary_probs.get(t, 0.0), t) for t in turn_nodes]
        turn_scores.sort(key=lambda x: x[0], reverse=True)

        items: list[RetrievedItem] = []
        for score, t_node in turn_scores[:k]:
            attrs = self.graph.nodes[t_node]
            tid = attrs.get("turn_id")
            text = attrs.get("text", "")
            meta = dict(self.turn_metadata.get(str(tid), {}))
            meta["timestamp"] = attrs.get("timestamp") or meta.get("timestamp")
            meta["speaker"] = attrs.get("speaker") or meta.get("speaker")
            items.append(
                RetrievedItem(
                    id=t_node,
                    text=text,
                    score=score,
                    turn_id=tid,
                    metadata=meta,
                )
            )

        return items

    def export_graph(self) -> nx.Graph:
        """Export the simple undirected projection of the internal NetworkX graph."""
        return self.graph

    def clear(self) -> None:
        """Reset internal memory state."""
        self.records.clear()
        self.graph.clear()
        self.entity_embeddings.clear()
        self.turn_texts.clear()
        self.turn_metadata.clear()
