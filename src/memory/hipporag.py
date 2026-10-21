"""Backend B: HippoRAG OpenIE Concept Graph + Personalized PageRank (NeurIPS 2024).

Implements the exact methodology from Jiménez Gutiérrez et al. (§2.2-§2.3):
  1. OpenIE Concept Graph G = (N, E ∪ E') where N are noun phrases/entities,
     E are relation triples, and E' are dense synonymy bridges.
  2. Occurrence Matrix P mapping entity nodes to source dialogue turns/facts.
  3. Node Specificity s_i = 1 / |P_i| weighting initial query restart probabilities.
  4. Personalized PageRank (PPR) spreading activation over the concept graph G.
  5. Passage score projection via p = n'^T P to rank retrieved memories.
"""
from __future__ import annotations

import math
from collections import defaultdict
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


class HippoRAGMemory(BaseMemory):
    def __init__(
        self,
        embedder: Any = None,
        synonym_threshold: float = 0.85,
        ppr_damping: float = 0.85,
        flatten_structure: bool = False,
    ):
        self.embedder = embedder
        self.synonym_threshold = synonym_threshold
        self.ppr_damping = ppr_damping
        self.flatten_structure = flatten_structure

        self.records: list[MemoryRecord] = []
        self.graph = nx.Graph()
        # occurrence_map: entity -> set of record indices
        self.occurrence_map: dict[str, set[int]] = defaultdict(set)
        self.entity_embeddings: dict[str, list[float]] = {}

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            import embed
            self.embedder = embed.Embedder()

    def add_records(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        self._ensure_embedder()
        start_idx = len(self.records)
        self.records.extend(records)

        new_entities: set[str] = set()
        for idx_offset, rec in enumerate(records):
            rec_idx = start_idx + idx_offset
            for ent in rec.entities:
                ent_clean = ent.strip()
                if ent_clean:
                    new_entities.add(ent_clean)
                    self.occurrence_map[ent_clean].add(rec_idx)
                    self.graph.add_node(ent_clean, type="entity")

            if not self.flatten_structure:
                # Add relational triples E
                for sub, pred, obj in rec.triples:
                    s_clean, o_clean = sub.strip(), obj.strip()
                    if s_clean and o_clean:
                        self.graph.add_node(s_clean, type="entity")
                        self.graph.add_node(o_clean, type="entity")
                        self.graph.add_edge(s_clean, o_clean, relation=pred, type="relation", weight=1.0)

        # Compute embeddings for new entities to add synonymy edges E'
        ents_to_embed = [e for e in new_entities if e not in self.entity_embeddings]
        if ents_to_embed:
            vecs = self.embedder.embed_texts(ents_to_embed)
            for e, v in zip(ents_to_embed, vecs):
                self.entity_embeddings[e] = v

        if not self.flatten_structure and len(self.entity_embeddings) > 1:
            all_ents = list(self.entity_embeddings.keys())
            # Compare newly added entities against existing entity vocabulary
            for new_ent in ents_to_embed:
                v_new = self.entity_embeddings[new_ent]
                for other_ent in all_ents:
                    if new_ent != other_ent and not self.graph.has_edge(new_ent, other_ent):
                        sim = _cosine(v_new, self.entity_embeddings[other_ent])
                        if sim >= self.synonym_threshold:
                            self.graph.add_edge(new_ent, other_ent, relation="synonym", type="synonym", weight=float(sim))

    def retrieve(self, query: str, k: int, timestamp: str | None = None) -> list[RetrievedItem]:
        if not self.records or self.graph.number_of_nodes() == 0:
            return []
        self._ensure_embedder()

        # Step 1: Query entity matching & node specificity weighting
        # Match query against entity vocabulary
        all_nodes = list(self.graph.nodes())
        q_vec = self.embedder.embed_texts([query])[0]
        node_scores: list[tuple[float, str]] = []

        q_lower = query.lower()
        for node in all_nodes:
            # Lexical exact match or semantic similarity
            score = 0.0
            if node.lower() in q_lower:
                score += 1.0
            if node in self.entity_embeddings:
                sim = _cosine(q_vec, self.entity_embeddings[node])
                score += max(0.0, sim)
            if score > 0.3:
                node_scores.append((score, node))

        if not node_scores:
            # Fallback to top cosine matching nodes
            for node in all_nodes:
                if node in self.entity_embeddings:
                    sim = _cosine(q_vec, self.entity_embeddings[node])
                    node_scores.append((sim, node))
            node_scores.sort(key=lambda x: x[0], reverse=True)
            node_scores = node_scores[:10]

        # Calculate restart personalization vector with node specificity: s_i = 1 / |P_i|
        personalization: dict[str, float] = {}
        for score, node in node_scores:
            passages_count = len(self.occurrence_map.get(node, set())) or 1
            node_specificity = 1.0 / passages_count
            personalization[node] = score * node_specificity

        total_weight = sum(personalization.values())
        if total_weight > 0:
            for n in personalization:
                personalization[n] /= total_weight
        else:
            personalization = {n: 1.0 / len(all_nodes) for n in all_nodes}

        # Step 2: Personalized PageRank on G = (N, E ∪ E')
        try:
            ppr_probs = nx.pagerank(
                self.graph,
                alpha=self.ppr_damping,
                personalization=personalization,
                max_iter=100,
                weight="weight",
            )
        except Exception:
            # In case of disconnected or single-node edge case
            ppr_probs = personalization

        # Step 3: Passage score projection: p = n'^T P
        record_scores: list[tuple[float, MemoryRecord]] = []
        for idx, rec in enumerate(self.records):
            score = 0.0
            for ent in rec.entities:
                score += ppr_probs.get(ent.strip(), 0.0)
            record_scores.append((score, rec))

        record_scores.sort(key=lambda x: x[0], reverse=True)
        top = record_scores[:k]

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

    def export_graph(self) -> nx.Graph:
        return self.graph

    def clear(self) -> None:
        self.records.clear()
        self.graph.clear()
        self.occurrence_map.clear()
        self.entity_embeddings.clear()
