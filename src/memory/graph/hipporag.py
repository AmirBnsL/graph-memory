"""Backend B: HippoRAG OpenIE Concept Graph + Personalized PageRank (NeurIPS 2024).

Implements the methodology from Jiménez Gutiérrez et al. (§2.2-§2.3):
  1. OpenIE Concept Graph G = (N, E ∪ E') where N are noun phrases/entities,
     E are relation triples, and E' are dense synonymy bridges.
  2. Occurrence Matrix P mapping entity nodes to source dialogue turns/facts.
  3. Node Specificity s_i = 1 / |P_i| weighting initial query restart probabilities.
  4. Personalized PageRank (PPR) spreading activation over the concept graph G.
  5. Passage score projection via p = n'^T P to rank retrieved memories.
"""
from __future__ import annotations

import math
import re
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


def _norm(e: str) -> str:
    return e.strip().lower()


STOP_NODES = {
    "user", "assistant", "ai", "speaker", "human", "i", "me", "my",
    "you", "your", "he", "she", "they", "we", "it", "someone", "person",
}


class HippoRAGMemory(BaseMemory):
    def __init__(
        self,
        embedder: Any = None,
        synonym_threshold: float = 0.85,
        ppr_damping: float = 0.5,
        flatten_structure: bool = False,
    ):
        self.embedder = embedder
        self.synonym_threshold = synonym_threshold
        self.ppr_damping = ppr_damping
        self.flatten_structure = flatten_structure

        self.records: list[MemoryRecord] = []
        self._record_vectors: list[list[float]] = []
        self.graph = nx.Graph()
        # occurrence_map: entity -> set of distinct turn IDs
        self.occurrence_map: dict[str, set[str]] = defaultdict(set)
        self.entity_embeddings: dict[str, list[float]] = {}

    def _ensure_embedder(self) -> None:
        if self.embedder is None:
            import embed
            self.embedder = embed.Embedder()

    def add_records(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        self._ensure_embedder()
        self.records.extend(records)

        # Cache fact vectors for proposition-level tie breaking
        fact_texts = [r.fact_text for r in records]
        new_r_vecs = self.embedder.embed_texts(fact_texts)
        self._record_vectors.extend(new_r_vecs)

        new_entities: set[str] = set()
        for rec in records:
            t_id = str(rec.turn_id)
            # Collect and normalize all candidate entities from entities and triples
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

            for en in rec_ents:
                new_entities.add(en)
                self.occurrence_map[en].add(t_id)
                self.graph.add_node(en, type="entity")

            if not self.flatten_structure:
                # Add relational triples E between non-stop nodes
                for sub, pred, obj in rec.triples:
                    s_norm, o_norm = _norm(sub), _norm(obj)
                    if (
                        s_norm
                        and o_norm
                        and s_norm not in STOP_NODES
                        and o_norm not in STOP_NODES
                        and s_norm != o_norm
                    ):
                        self.graph.add_edge(s_norm, o_norm, relation=pred.strip(), type="relation", weight=1.0)

        # Compute embeddings for new entities to add synonymy edges E'
        ents_to_embed = [e for e in new_entities if e not in self.entity_embeddings]
        if ents_to_embed:
            vecs = self.embedder.embed_texts(ents_to_embed)
            for e, v in zip(ents_to_embed, vecs):
                self.entity_embeddings[e] = v

        if not self.flatten_structure and len(self.entity_embeddings) > 1:
            all_ents = list(self.entity_embeddings.keys())
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

        # Step 1: Query entity matching & node specificity weighting (HippoRAG §2.3)
        all_nodes = list(self.graph.nodes())
        if hasattr(self.embedder, "embed_query"):
            q_vec = self.embedder.embed_query(query)
        else:
            q_vec = self.embedder.embed_texts([query])[0]

        q_lower = query.lower()
        node_scores: list[tuple[float, str]] = []

        for node in all_nodes:
            sim = 0.0
            if node in self.entity_embeddings:
                sim = max(0.0, _cosine(q_vec, self.entity_embeddings[node]))
            # Word-boundary exact match bonus
            if len(node) >= 3 and re.search(r"\b" + re.escape(node) + r"\b", q_lower):
                sim += 0.5
            if sim > 0.35:
                node_scores.append((sim, node))

        # Sort and take top-5 seed nodes (HippoRAG §2.3)
        node_scores.sort(key=lambda x: x[0], reverse=True)
        top_seeds = node_scores[:5]

        if not top_seeds:
            # Fallback: top cosine matching nodes
            for node in all_nodes:
                if node in self.entity_embeddings:
                    sim = _cosine(q_vec, self.entity_embeddings[node])
                    node_scores.append((sim, node))
            node_scores.sort(key=lambda x: x[0], reverse=True)
            top_seeds = node_scores[:5]

        # Calculate restart personalization vector with node specificity: s_i = 1 / |P_i|
        personalization: dict[str, float] = {}
        for score, node in top_seeds:
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
                dangling=personalization,
                max_iter=100,
                weight="weight",
            )
        except Exception:
            ppr_probs = personalization

        # Step 3: Passage score projection: p = n'^T P
        record_scores: list[tuple[float, MemoryRecord]] = []
        for idx, rec in enumerate(self.records):
            # Collect normalized entities belonging to this record
            rec_text_lower = rec.fact_text.lower()
            matched_ents: set[str] = set()
            for ent in rec.entities:
                en = _norm(ent)
                if en in self.graph and (en in rec_text_lower or len(rec.entities) <= 3):
                    matched_ents.add(en)
            for sub, _, obj in rec.triples:
                s_norm, o_norm = _norm(sub), _norm(obj)
                if s_norm in self.graph and s_norm in rec_text_lower:
                    matched_ents.add(s_norm)
                if o_norm in self.graph and o_norm in rec_text_lower:
                    matched_ents.add(o_norm)

            ppr_sum = 0.0
            for en in matched_ents:
                specificity = 1.0 / (len(self.occurrence_map.get(en, set())) or 1)
                ppr_sum += ppr_probs.get(en, 0.0) * specificity

            # Proposition semantic similarity tie-breaker
            cos_prop = 0.0
            if idx < len(self._record_vectors):
                cos_prop = max(0.0, _cosine(q_vec, self._record_vectors[idx]))

            final_score = ppr_sum + 0.25 * cos_prop
            record_scores.append((final_score, rec))

        record_scores.sort(key=lambda x: x[0], reverse=True)

        # Step 4: Turn diversification to prevent a single turn from crowding out all k slots
        diversified_top: list[tuple[float, MemoryRecord]] = []
        turn_counts: dict[str, int] = defaultdict(int)
        remaining: list[tuple[float, MemoryRecord]] = []

        # Pass 1: Allow up to 2 propositions per turn
        for score, rec in record_scores:
            t_id = str(rec.turn_id)
            if turn_counts[t_id] < 2:
                diversified_top.append((score, rec))
                turn_counts[t_id] += 1
                if len(diversified_top) >= k:
                    break
            else:
                remaining.append((score, rec))

        # Pass 2: Fill remaining slots if needed
        if len(diversified_top) < k:
            for score, rec in remaining:
                diversified_top.append((score, rec))
                if len(diversified_top) >= k:
                    break

        return [
            RetrievedItem(
                id=rec.id,
                text=rec.fact_text,
                score=score,
                turn_id=rec.turn_id,
                metadata={"timestamp": rec.timestamp, "speaker": rec.speaker, "sample_id": rec.sample_id},
            )
            for score, rec in diversified_top
        ]

    def export_graph(self) -> nx.Graph:
        return self.graph

    def clear(self) -> None:
        self.records.clear()
        self._record_vectors.clear()
        self.graph.clear()
        self.occurrence_map.clear()
        self.entity_embeddings.clear()
