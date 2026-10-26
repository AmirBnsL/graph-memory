"""Backend: Dual-Level Graph (LightRAG) Memory.

Implements the Dual-Level Graph memory architecture from Guo et al. (2024):
  1. Level 1: Entity vertices (node id entity:<normalized_entity>) with relational
     edges from OpenIE triples and synonymy bridges between semantically close entities.
  2. Level 2: Thematic cluster vertices (node id theme:<cluster_id>) derived via
     deterministic agglomerative clustering of entity embeddings, linked bidirectionally
     to constituent member entities.
  3. Dual-Level Traversal & Retrieval:
     - Low-level matching: query entities matched against entity vertices via embedding
       cosine similarity (threshold >= 0.80).
     - High-level matching: query embedding matched against thematic cluster centroids.
       Top-2 thematic clusters receive restart activation.
     - Dual-level Personalized PageRank: restart vector allocates 60% mass to low-level
       query entities and 40% mass to matched high-level thematic clusters (damping alpha = 0.5).
     - Candidate turn scoring: maps stationary activation of constituent entities to source
       dialogue turns, returning ranked RetrievedItem instances.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

try:
    from graph_memory.pipeline.extract import MemoryRecord
    from graph_memory.memory.base import BaseMemory, RetrievedItem
    from graph_memory.memory.graph.hipporag2 import _GLOBAL_ENTITY_EMB_CACHE, _save_entity_cache
except ImportError:
    from extract import MemoryRecord
    from memory.base import BaseMemory, RetrievedItem
    try:
        from memory.graph.hipporag2 import _GLOBAL_ENTITY_EMB_CACHE, _save_entity_cache
    except ImportError:
        _GLOBAL_ENTITY_EMB_CACHE = {}
        def _save_entity_cache(): pass


def _cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
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


class LightRAGMemory(BaseMemory):
    """Dual-Level Graph (LightRAG) memory engine."""

    def __init__(
        self,
        embedder: Any = None,
        synonym_threshold: float = 0.85,
        ppr_damping: float = 0.5,
        low_level_threshold: float = 0.80,
        cluster_distance_threshold: float = 0.65,
        max_clusters: int = 8,
    ):
        self.embedder = embedder
        self.synonym_threshold = synonym_threshold
        self.ppr_damping = ppr_damping
        self.low_level_threshold = low_level_threshold
        self.cluster_distance_threshold = cluster_distance_threshold
        self.max_clusters = max_clusters

        self.records: list[MemoryRecord] = []
        self.graph: nx.Graph = nx.Graph()

        # occurrence_map: entity name or entity node id -> set of turn IDs
        self.occurrence_map: dict[str, set[Any]] = defaultdict(set)
        # turn_records: turn_id (str) -> list of MemoryRecord instances
        self.turn_records: dict[str, list[MemoryRecord]] = defaultdict(list)
        # turn_entities: turn_id (str) -> set of entity node IDs (e.g. "entity:alice")
        self.turn_entities: dict[str, set[str]] = defaultdict(set)
        # entity_embeddings: normalized entity name -> embedding vector
        self.entity_embeddings: dict[str, list[float]] = {}
        # cluster metadata
        self.cluster_centroids: dict[str, list[float]] = {}
        self.cluster_members: dict[str, list[str]] = {}

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
            orig_tid = rec.turn_id
            t_id = str(orig_tid)
            self.turn_records[t_id].append(rec)

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
                entity_node = f"entity:{en}"
                self.occurrence_map[en].add(orig_tid)
                self.occurrence_map[en].add(t_id)
                self.occurrence_map[entity_node].add(orig_tid)
                self.occurrence_map[entity_node].add(t_id)
                self.turn_entities[t_id].add(entity_node)
                self.graph.add_node(entity_node, type="entity", name=en)

            # Add Level 1 relational edges from triples
            for sub, pred, obj in rec.triples:
                s_norm, o_norm = _norm(sub), _norm(obj)
                if (
                    s_norm
                    and o_norm
                    and s_norm not in STOP_NODES
                    and o_norm not in STOP_NODES
                    and s_norm != o_norm
                ):
                    u = f"entity:{s_norm}"
                    v = f"entity:{o_norm}"
                    self.graph.add_edge(u, v, relation=pred.strip(), type="relation", weight=1.0)

        # Compute embeddings for any newly encountered entities
        ents_to_embed = [e for e in new_entities if e not in self.entity_embeddings]
        if ents_to_embed:
            emb_key = str(getattr(self.embedder, "s", {}).get("name", id(self.embedder)))
            uncached = [e for e in ents_to_embed if (emb_key, e) not in _GLOBAL_ENTITY_EMB_CACHE]
            if uncached:
                vecs = self.embedder.embed_texts(uncached)
                for e, v in zip(uncached, vecs):
                    _GLOBAL_ENTITY_EMB_CACHE[(emb_key, e)] = list(v)
                _save_entity_cache()
            for e in ents_to_embed:
                self.entity_embeddings[e] = _GLOBAL_ENTITY_EMB_CACHE[(emb_key, e)]

        # Level 1: Add synonymy edges between entities with cosine similarity >= synonym_threshold
        if len(self.entity_embeddings) > 1:
            all_ents = sorted(self.entity_embeddings.keys())
            ents_mat = np.array([self.entity_embeddings[e] for e in all_ents], dtype=np.float32)
            norms = np.linalg.norm(ents_mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            ents_mat = ents_mat / norms
            sim_matrix = ents_mat @ ents_mat.T
            for i, u_ent in enumerate(all_ents):
                u = f"entity:{u_ent}"
                for j in range(i + 1, len(all_ents)):
                    sim = float(sim_matrix[i, j])
                    if sim >= self.synonym_threshold:
                        v = f"entity:{all_ents[j]}"
                        self.graph.add_edge(u, v, relation="synonym", type="synonym", weight=sim)

        # Level 2: Construct thematic cluster vertices
        self._build_thematic_clusters()

    def _build_thematic_clusters(self) -> None:
        """Form Level 2 thematic cluster vertices via deterministic agglomerative clustering."""
        # Remove existing thematic cluster nodes and incident edges
        existing_themes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "thematic_cluster"]
        self.graph.remove_nodes_from(existing_themes)
        self.cluster_centroids.clear()
        self.cluster_members.clear()

        entities = sorted(self.entity_embeddings.keys())
        n = len(entities)
        if n == 0:
            return

        if n == 1:
            clusters = {0: [entities[0]]}
        else:
            X = np.array([self.entity_embeddings[e] for e in entities], dtype=float)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X_norm = X / norms

            model = AgglomerativeClustering(
                metric="cosine",
                linkage="average",
                distance_threshold=self.cluster_distance_threshold,
                n_clusters=None,
            )
            labels = model.fit_predict(X_norm)

            groups: dict[int, list[str]] = defaultdict(list)
            for ent, lab in zip(entities, labels):
                groups[int(lab)].append(ent)

            multi = {k: v for k, v in groups.items() if len(v) >= 2}
            singletons = [v[0] for k, v in groups.items() if len(v) == 1]

            if not multi:
                target_k = max(1, min(n // 2, self.max_clusters))
                model2 = AgglomerativeClustering(metric="cosine", linkage="average", n_clusters=target_k)
                labels2 = model2.fit_predict(X_norm)
                final_groups: dict[int, list[str]] = defaultdict(list)
                for ent, lab in zip(entities, labels2):
                    final_groups[int(lab)].append(ent)
                clusters = {i: grp for i, grp in enumerate(final_groups.values())}
            else:
                for s_ent in singletons:
                    s_vec = self.entity_embeddings[s_ent]
                    best_k = None
                    best_sim = -float("inf")
                    for k, members in multi.items():
                        c_vec = np.mean([self.entity_embeddings[m] for m in members], axis=0)
                        sim = _cosine(s_vec, c_vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_k = k
                    if best_k is not None:
                        multi[best_k].append(s_ent)

                cluster_list = list(multi.values())
                # Cap clusters at max_clusters by iteratively merging closest pairs
                while len(cluster_list) > self.max_clusters:
                    best_pair = None
                    best_sim = -float("inf")
                    for i in range(len(cluster_list)):
                        c1 = np.mean([self.entity_embeddings[m] for m in cluster_list[i]], axis=0)
                        for j in range(i + 1, len(cluster_list)):
                            c2 = np.mean([self.entity_embeddings[m] for m in cluster_list[j]], axis=0)
                            sim = _cosine(c1, c2)
                            if sim > best_sim:
                                best_sim = sim
                                best_pair = (i, j)
                    if best_pair is None:
                        break
                    i, j = best_pair
                    cluster_list[i].extend(cluster_list[j])
                    cluster_list.pop(j)

                clusters = {i: grp for i, grp in enumerate(cluster_list)}

        # Add thematic cluster nodes and member edges to graph
        for c_id, members in clusters.items():
            cluster_node = f"theme:{c_id}"
            centroid = np.mean([self.entity_embeddings[m] for m in members], axis=0).tolist()
            self.cluster_centroids[cluster_node] = centroid
            self.cluster_members[cluster_node] = list(members)

            self.graph.add_node(cluster_node, type="thematic_cluster", cluster_id=c_id)
            for m in members:
                entity_node = f"entity:{m}"
                self.graph.add_edge(
                    cluster_node,
                    entity_node,
                    relation="member_of",
                    type="thematic",
                    weight=1.0,
                )

    def retrieve(self, query: str, k: int, timestamp: str | None = None) -> list[RetrievedItem]:
        if not self.records or self.graph.number_of_nodes() == 0:
            return []
        self._ensure_embedder()

        if hasattr(self.embedder, "embed_query"):
            q_vec = self.embedder.embed_query(query)
        else:
            q_vec = self.embedder.embed_texts([query])[0]

        q_lower = query.lower()

        # Step 1: Low-level matching
        # Match query entities against entity vertices via embedding cosine similarity (>= 0.80)
        # or exact word boundary match in query text.
        low_level_matches: list[tuple[str, float]] = []
        entity_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]
        all_entity_scores: list[tuple[str, float]] = []

        for en_node in entity_nodes:
            en_name = en_node[len("entity:"):]
            sim_q = 0.0
            if en_name in self.entity_embeddings:
                sim_q = _cosine(q_vec, self.entity_embeddings[en_name])

            # Check exact word boundary match in query text
            is_exact = bool(len(en_name) >= 2 and re.search(r"\b" + re.escape(en_name) + r"\b", q_lower))
            exact_sim = 1.0 if is_exact else 0.0

            best_sim = max(exact_sim, sim_q)
            all_entity_scores.append((en_node, best_sim))
            if best_sim >= self.low_level_threshold:
                low_level_matches.append((en_node, best_sim))

        # Fallback if no entity reached low_level_threshold
        if not low_level_matches and all_entity_scores:
            all_entity_scores.sort(key=lambda x: x[1], reverse=True)
            low_level_matches = all_entity_scores[:min(5, len(all_entity_scores))]

        # Step 2: High-level matching
        # Compute cosine similarity between query embedding and thematic cluster centroids
        theme_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "thematic_cluster"]
        cluster_scores: list[tuple[str, float]] = []
        for t_node in theme_nodes:
            centroid = self.cluster_centroids.get(t_node)
            if centroid is not None:
                sim_c = _cosine(q_vec, centroid)
                cluster_scores.append((t_node, sim_c))

        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        top_clusters = cluster_scores[:2]

        # Step 3: Dual-level Personalized PageRank
        # Restart vector allocates 60% mass to low-level query entities and 40% mass to top-2 thematic clusters
        personalization: dict[str, float] = {}
        has_low = bool(low_level_matches)
        has_high = bool(top_clusters)

        low_mass = 0.60 if (has_low and has_high) else (1.0 if has_low else 0.0)
        high_mass = 0.40 if (has_low and has_high) else (1.0 if has_high else 0.0)

        if has_low:
            sum_low = sum(s for _, s in low_level_matches)
            if sum_low > 0:
                for node, s in low_level_matches:
                    personalization[node] = personalization.get(node, 0.0) + low_mass * (s / sum_low)
            else:
                for node, _ in low_level_matches:
                    personalization[node] = personalization.get(node, 0.0) + low_mass / len(low_level_matches)

        if has_high:
            pos_sims = [max(0.0, s) for _, s in top_clusters]
            sum_high = sum(pos_sims)
            if sum_high > 0:
                for (node, _), s in zip(top_clusters, pos_sims):
                    personalization[node] = personalization.get(node, 0.0) + high_mass * (s / sum_high)
            else:
                for node, _ in top_clusters:
                    personalization[node] = personalization.get(node, 0.0) + high_mass / len(top_clusters)

        # Normalize personalization vector
        total_p = sum(personalization.values())
        if total_p > 0:
            for n in personalization:
                personalization[n] /= total_p
        else:
            all_nodes = list(self.graph.nodes())
            personalization = {n: 1.0 / len(all_nodes) for n in all_nodes} if all_nodes else {}

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
            ppr_probs = dict(personalization)

        # Step 4: Candidate turn scoring
        # Final score of a turn is the sum of stationary activation probabilities of its constituent entities
        scored_turns: list[tuple[float, str, Any, str, MemoryRecord]] = []
        for t_id, recs in self.turn_records.items():
            primary_rec = recs[0]
            orig_tid = primary_rec.turn_id
            turn_ents = self.turn_entities.get(t_id, set())

            # Sum stationary probability across constituent entity nodes
            ent_score = sum(ppr_probs.get(en_node, 0.0) for en_node in turn_ents)
            if ent_score <= 0.0:
                ent_score = 1e-6

            # Combine distinct propositions for turn text
            unique_props = list(dict.fromkeys(r.fact_text for r in recs))
            turn_text = " ".join(unique_props)

            scored_turns.append((ent_score, t_id, orig_tid, turn_text, primary_rec))

        scored_turns.sort(key=lambda x: x[0], reverse=True)
        top_turns = scored_turns[:k]

        return [
            RetrievedItem(
                id=f"turn:{orig_tid}",
                text=turn_text,
                score=score,
                turn_id=orig_tid,
                metadata={
                    "timestamp": primary_rec.timestamp,
                    "speaker": primary_rec.speaker,
                    "sample_id": primary_rec.sample_id,
                },
            )
            for score, t_id, orig_tid, turn_text, primary_rec in top_turns
        ]

    def export_graph(self) -> nx.Graph:
        """Returns the simple undirected projection of the internal NetworkX graph."""
        return self.graph

    def clear(self) -> None:
        self.records.clear()
        self.graph.clear()
        self.occurrence_map.clear()
        self.turn_records.clear()
        self.turn_entities.clear()
        self.entity_embeddings.clear()
        self.cluster_centroids.clear()
        self.cluster_members.clear()
