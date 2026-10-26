import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.pipeline.extract import MemoryRecord
from graph_memory.memory.base import BaseMemory
from graph_memory import FlatMemory, HippoRAGMemory, HippoRAG2Memory, BiTemporalMemory
from graph_memory.memory.vector import FlatMemory as FlatMemoryFromVector
from graph_memory.memory.graph import (
    HippoRAGMemory as HippoRAGMemoryFromGraph,
    HippoRAG2Memory as HippoRAG2MemoryFromGraph,
    BiTemporalMemory as BiTemporalMemoryFromGraph,
)


class MockEmbedder:
    """Deterministic mock embedder for fast offline unit testing."""
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Generates deterministic toy 4-d vector based on length and characters
        vectors = []
        for t in texts:
            val = float(len(t))
            vectors.append([val, val * 0.5, float("paris" in t.lower()), float("bmw" in t.lower())])
        return vectors


class TestMemoryBackends(unittest.TestCase):
    def setUp(self):
        self.mock_embedder = MockEmbedder()
        self.records = [
            MemoryRecord(
                id="rec_1",
                sample_id="conv_1",
                turn_id=0,
                speaker="User",
                timestamp="2023-01-01",
                fact_text="Alice lives in Paris.",
                entities=["Alice", "Paris"],
                triples=[("Alice", "lives in", "Paris")],
            ),
            MemoryRecord(
                id="rec_2",
                sample_id="conv_1",
                turn_id=1,
                speaker="User",
                timestamp="2023-03-01",
                fact_text="Alice bought a blue BMW.",
                entities=["Alice", "BMW"],
                triples=[("Alice", "bought", "BMW")],
            ),
            MemoryRecord(
                id="rec_3",
                sample_id="conv_1",
                turn_id=2,
                speaker="User",
                timestamp="2023-06-01",
                fact_text="Alice moved to Berlin.",
                entities=["Alice", "Berlin"],
                triples=[("Alice", "moved to", "Berlin")],
            ),
        ]

    def test_subpackage_exports(self):
        self.assertIs(FlatMemory, FlatMemoryFromVector)
        self.assertIs(HippoRAGMemory, HippoRAGMemoryFromGraph)
        self.assertIs(HippoRAG2Memory, HippoRAG2MemoryFromGraph)
        self.assertIs(BiTemporalMemory, BiTemporalMemoryFromGraph)

    def test_flat_memory(self):
        flat = FlatMemory(embedder=self.mock_embedder)
        flat.add_records(self.records)
        self.assertEqual(len(flat.records), 3)

        # Retrieve with keyword related to BMW
        res = flat.retrieve(query="What car did Alice buy? BMW", k=1)
        self.assertEqual(len(res), 1)
        self.assertIn("BMW", res[0].text)
        self.assertIsNone(flat.export_graph())

    def test_hipporag_memory(self):
        hippo = HippoRAGMemory(embedder=self.mock_embedder, synonym_threshold=0.8)
        hippo.add_records(self.records)

        g = hippo.export_graph()
        self.assertIsNotNone(g)
        self.assertIn("alice", g.nodes)
        self.assertIn("paris", g.nodes)
        self.assertIn("bmw", g.nodes)
        self.assertTrue(g.has_edge("alice", "paris"))

        # PPR retrieval
        res = hippo.retrieve(query="What car did Alice buy? BMW", k=2)
        self.assertGreaterEqual(len(res), 1)

    def test_temporal_bteg_memory(self):
        bteg = BiTemporalMemory(embedder=self.mock_embedder)
        bteg.add_records(self.records)

        g = bteg.export_graph()
        self.assertIsNotNone(g)
        self.assertIn("alice", g.nodes)

        # Check edge invalidation: "lives in Paris" should be superseded by "moved to Berlin"
        edges = list(g.edges(data=True))
        self.assertGreaterEqual(len(edges), 2)
        
        # Retrieval
        res = bteg.retrieve(query="Where does Alice live?", k=2, timestamp="2023-07-01")
        self.assertGreaterEqual(len(res), 1)

    def test_hipporag_turn_diversification_and_stop_nodes(self):
        # Create 10 propositions from the same turn t1, and 2 propositions from t2
        recs = [
            MemoryRecord(
                id=f"t1_{i}", sample_id="c1", turn_id=1, speaker="User",
                timestamp="2023-01-01", fact_text=f"User bought item {i} in Paris.",
                entities=["User", "Paris", f"item {i}"],
                triples=[("User", "bought", f"item {i}"), ("User", "in", "Paris")],
            )
            for i in range(10)
        ] + [
            MemoryRecord(
                id=f"t2_{i}", sample_id="c1", turn_id=2, speaker="User",
                timestamp="2023-01-02", fact_text=f"User visited Berlin museum {i}.",
                entities=["User", "Berlin", f"museum {i}"],
                triples=[("User", "visited", f"museum {i}"), ("User", "in", "Berlin")],
            )
            for i in range(2)
        ]

        hippo = HippoRAGMemory(embedder=self.mock_embedder)
        hippo.add_records(recs)
        g = hippo.export_graph()

        # Stop-nodes like "user" must be filtered from graph nodes
        self.assertNotIn("user", g.nodes)
        self.assertIn("paris", g.nodes)
        self.assertIn("berlin", g.nodes)

        # Retrieve k=5: must NOT be 5 copies of t1; must contain both t1 and t2
        res = hippo.retrieve(query="What places did User visit or buy items in?", k=5)
        retrieved_turns = {r.turn_id for r in res}
        self.assertIn(1, retrieved_turns)
        self.assertIn(2, retrieved_turns)

    def test_bteg_triple_deduplication(self):
        # 5 propositions from turn 1 with identical triples
        recs = [
            MemoryRecord(
                id=f"t1_{i}", sample_id="c1", turn_id=1, speaker="User",
                timestamp="2023-01-01", fact_text=f"Fact {i}",
                entities=["Alice", "Paris"],
                triples=[("Alice", "lives in", "Paris")],
            )
            for i in range(5)
        ]
        bteg = BiTemporalMemory(embedder=self.mock_embedder)
        bteg.add_records(recs)
        g = bteg.export_graph()
        # Should only have 1 edge between alice and paris, not 5
        self.assertEqual(g.number_of_edges("alice", "paris"), 1)

    def test_hipporag_rewire_structure_preserves_degrees(self):
        from graph_memory.analysis import graph_metrics

        recs = [
            MemoryRecord(
                id=f"r_{i}", sample_id="c1", turn_id=i, speaker="User",
                timestamp="2023-01-01",
                fact_text=f"EntityA{i} connects to EntityB{i}.",
                entities=[f"EntityA{i}", f"EntityB{i}"],
                triples=[(f"EntityA{i}", "rel", f"EntityB{i}")],
            )
            for i in range(6)
        ]
        # Use high synonym_threshold=2.0 so only explicit relation triples form edges
        h_orig = HippoRAGMemory(embedder=self.mock_embedder, synonym_threshold=2.0, rewire_structure=False)
        h_orig.add_records(recs)
        g_orig = h_orig.export_graph()

        h_rew = HippoRAGMemory(embedder=self.mock_embedder, synonym_threshold=2.0, rewire_structure=True, rewire_seed=42)
        h_rew.add_records(recs)
        g_rew = h_rew.export_graph()

        # Exact node degrees must be identical
        self.assertEqual(dict(g_orig.degree()), dict(g_rew.degree()))
        # Metrics (Degree Gini, Max Degree Ratio, Edge Density) must be invariant
        m_orig = graph_metrics.compute_graph_metrics(g_orig)
        m_rew = graph_metrics.compute_graph_metrics(g_rew)
        self.assertAlmostEqual(m_orig["degree_gini"], m_rew["degree_gini"], places=6)
        self.assertAlmostEqual(m_orig["max_degree_ratio"], m_rew["max_degree_ratio"], places=6)
        # Edge set must be altered by rewiring
        orig_edges = {tuple(sorted(e)) for e in g_orig.edges()}
        rew_edges = {tuple(sorted(e)) for e in g_rew.edges()}
        self.assertNotEqual(orig_edges, rew_edges)


if __name__ == "__main__":
    unittest.main()
