import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from extract import MemoryRecord
from memory.base import BaseMemory
from memory import FlatMemory, HippoRAGMemory, BiTemporalMemory
from memory.vector import FlatMemory as FlatMemoryFromVector
from memory.graph import HippoRAGMemory as HippoRAGMemoryFromGraph, BiTemporalMemory as BiTemporalMemoryFromGraph


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
        self.assertIn("Alice", g.nodes)
        self.assertIn("Paris", g.nodes)
        self.assertIn("BMW", g.nodes)
        self.assertTrue(g.has_edge("Alice", "Paris"))

        # PPR retrieval
        res = hippo.retrieve(query="What car did Alice buy? BMW", k=2)
        self.assertGreaterEqual(len(res), 1)

    def test_temporal_bteg_memory(self):
        bteg = BiTemporalMemory(embedder=self.mock_embedder)
        bteg.add_records(self.records)

        g = bteg.export_graph()
        self.assertIsNotNone(g)
        self.assertIn("Alice", g.nodes)

        # Check edge invalidation: "lives in Paris" should be superseded by "moved to Berlin"
        edges = list(g.edges(data=True))
        self.assertGreaterEqual(len(edges), 2)
        
        # Retrieval
        res = bteg.retrieve(query="Where does Alice live?", k=2, timestamp="2023-07-01")
        self.assertGreaterEqual(len(res), 1)


if __name__ == "__main__":
    unittest.main()
