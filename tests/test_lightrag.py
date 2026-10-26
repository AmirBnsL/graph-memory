import sys
import unittest
from pathlib import Path
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.pipeline.extract import MemoryRecord
from graph_memory.memory.base import BaseMemory, RetrievedItem
from graph_memory.analysis.graph_metrics import compute_graph_metrics


class MockEmbedder:
    """Deterministic mock embedder for fast offline unit testing."""

    def __init__(self):
        # Predefined embeddings for specific terms to test synonymy and clustering
        self.known_vectors = {
            "alice": [1.0, 0.1, 0.0, 0.0],
            "bob": [0.95, 0.12, 0.0, 0.0],       # Very close to alice (synonym/cluster)
            "paris": [0.0, 1.0, 0.1, 0.0],
            "berlin": [0.0, 0.95, 0.12, 0.0],     # Very close to paris
            "france": [0.0, 0.90, 0.15, 0.0],
            "bmw": [0.0, 0.0, 1.0, 0.1],
            "car": [0.0, 0.0, 0.95, 0.12],        # Very close to bmw
            "automobile": [0.0, 0.0, 0.92, 0.14],
        }

    def _vec_for_text(self, text: str) -> list[float]:
        import hashlib
        t = text.lower().strip()
        for k, v in self.known_vectors.items():
            if k in t:
                return list(v)
        # Deterministic pseudo-embedding based on md5 hash
        h = int.from_bytes(hashlib.md5(t.encode("utf-8")).digest()[:4], "big")
        v = [float((h >> (i * 8)) & 0xFF) / 255.0 for i in range(4)]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vec_for_text(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec_for_text(query)


class TestLightRAGMemory(unittest.TestCase):
    def setUp(self):
        self.mock_embedder = MockEmbedder()
        self.records = [
            MemoryRecord(
                id="rec_1",
                sample_id="conv_1",
                turn_id=0,
                speaker="User",
                timestamp="2023-01-01",
                fact_text="Alice lives in Paris, France.",
                entities=["Alice", "Paris", "France"],
                triples=[("Alice", "lives in", "Paris"), ("Paris", "is in", "France")],
            ),
            MemoryRecord(
                id="rec_2",
                sample_id="conv_1",
                turn_id=1,
                speaker="User",
                timestamp="2023-03-01",
                fact_text="Bob bought a blue BMW car.",
                entities=["Bob", "BMW", "car"],
                triples=[("Bob", "bought", "BMW"), ("BMW", "is a", "car")],
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

    def test_imports_and_registration(self):
        from graph_memory import LightRAGMemory
        from graph_memory.memory import LightRAGMemory as LightRAGMemoryFromMem
        from graph_memory.memory.graph import LightRAGMemory as LightRAGMemoryFromGraph

        self.assertIs(LightRAGMemory, LightRAGMemoryFromMem)
        self.assertIs(LightRAGMemory, LightRAGMemoryFromGraph)
        self.assertTrue(issubclass(LightRAGMemory, BaseMemory))

    def test_ingest_and_graph_tiers(self):
        from graph_memory import LightRAGMemory

        mem = LightRAGMemory(embedder=self.mock_embedder, synonym_threshold=0.85)
        mem.add_records(self.records)

        g = mem.export_graph()
        self.assertIsInstance(g, nx.Graph)

        # Level 1: Entity vertices
        entity_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "entity"]
        self.assertIn("entity:alice", entity_nodes)
        self.assertIn("entity:paris", entity_nodes)
        self.assertIn("entity:bmw", entity_nodes)
        self.assertIn("entity:bob", entity_nodes)

        # Relational edges
        self.assertTrue(g.has_edge("entity:alice", "entity:paris"))
        self.assertTrue(g.has_edge("entity:bob", "entity:bmw"))

        # Synonym edges
        # Alice and Bob have cosine similarity > 0.85 in mock_embedder
        self.assertTrue(g.has_edge("entity:alice", "entity:bob"))
        # BMW and car have cosine similarity > 0.85
        self.assertTrue(g.has_edge("entity:bmw", "entity:car"))

        # Level 2: Thematic cluster vertices
        theme_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "thematic_cluster"]
        self.assertGreaterEqual(len(theme_nodes), 1)

        # Thematic edges connect member entities to cluster
        for t_node in theme_nodes:
            neighbors = list(g.neighbors(t_node))
            # Each cluster has >= 2 entities if possible
            self.assertGreaterEqual(len(neighbors), 2)
            for neighbor in neighbors:
                self.assertTrue(neighbor.startswith("entity:"))
                edge_data = g.get_edge_data(t_node, neighbor)
                self.assertEqual(edge_data.get("weight"), 1.0)

        # Occurrence mapping
        self.assertIn(0, mem.occurrence_map["entity:alice"] | mem.occurrence_map.get("alice", set()))
        self.assertIn(1, mem.occurrence_map["entity:bmw"] | mem.occurrence_map.get("bmw", set()))

    def test_export_graph_metrics(self):
        from graph_memory import LightRAGMemory

        mem = LightRAGMemory(embedder=self.mock_embedder)
        mem.add_records(self.records)

        g = mem.export_graph()
        metrics = compute_graph_metrics(g)

        self.assertGreater(metrics["num_nodes"], 0)
        self.assertGreater(metrics["num_edges"], 0)
        self.assertGreater(metrics["largest_component_ratio"], 0.0)
        self.assertGreaterEqual(metrics["degree_gini"], 0.0)
        self.assertGreaterEqual(metrics["max_degree_ratio"], 1.0)
        self.assertGreaterEqual(metrics["edge_density"], 0.0)

    def test_dual_level_retrieval(self):
        from graph_memory import LightRAGMemory

        mem = LightRAGMemory(embedder=self.mock_embedder)
        mem.add_records(self.records)

        # Query about BMW/car in turn 1
        items = mem.retrieve(query="What car did Bob buy? BMW", k=2)
        self.assertIsInstance(items, list)
        self.assertGreater(len(items), 0)
        self.assertLessEqual(len(items), 2)

        top_item = items[0]
        self.assertIsInstance(top_item, RetrievedItem)
        self.assertTrue(top_item.id.startswith("turn:"))
        self.assertEqual(str(top_item.turn_id), "1")
        self.assertGreater(top_item.score, 0.0)
        self.assertIn("BMW", top_item.text)

        # Verify all retrieved items have positive scores and valid turn IDs
        for item in items:
            self.assertGreater(item.score, 0.0)
            self.assertIsNotNone(item.turn_id)
            self.assertEqual(item.id, f"turn:{item.turn_id}")

    def test_pipeline_answer_memory(self):
        from unittest.mock import patch
        from graph_memory.pipeline import answer
        from graph_memory.pipeline.load import Conversation, Turn, QA

        conv = Conversation(
            sample_id="conv_pipe_1",
            speaker_a="User",
            speaker_b="Assistant",
            turns=[
                Turn(dia_id="t0", speaker="User", text="Alice visited Paris.", session_id=0, date_time="2023-01-01"),
                Turn(dia_id="t1", speaker="User", text="Bob bought a BMW in Berlin.", session_id=0, date_time="2023-02-01"),
            ],
            qa=[],
        )
        qa = QA(question="What did Bob buy?", answer="BMW", category=1, evidence=["t1"])

        with patch("graph_memory.pipeline.extract.extract_conversation") as mock_extract, \
             patch("graph_memory.core.llm.complete", return_value="Bob bought a BMW."):
            mock_extract.return_value = self.records
            answer.clear_memory_cache()

            res = answer.answer_memory(
                conv=conv,
                qa=qa,
                k=2,
                mode_cfg={"backend": "lightrag"},
                retriever={"name": "all-MiniLM-L6-v2"},
            )
            self.assertEqual(res.answer, "Bob bought a BMW.")
            self.assertIn("1", res.context_ids)

    def test_edge_case_single_entity_record(self):
        from graph_memory import LightRAGMemory

        rec = MemoryRecord(
            id="single_rec",
            sample_id="conv_single",
            turn_id=0,
            speaker="User",
            timestamp="2023-01-01",
            fact_text="Alice lives alone.",
            entities=["Alice"],
            triples=[],
        )
        mem = LightRAGMemory(embedder=self.mock_embedder)
        mem.add_records([rec])

        g = mem.export_graph()
        self.assertIn("entity:alice", g.nodes)
        self.assertIn("theme:0", g.nodes)

        items = mem.retrieve("Where is Alice?", k=1)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].turn_id, 0)
        self.assertGreater(items[0].score, 0.0)

    def test_clear(self):
        from graph_memory import LightRAGMemory

        mem = LightRAGMemory(embedder=self.mock_embedder)
        mem.add_records(self.records)
        self.assertGreater(mem.graph.number_of_nodes(), 0)

        mem.clear()
        self.assertEqual(mem.graph.number_of_nodes(), 0)
        self.assertEqual(len(mem.records), 0)
        self.assertEqual(len(mem.occurrence_map), 0)
        self.assertEqual(len(mem.retrieve("BMW", k=1)), 0)


if __name__ == "__main__":
    unittest.main()
