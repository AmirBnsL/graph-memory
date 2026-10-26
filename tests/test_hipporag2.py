import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.pipeline.extract import MemoryRecord
from graph_memory.memory.base import BaseMemory, RetrievedItem
from graph_memory import HippoRAG2Memory
from graph_memory.memory.graph import HippoRAG2Memory as HippoRAG2MemoryFromGraph
from graph_memory.memory import HippoRAG2Memory as HippoRAG2MemoryFromMemory
from graph_memory.analysis.graph_metrics import compute_graph_metrics


class MockEmbedder:
    """Deterministic mock embedder for fast offline unit testing."""
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            val = float(len(t))
            vectors.append([
                val,
                val * 0.5,
                float("paris" in t.lower() or "france" in t.lower()),
                float("bmw" in t.lower() or "car" in t.lower()),
            ])
        return vectors


class TestHippoRAG2Memory(unittest.TestCase):
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
        self.assertTrue(issubclass(HippoRAG2Memory, BaseMemory))
        self.assertIs(HippoRAG2Memory, HippoRAG2MemoryFromGraph)
        self.assertIs(HippoRAG2Memory, HippoRAG2MemoryFromMemory)

    def test_empty_memory(self):
        mem = HippoRAG2Memory(embedder=self.mock_embedder)
        self.assertEqual(mem.retrieve("Any query", k=5), [])
        g = mem.export_graph()
        self.assertIsNotNone(g)
        self.assertEqual(g.number_of_nodes(), 0)
        mem.clear()
        self.assertEqual(len(mem.records), 0)

    def test_graph_construction_and_node_types(self):
        mem = HippoRAG2Memory(embedder=self.mock_embedder, synonym_threshold=0.85)
        mem.add_records(self.records)

        g = mem.export_graph()
        self.assertIsNotNone(g)

        # Check Turn vertices
        self.assertIn("turn:0", g.nodes)
        self.assertIn("turn:1", g.nodes)
        self.assertIn("turn:2", g.nodes)
        self.assertEqual(g.nodes["turn:0"]["type"], "turn")
        self.assertEqual(g.nodes["turn:0"]["turn_id"], 0)
        self.assertEqual(g.nodes["turn:0"]["text"], "Alice lives in Paris.")

        # Check Entity vertices
        self.assertIn("entity:alice", g.nodes)
        self.assertIn("entity:paris", g.nodes)
        self.assertIn("entity:bmw", g.nodes)
        self.assertIn("entity:berlin", g.nodes)
        self.assertEqual(g.nodes["entity:alice"]["type"], "entity")
        self.assertEqual(g.nodes["entity:alice"]["name"], "alice")

        # Check Bipartite Occurrence Edges
        self.assertTrue(g.has_edge("turn:0", "entity:alice"))
        self.assertTrue(g.has_edge("turn:0", "entity:paris"))
        self.assertTrue(g.has_edge("turn:1", "entity:alice"))
        self.assertTrue(g.has_edge("turn:1", "entity:bmw"))
        self.assertEqual(g.edges["turn:0", "entity:alice"]["weight"], 1.0)
        self.assertEqual(g.edges["turn:0", "entity:alice"]["type"], "occurrence")

        # Check Entity-Entity Relational Edges
        self.assertTrue(g.has_edge("entity:alice", "entity:paris"))
        self.assertEqual(g.edges["entity:alice", "entity:paris"]["weight"], 1.0)
        self.assertEqual(g.edges["entity:alice", "entity:paris"]["type"], "relation")
        self.assertEqual(g.edges["entity:alice", "entity:paris"]["relation"], "lives in")

    def test_multi_proposition_turn_text_aggregation(self):
        records = [
            MemoryRecord(
                id="r1",
                sample_id="c1",
                turn_id=10,
                speaker="Alice",
                timestamp="2024-01-01",
                fact_text="Alice adopted a cat.",
                entities=["Alice", "cat"],
                triples=[("Alice", "adopted", "cat")],
            ),
            MemoryRecord(
                id="r2",
                sample_id="c1",
                turn_id=10,
                speaker="Alice",
                timestamp="2024-01-01",
                fact_text="The cat is named Luna.",
                entities=["cat", "Luna"],
                triples=[("cat", "named", "Luna")],
            ),
        ]
        mem = HippoRAG2Memory(embedder=self.mock_embedder)
        mem.add_records(records)

        g = mem.export_graph()
        self.assertIn("turn:10", g.nodes)
        turn_text = g.nodes["turn:10"]["text"]
        self.assertIn("Alice adopted a cat.", turn_text)
        self.assertIn("The cat is named Luna.", turn_text)

    def test_bipartite_transition_probabilities(self):
        mem = HippoRAG2Memory(embedder=self.mock_embedder, synonym_threshold=0.85)
        mem.add_records(self.records)

        transitions = mem._build_transition_matrix()

        # For entity:alice: incident to turns (0, 1, 2) and entities (paris, bmw, berlin)
        # 50% distributed among turn neighbors; 50% among entity neighbors
        t_alice = transitions["entity:alice"]
        turn_prob = sum(prob for target, prob in t_alice.items() if target.startswith("turn:"))
        ent_prob = sum(prob for target, prob in t_alice.items() if target.startswith("entity:"))
        self.assertAlmostEqual(turn_prob, 0.5, places=5)
        self.assertAlmostEqual(ent_prob, 0.5, places=5)
        self.assertAlmostEqual(sum(t_alice.values()), 1.0, places=5)

        # For turn:1: connected to entity:alice and entity:bmw
        # 100% distributed equally among constituent entity nodes
        t_turn1 = transitions["turn:1"]
        self.assertAlmostEqual(sum(t_turn1.values()), 1.0, places=5)
        for target, prob in t_turn1.items():
            self.assertTrue(target.startswith("entity:"))
            self.assertAlmostEqual(prob, 0.5, places=5)

    def test_transition_probabilities_boundary_conditions(self):
        # Entity with only turn neighbors (no relational or synonym edges)
        mem = HippoRAG2Memory(embedder=self.mock_embedder)
        rec = [
            MemoryRecord(
                id="rec_solo",
                sample_id="c_solo",
                turn_id=99,
                speaker="User",
                timestamp="2024-01-01",
                fact_text="Solo entity test.",
                entities=["QuantumPhysics"],
                triples=[],
            )
        ]
        mem.add_records(rec)
        transitions = mem._build_transition_matrix()

        # entity:quantumphysics has only turn:99 as neighbor -> 100% to turn:99
        self.assertIn("entity:quantumphysics", transitions)
        self.assertEqual(transitions["entity:quantumphysics"], {"turn:99": 1.0})

        # turn:99 has only entity:quantumphysics as neighbor -> 100% to entity:quantumphysics
        self.assertIn("turn:99", transitions)
        self.assertEqual(transitions["turn:99"], {"entity:quantumphysics": 1.0})

    def test_retrieval_and_turn_scoring(self):
        mem = HippoRAG2Memory(embedder=self.mock_embedder, synonym_threshold=0.85)
        mem.add_records(self.records)

        # Query targeting BMW in turn 1
        retrieved = mem.retrieve(query="What car did Alice buy? BMW", k=2)

        self.assertGreaterEqual(len(retrieved), 1)
        self.assertLessEqual(len(retrieved), 2)

        top_item = retrieved[0]
        self.assertIsInstance(top_item, RetrievedItem)
        self.assertEqual(top_item.id, "turn:1")
        self.assertEqual(top_item.turn_id, 1)
        self.assertIn("BMW", top_item.text)
        self.assertGreater(top_item.score, 0.0)

        # Verify descending order of scores
        if len(retrieved) > 1:
            self.assertGreaterEqual(retrieved[0].score, retrieved[1].score)

    def test_compute_graph_metrics(self):
        mem = HippoRAG2Memory(embedder=self.mock_embedder, synonym_threshold=0.85)
        mem.add_records(self.records)

        g = mem.export_graph()
        metrics = compute_graph_metrics(g)

        self.assertEqual(metrics["num_nodes"], g.number_of_nodes())
        self.assertEqual(metrics["num_edges"], g.number_of_edges())
        self.assertGreater(metrics["largest_component_ratio"], 0.0)
        self.assertGreaterEqual(metrics["degree_gini"], 0.0)
        self.assertGreaterEqual(metrics["max_degree_ratio"], 1.0)

    def test_clear_resets_all_state(self):
        mem = HippoRAG2Memory(embedder=self.mock_embedder)
        mem.add_records(self.records)
        self.assertGreater(mem.graph.number_of_nodes(), 0)
        self.assertGreater(len(mem.records), 0)

        mem.clear()
        self.assertEqual(mem.graph.number_of_nodes(), 0)
        self.assertEqual(len(mem.records), 0)
        self.assertEqual(len(mem.entity_embeddings), 0)
        self.assertEqual(len(mem.turn_texts), 0)
        self.assertEqual(len(mem.turn_metadata), 0)


if __name__ == "__main__":
    unittest.main()
