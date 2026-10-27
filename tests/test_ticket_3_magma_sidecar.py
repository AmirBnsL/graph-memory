import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from graph_memory.analysis import external_inspector, predict_utility


class TestTicket3MagmaSidecar(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_magma_multi_graph_projection_and_metrics(self):
        # Create a MultiDiGraph with 4 edge types (Temporal, Causal, Entity, Semantic)
        mg = nx.MultiDiGraph()
        for i in range(10):
            mg.add_node(i, event=f"event_{i}")

        # Add parallel and directed edges
        mg.add_edge(0, 1, relation="temporal")
        mg.add_edge(0, 1, relation="causal")
        mg.add_edge(1, 2, relation="entity")
        mg.add_edge(2, 3, relation="semantic")
        mg.add_edge(3, 0, relation="temporal")
        # Self-loop to verify stripping
        mg.add_edge(0, 0, relation="temporal")

        cache_file = self.dir_path / "magma_graph.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(mg, f)

        simple_g = external_inspector.extract_magma_graph(cache_file)
        self.assertIsInstance(simple_g, nx.Graph)
        self.assertFalse(simple_g.is_directed())
        self.assertEqual(simple_g.number_of_nodes(), 10)
        self.assertFalse(simple_g.has_edge(0, 0), "Self-loops must be stripped")
        self.assertTrue(simple_g.has_edge(0, 1))

    def test_inspect_magma_run_end_to_end_through_h2_seam(self):
        mg = nx.MultiDiGraph()
        for i in range(6):
            mg.add_node(i)
        mg.add_edge(0, 1)
        mg.add_edge(1, 2)
        mg.add_edge(2, 3)

        cache_file = self.dir_path / "magma_graph.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(mg, f)

        mock_qa = [
            {
                "sample_id": "sample_0",
                "question": "What happened first?",
                "answer": "Event A",
                "evidence": ["D1:1"],
                "context_ids": ["D1:1", "D1:2"],
                "F1": 0.82,
                "category": 1,
            }
        ]

        run_dict = external_inspector.inspect_upstream_run(
            framework="magma",
            graph_artifacts_by_sample={"sample_0": cache_file},
            qa_results=mock_qa,
            benchmark="locomo",
            k=10,
            n=10,
        )

        self.assertEqual(run_dict["mode"], "memory_magma")
        self.assertIn("graph_metrics", run_dict)
        self.assertIn("degree_gini", run_dict["graph_metrics"])
        self.assertIn("max_degree_ratio", run_dict["graph_metrics"])

        # Feed run_dict into collect_sample_observations
        obs = predict_utility.collect_sample_observations([run_dict], natural_only=True)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["mode"], "memory_magma")
        self.assertAlmostEqual(obs[0]["evidence_coverage"], 1.0)
        self.assertAlmostEqual(obs[0]["precision_at_k"], 0.5)


if __name__ == "__main__":
    unittest.main()
