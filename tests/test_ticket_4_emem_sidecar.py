import pickle
import tempfile
import unittest
from pathlib import Path

from graph_memory.analysis import external_inspector, predict_utility


class MockIGraph:
    def __init__(self, n_vertices: int, edgelist: list[tuple[int, int]]):
        self._n = n_vertices
        self._edges = edgelist

    def vcount(self) -> int:
        return self._n

    def get_edgelist(self) -> list[tuple[int, int]]:
        return self._edges


class TestTicket4EmemSidecar(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_emem_igraph_conversion_and_metrics(self):
        # Create mock igraph Session-EDU-Argument graph
        # 0: Session node, 1-3: EDUs, 4-6: Arguments
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 5)]
        mock_ig = MockIGraph(n_vertices=7, edgelist=edges)

        cache_file = self.dir_path / "emem_graph.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(mock_ig, f)

        simple_g = external_inspector.extract_emem_graph(cache_file)
        self.assertEqual(simple_g.number_of_nodes(), 7)
        self.assertEqual(simple_g.number_of_edges(), len(edges))
        self.assertTrue(simple_g.has_edge(0, 1))
        self.assertTrue(simple_g.has_edge(4, 5))

    def test_emem_structural_gain_pairing_with_flat_control(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        mock_ig = MockIGraph(n_vertices=4, edgelist=edges)

        cache_file = self.dir_path / "emem_graph.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(mock_ig, f)

        ppr_qa = [
            {
                "sample_id": "0",
                "question": "What is the answer?",
                "answer": "Gold",
                "evidence": ["D1:1"],
                "context_ids": ["D1:1"],
                "F1": 0.90,
                "category": 1,
            }
        ]
        flat_qa = [
            {
                "sample_id": "0",
                "question": "What is the answer?",
                "answer": "Gold",
                "evidence": ["D1:1"],
                "context_ids": ["D1:2"],
                "F1": 0.50,
                "category": 1,
            }
        ]

        run_ppr = external_inspector.inspect_upstream_run(
            framework="emem_g",
            graph_artifacts_by_sample={"0": cache_file},
            qa_results=ppr_qa,
            benchmark="locomo",
            k=10,
            n=10,
        )
        run_flat = external_inspector.inspect_upstream_run(
            framework="emem_flat",
            graph_artifacts_by_sample={"0": cache_file},
            qa_results=flat_qa,
            benchmark="locomo",
            k=10,
            n=10,
        )

        self.assertEqual(run_ppr["mode"], "memory_emem_g")
        self.assertEqual(run_flat["mode"], "memory_emem_flat")

        # Collect observations across both runs
        obs = predict_utility.collect_sample_observations([run_ppr, run_flat], natural_only=True)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["mode"], "memory_emem_g")
        self.assertAlmostEqual(obs[0]["F1"], 0.90)
        self.assertAlmostEqual(obs[0]["delta_f1_vs_flat"], 0.40)
        self.assertAlmostEqual(obs[0]["delta_cov_vs_flat"], 1.0)


if __name__ == "__main__":
    unittest.main()
