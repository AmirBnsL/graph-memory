import pickle
import tempfile
import unittest
from pathlib import Path

from graph_memory.analysis import external_inspector, predict_utility


class MockMemoryNote:
    def __init__(self, note_id: str, content: str, links: list[int]):
        self.id = note_id
        self.content = content
        self.links = links


class TestTicket5AmemSidecar(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_amem_memory_note_pickle_extraction_and_metrics(self):
        # Create mock notes with Zettelkasten associative links
        notes = {
            "n0": MockMemoryNote("n0", "Turn 0 text", links=[1, 2]),
            "n1": MockMemoryNote("n1", "Turn 1 text", links=[0, 3]),
            "n2": MockMemoryNote("n2", "Turn 2 text", links=[0]),
            "n3": MockMemoryNote("n3", "Turn 3 text", links=[1]),
            "n4": MockMemoryNote("n4", "Turn 4 text", links=[]),
        }

        cache_file = self.dir_path / "memory_cache_sample_0.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(notes, f)

        simple_g = external_inspector.extract_amem_graph(cache_file)
        self.assertEqual(simple_g.number_of_nodes(), 5)
        self.assertTrue(simple_g.has_edge(0, 1))
        self.assertTrue(simple_g.has_edge(0, 2))
        self.assertTrue(simple_g.has_edge(1, 3))
        self.assertFalse(simple_g.has_edge(0, 4))

    def test_amem_run_end_to_end_through_h2_seam(self):
        notes = {
            "0": {"id": "0", "links": [1]},
            "1": {"id": "1", "links": [0]},
        }
        cache_file = self.dir_path / "memory_cache.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(notes, f)

        mock_qa = [
            {
                "sample_id": "0",
                "question": "What is A-MEM?",
                "answer": "Agentic memory",
                "evidence": ["D1:1"],
                "context_ids": ["D1:1"],
                "F1": 0.95,
                "category": 1,
            }
        ]

        run_dict = external_inspector.inspect_upstream_run(
            framework="amem",
            graph_artifacts_by_sample={"0": cache_file},
            qa_results=mock_qa,
            benchmark="locomo",
            k=10,
            n=10,
        )

        self.assertEqual(run_dict["mode"], "memory_amem")
        obs = predict_utility.collect_sample_observations([run_dict], natural_only=True)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["mode"], "memory_amem")
        self.assertAlmostEqual(obs[0]["F1"], 0.95)
        self.assertIn("degree_gini", obs[0])
        self.assertIn("largest_component_ratio", obs[0])


if __name__ == "__main__":
    unittest.main()
