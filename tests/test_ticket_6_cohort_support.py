import unittest
from graph_memory.analysis import predict_utility


class TestTicket6CohortSupport(unittest.TestCase):
    def test_cohort_filtering(self):
        runs = [
            {
                "benchmark": "locomo",
                "mode": "memory_hipporag",
                "extractor_model": "qwen2.5:7b-instruct",
                "k": 10,
                "n": 10,
                "graph_metrics_by_sample": {"0": {"num_nodes": 10, "degree_gini": 0.4}},
                "rows": [{"sample_id": "0", "question": "Q1", "F1": 0.8, "evidence_coverage": 0.8}],
            },
            {
                "benchmark": "locomo",
                "mode": "memory_magma",
                "extractor_model": "qwen2.5:7b-instruct",
                "k": 10,
                "n": 10,
                "graph_metrics_by_sample": {"0": {"num_nodes": 15, "degree_gini": 0.5}},
                "rows": [{"sample_id": "0", "question": "Q2", "F1": 0.85, "evidence_coverage": 0.9}],
            },
        ]

        obs_all = predict_utility.collect_sample_observations(runs, natural_only=True, cohort="all")
        obs_a = predict_utility.collect_sample_observations(runs, natural_only=True, cohort="part_a")
        obs_b = predict_utility.collect_sample_observations(runs, natural_only=True, cohort="part_b")

        self.assertEqual(len(obs_all), 2)
        self.assertEqual(len(obs_a), 1)
        self.assertEqual(obs_a[0]["mode"], "memory_hipporag")
        self.assertEqual(len(obs_b), 1)
        self.assertEqual(obs_b[0]["mode"], "memory_magma")


if __name__ == "__main__":
    unittest.main()
