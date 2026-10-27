import unittest
from pathlib import Path

from graph_memory.core import llm
from graph_memory.analysis import predict_utility


class TestTicket1Prefactor(unittest.TestCase):
    def test_vllm_preset_resolution(self):
        cfg = llm.resolve_config({"provider": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct"})
        self.assertEqual(cfg["provider"], "vllm")
        self.assertEqual(cfg["base_url"], "http://localhost:8000/v1")
        self.assertEqual(cfg["api_key"], "vllm")
        self.assertEqual(cfg["model"], "Qwen/Qwen2.5-7B-Instruct")

    def test_natural_modes_includes_part_b(self):
        self.assertIn("memory_magma", predict_utility.NATURAL_MODES)
        self.assertIn("memory_emem_g", predict_utility.NATURAL_MODES)
        self.assertIn("memory_amem", predict_utility.NATURAL_MODES)

    def test_collect_sample_observations_with_part_b_and_native_emem_flat(self):
        mock_runs = [
            {
                "benchmark": "locomo",
                "mode": "memory_emem_flat",
                "extractor_model": "Qwen/Qwen2.5-7B-Instruct",
                "k": 10,
                "n": 10,
                "rows": [
                    {"sample_id": "0", "question": "Where is Alice?", "F1": 0.60, "evidence_coverage": 0.50}
                ],
            },
            {
                "benchmark": "locomo",
                "mode": "memory_emem_g",
                "extractor_model": "Qwen/Qwen2.5-7B-Instruct",
                "k": 10,
                "n": 10,
                "graph_metrics_by_sample": {
                    "0": {
                        "num_nodes": 20,
                        "degree_gini": 0.45,
                        "max_degree_ratio": 5.2,
                        "largest_component_ratio": 0.95,
                        "edge_density": 0.08,
                        "bridge_edge_ratio": 0.12,
                    }
                },
                "rows": [
                    {
                        "sample_id": "0",
                        "question": "Where is Alice?",
                        "F1": 0.85,
                        "evidence_coverage": 0.80,
                        "evidence": ["0"],
                        "context_ids": ["0", "1"],
                    }
                ],
            },
            {
                "benchmark": "locomo",
                "mode": "memory_magma",
                "extractor_model": "Qwen/Qwen2.5-7B-Instruct",
                "k": 10,
                "n": 10,
                "graph_metrics_by_sample": {
                    "0": {
                        "num_nodes": 25,
                        "degree_gini": 0.50,
                        "max_degree_ratio": 6.1,
                        "largest_component_ratio": 0.90,
                        "edge_density": 0.07,
                        "bridge_edge_ratio": 0.15,
                    }
                },
                "rows": [
                    {
                        "sample_id": "0",
                        "question": "Where is Alice?",
                        "F1": 0.80,
                        "evidence_coverage": 0.75,
                        "evidence": ["0"],
                        "context_ids": ["0"],
                    }
                ],
            },
        ]

        obs = predict_utility.collect_sample_observations(mock_runs, natural_only=True)
        self.assertEqual(len(obs), 2)
        modes = {o["mode"] for o in obs}
        self.assertEqual(modes, {"memory_emem_g", "memory_magma"})

        emem_obs = next(o for o in obs if o["mode"] == "memory_emem_g")
        self.assertAlmostEqual(emem_obs["delta_f1_vs_flat"], 0.25)
        self.assertAlmostEqual(emem_obs["delta_cov_vs_flat"], 0.30)


if __name__ == "__main__":
    unittest.main()
