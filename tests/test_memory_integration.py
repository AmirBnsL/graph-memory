import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.pipeline import eval as eval_mod
from graph_memory.pipeline.load import Conversation, QA, Turn


class TestMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.conv = Conversation(
            sample_id="test_conv_int",
            speaker_a="User",
            speaker_b="Assistant",
            turns=[
                Turn(dia_id="t0", speaker="User", text="I bought a blue BMW in May 2023.", session_id=0, date_time="2023-05-01"),
                Turn(dia_id="t1", speaker="Assistant", text="Nice! Do you like driving it?", session_id=0, date_time="2023-05-01"),
                Turn(dia_id="t2", speaker="User", text="Yes, but I moved to Berlin in June.", session_id=0, date_time="2023-06-01"),
            ],
            qa=[
                QA(
                    question="What car did I buy?",
                    answer="blue BMW",
                    category=1,
                    evidence=["t0"],
                ),
                QA(
                    question="Where did I move?",
                    answer="Berlin",
                    category=4,
                    evidence=["t2"],
                ),
            ],
        )

    @patch("graph_memory.pipeline.eval.load_longmemeval")
    @patch("graph_memory.core.llm.complete")
    def test_memory_modes_smoke(self, mock_complete, mock_loader):
        mock_loader.return_value = [self.conv]

        # Return mock JSON for NER/OpenIE and answer
        def mock_llm_response(prompt, system=None, **kwargs):
            if "entity extraction" in (system or "").lower():
                return '{"named_entities": ["User", "blue BMW", "Berlin"]}'
            if "open information extraction" in (system or "").lower():
                return '{"triples": [["User", "bought", "blue BMW"], ["User", "moved to", "Berlin"]], "propositions": ["User bought a blue BMW.", "User moved to Berlin."]}'
            return "blue BMW"

        mock_complete.side_effect = mock_llm_response

        # Test Flat Memory
        s_flat = eval_mod.run(
            mode="memory",
            k=2,
            n_conv=1,
            n_qa=2,
            categories=None,
            benchmark="longmemeval",
            mode_cfg={"backend": "flat"},
            reader={"provider": "ollama", "model": "mock-qwen"},
        )
        self.assertEqual(s_flat["mode"], "memory_flat")
        self.assertEqual(len(s_flat["rows"]), 2)
        self.assertGreater(s_flat["F1_answerable"], 0.0)
        self.assertIsNone(s_flat.get("graph_metrics"))

        # Test HippoRAG Memory
        s_hippo = eval_mod.run(
            mode="memory",
            k=2,
            n_conv=1,
            n_qa=2,
            categories=None,
            benchmark="longmemeval",
            mode_cfg={"backend": "hipporag"},
            reader={"provider": "ollama", "model": "mock-qwen"},
        )
        self.assertEqual(s_hippo["mode"], "memory_hipporag")
        self.assertEqual(len(s_hippo["rows"]), 2)
        self.assertIsNotNone(s_hippo.get("graph_metrics"))
        self.assertIn("largest_component_ratio", s_hippo["graph_metrics"])

        # Test HippoRAG Flattened (Structure Ablation)
        s_hippo_flat = eval_mod.run(
            mode="memory",
            k=2,
            n_conv=1,
            n_qa=2,
            categories=None,
            benchmark="longmemeval",
            mode_cfg={"backend": "hipporag", "flatten_structure": True},
            reader={"provider": "ollama", "model": "mock-qwen"},
        )
        self.assertEqual(s_hippo_flat["mode"], "memory_hipporag_flattened")
        self.assertEqual(len(s_hippo_flat["rows"]), 2)

        # Test Bi-Temporal Memory
        s_temp = eval_mod.run(
            mode="memory",
            k=2,
            n_conv=1,
            n_qa=2,
            categories=None,
            benchmark="longmemeval",
            mode_cfg={"backend": "temporal_bteg"},
            reader={"provider": "ollama", "model": "mock-qwen"},
        )
        self.assertEqual(s_temp["mode"], "memory_temporal_bteg")
        self.assertEqual(len(s_temp["rows"]), 2)
        self.assertIsNotNone(s_temp.get("graph_metrics"))
        self.assertIn("temporal_edge_ratio", s_temp["graph_metrics"])

        # Test HippoRAG Rewired + Extractor Model Decoupling
        s_rewired = eval_mod.run(
            mode="memory",
            k=2,
            n_conv=1,
            n_qa=2,
            categories=None,
            benchmark="longmemeval",
            mode_cfg={
                "backend": "hipporag",
                "rewire_structure": True,
                "extractor_model": "mock-qwen-14b",
            },
            reader={"provider": "ollama", "model": "mock-qwen"},
        )
        self.assertEqual(s_rewired["mode"], "memory_hipporag_rewired")
        self.assertEqual(s_rewired.get("extractor_model"), "mock-qwen-14b")
        self.assertEqual(s_rewired.get("model"), "mock-qwen")


if __name__ == "__main__":
    unittest.main()
