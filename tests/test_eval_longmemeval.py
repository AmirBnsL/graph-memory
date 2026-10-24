import unittest
from unittest.mock import patch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.pipeline.load import load_longmemeval, LONGMEMEVAL_CATEGORIES
from graph_memory.pipeline.eval import summarize, run, find_existing
from graph_memory.pipeline.answer import AnswerResult


class TestEvalLongMemEval(unittest.TestCase):
    def test_longmemeval_summary_category_names(self):
        convs = load_longmemeval(limit=5, split="oracle")
        rows = []
        for c in convs:
            qa = c.qa[0]
            rows.append({
                "sample_id": c.sample_id,
                "category": qa.category,
                "adversarial": qa.category == 5,
                "question": qa.question,
                "gold": qa.answer,
                "adversarial_answer": qa.adversarial_answer,
                "pred": "test prediction",
                "F1": 0.0 if qa.category == 5 else 0.5,
                "EM": 0.0,
                "refusal_correct": False if qa.category == 5 else None,
                "latency_s": 0.1,
                "prompt_chars": 100,
                "evidence": qa.evidence,
                "context_ids": [],
                "recall_any": 0.0,
                "recall_all": 0.0,
                "evidence_coverage": 0.0,
            })
        summary = summarize(rows, mode="vector", k=2, t_start=0.0,
                            identity=("test_prov", "test_model"),
                            benchmark="longmemeval")
        self.assertEqual(summary["benchmark"], "longmemeval")
        self.assertIn("category_names", summary)
        # Check that categories map to LongMemEval names
        for cat_id_str, name in summary["category_names"].items():
            expected = LONGMEMEVAL_CATEGORIES.get(int(cat_id_str))
            self.assertEqual(name, expected)

    @patch("graph_memory.pipeline.eval.answer_mod.answer")
    def test_eval_longmemeval_run_dry(self, mock_answer):
        mock_answer.return_value = AnswerResult("mock answer", ["loc_0"], 100, 0.1)
        summary = run(
            mode="vector",
            k=2,
            n_conv=2,
            n_qa=2,
            categories=None,
            retriever="qwen3-06b",
            benchmark="longmemeval",
        )
        self.assertEqual(summary["benchmark"], "longmemeval")
        self.assertEqual(summary["n"], 2)
        self.assertIn("F1_paper", summary)
        self.assertIn("recall_any", summary)
        self.assertEqual(len(summary["rows"]), 2)

    def test_hydra_config_longmemeval(self):
        from omegaconf import OmegaConf
        cfg_bench = OmegaConf.load(ROOT / "conf" / "benchmark" / "longmemeval.yaml")
        self.assertEqual(cfg_bench.name, "longmemeval")
        self.assertEqual(cfg_bench.loader, "longmemeval")
        self.assertEqual(cfg_bench.split, "oracle")

        cfg_reader = OmegaConf.load(ROOT / "conf" / "reader" / "qwen25-7b.yaml")
        self.assertEqual(cfg_reader.num_ctx, 32768)


if __name__ == "__main__":
    unittest.main()
