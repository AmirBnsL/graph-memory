import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.upstream.build_longmemeval_50 import build_stratified_subset, convert_longmemeval_to_locomo_format


class TestTicket2LongMemEval50(unittest.TestCase):
    def test_stratified_subset_counts_and_determinism(self):
        oracle_file = ROOT / "data/longmemeval/longmemeval_oracle.json"
        self.assertTrue(oracle_file.exists(), "longmemeval_oracle.json must exist")

        rows_1, prov_1 = build_stratified_subset(oracle_file, seed=42, samples_per_category=10)
        rows_2, prov_2 = build_stratified_subset(oracle_file, seed=42, samples_per_category=10)

        self.assertEqual(len(rows_1), 50)
        self.assertEqual(len(rows_2), 50)
        self.assertEqual(prov_1["question_ids"], prov_2["question_ids"])

        counts = prov_1["category_counts"]
        self.assertEqual(len(counts), 5)
        for cat, cnt in counts.items():
            self.assertEqual(cnt, 10, f"Category {cat} should have exactly 10 samples")

    def test_locomo_format_conversion(self):
        oracle_file = ROOT / "data/longmemeval/longmemeval_oracle.json"
        rows, _ = build_stratified_subset(oracle_file, seed=42, samples_per_category=2)
        locomo_samples = convert_longmemeval_to_locomo_format(rows)

        self.assertEqual(len(locomo_samples), len(rows))
        for sample in locomo_samples:
            self.assertIn("sample_id", sample)
            self.assertIn("question_id", sample)
            self.assertIn("conversation", sample)
            self.assertIn("qa", sample)
            self.assertEqual(len(sample["qa"]), 1)
            self.assertIn("question", sample["qa"][0])
            self.assertIn("answer", sample["qa"][0])
            self.assertIn("category", sample["qa"][0])
            conv = sample["conversation"]
            self.assertIn("speaker_a", conv)
            self.assertIn("speaker_b", conv)
            self.assertTrue(any(k.startswith("session_1") for k in conv.keys()))


if __name__ == "__main__":
    unittest.main()
