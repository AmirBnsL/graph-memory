import unittest
from pathlib import Path
from collections import Counter
import sys

# Ensure src is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load import load_longmemeval, Conversation, Turn, QA, LONGMEMEVAL_CATEGORIES

class TestLoadLongMemEval(unittest.TestCase):
    def test_load_longmemeval_oracle_sample(self):
        convs = load_longmemeval(limit=5, split="oracle")
        self.assertEqual(len(convs), 5)
        for c in convs:
            self.assertIsInstance(c, Conversation)
            self.assertTrue(len(c.turns) > 0)
            self.assertEqual(len(c.qa), 1)
            qa = c.qa[0]
            self.assertIsInstance(qa, QA)
            self.assertTrue(bool(qa.question))
            self.assertIn(qa.category, {1, 2, 3, 4, 5})
            if c.sample_id.endswith("_abs"):
                self.assertEqual(qa.category, 5)
                self.assertEqual(qa.adversarial_answer, qa.answer)
            for t in c.turns:
                self.assertIsInstance(t, Turn)
                self.assertTrue(bool(t.dia_id))
                self.assertIn(t.speaker, {"user", "assistant"})

    def test_load_longmemeval_categories_distribution(self):
        convs = load_longmemeval(split="oracle")
        self.assertEqual(len(convs), 500)
        cats = Counter(c.qa[0].category for c in convs)
        # 5 abilities:
        # 1: extraction (150)
        # 2: multi-session (121)
        # 3: temporal (127)
        # 4: update (72)
        # 5: abstention (30)
        self.assertEqual(cats[1], 150)
        self.assertEqual(cats[2], 121)
        self.assertEqual(cats[3], 127)
        self.assertEqual(cats[4], 72)
        self.assertEqual(cats[5], 30)
        self.assertEqual(sum(cats.values()), 500)

if __name__ == "__main__":
    unittest.main()
