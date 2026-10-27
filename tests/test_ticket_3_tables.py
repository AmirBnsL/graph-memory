import unittest
from pathlib import Path
from graph_memory.analysis import make_tables

ROOT = Path(__file__).resolve().parent.parent


class TestTicket3Tables(unittest.TestCase):
    def test_glossary_contains_all_h1_substrates(self):
        glossary_text = (ROOT / "GLOSSARY.md").read_text()
        self.assertIn("**Vector Baseline**:", glossary_text)
        self.assertIn("**Flat Memory**:", glossary_text)
        self.assertTrue(
            "**Concept Graph**:" in glossary_text or "**Concept Graph (Canonical)**:" in glossary_text,
            "Concept graph canonical entry missing from GLOSSARY.md",
        )
        self.assertIn("**Flattened Graph**:", glossary_text)
        self.assertIn("**Rewired Graph**:", glossary_text)
        self.assertIn("**Bi-Temporal Event Graph (BTEG)**:", glossary_text)

    def test_adr_0001_exists(self):
        adr_file = ROOT / "docs" / "adr" / "0001-h1-three-way-decomposition.md"
        self.assertTrue(adr_file.exists())
        content = adr_file.read_text()
        self.assertIn("Three-Way Hypothesis 1 Decomposition", content)
        self.assertIn("Delta_{\\text{extraction}}", content)
        self.assertIn("Delta_{\\text{entity\\_bottleneck}}", content)
        self.assertIn("Delta_{\\text{topology}}", content)

    def test_table_generation_canonical_labels(self):
        runs = make_tables.load_runs()
        self.assertGreater(len(runs), 0)
        # Verify run grouping works without error
        groups = make_tables._group_h1_runs(runs)
        self.assertGreater(len(groups), 0)


if __name__ == "__main__":
    unittest.main()
