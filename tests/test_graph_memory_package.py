"""Verify the graph_memory package architecture, imports, and CLI."""
import subprocess
import sys
import unittest


class TestGraphMemoryPackage(unittest.TestCase):
    def test_top_level_package_exports(self):
        import graph_memory
        self.assertTrue(hasattr(graph_memory, "__version__"))
        self.assertTrue(hasattr(graph_memory, "BaseMemory"))
        self.assertTrue(hasattr(graph_memory, "FlatMemory"))
        self.assertTrue(hasattr(graph_memory, "HippoRAGMemory"))
        self.assertTrue(hasattr(graph_memory, "BiTemporalMemory"))

    def test_core_package_exports(self):
        from graph_memory.core import group_name, to_plain, complete, embed, build_manifest
        self.assertTrue(callable(group_name))
        self.assertTrue(callable(to_plain))
        self.assertTrue(callable(complete))
        self.assertTrue(callable(embed))
        self.assertTrue(callable(build_manifest))

    def test_pipeline_package_exports(self):
        from graph_memory.pipeline import (
            Conversation,
            QA,
            Turn,
            MemoryRecord,
            AnswerResult,
            load_longmemeval,
            load_loco,
            answer,
        )
        self.assertTrue(callable(load_longmemeval))
        self.assertTrue(callable(load_loco))
        self.assertTrue(callable(answer))

    def test_analysis_package_exports(self):
        from graph_memory.analysis import compute_graph_metrics
        self.assertTrue(callable(compute_graph_metrics))

    def test_cli_help(self):
        p = subprocess.run(
            [sys.executable, "-m", "graph_memory.cli", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0)
        self.assertIn("Graph Memory: Long-Term Conversational Memory", p.stdout)
        self.assertIn("run", p.stdout)
        self.assertIn("eval", p.stdout)
        self.assertIn("tables", p.stdout)
        self.assertIn("metrics", p.stdout)

    def test_colab_mgr_help(self):
        p = subprocess.run(
            [sys.executable, "-m", "graph_memory.colab_mgr", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0)
        self.assertIn("Reproducible Google Colab VM Operations Manager", p.stdout)
        self.assertIn("setup", p.stdout)
        self.assertIn("sweep", p.stdout)
        self.assertIn("pull", p.stdout)
