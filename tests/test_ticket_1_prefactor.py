import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.memory.vector.flat import FlatMemory
from graph_memory.memory.graph.hipporag import HippoRAGMemory
from graph_memory.memory.graph.temporal_bteg import BiTemporalMemory
from graph_memory.cli import run_metrics, run_eval
from graph_memory.analysis import graph_metrics, predict_utility


class TestTicket1Prefactor(unittest.TestCase):
    def test_memory_backends_ensure_embedder_no_bare_import_failure(self):
        # Initializing backends without embedder should not fail on _ensure_embedder
        flat = FlatMemory(embedder=None)
        flat._ensure_embedder()
        self.assertIsNotNone(flat.embedder)

        hippo = HippoRAGMemory(embedder=None)
        hippo._ensure_embedder()
        self.assertIsNotNone(hippo.embedder)

        bteg = BiTemporalMemory(embedder=None)
        bteg._ensure_embedder()
        self.assertIsNotNone(bteg.embedder)

    def test_cli_metrics_entrypoint(self):
        self.assertTrue(hasattr(graph_metrics, "main"))
        self.assertTrue(callable(graph_metrics.main))

    def test_predict_utility_main_import(self):
        self.assertTrue(hasattr(predict_utility, "main"))
        self.assertTrue(callable(predict_utility.main))

    def test_hydra_memory_submodes(self):
        from hydra import compose, initialize_config_dir
        conf_dir = str(ROOT / "conf")
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            for mode_name in [
                "memory/flat",
                "memory/hipporag",
                "memory/hipporag_flattened",
                "memory/hipporag_rewired",
                "memory/temporal_bteg",
            ]:
                cfg = compose("config", overrides=[f"mode={mode_name}"])
                self.assertEqual(cfg.mode.name, "memory")


if __name__ == "__main__":
    unittest.main()
