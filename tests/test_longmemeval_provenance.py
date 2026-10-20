import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

class TestLongMemEvalProvenance(unittest.TestCase):
    def test_longmemeval_dataset_pinned(self):
        pin_file = ROOT / "provenance" / "datasets.json"
        self.assertTrue(pin_file.exists(), f"Provenance file {pin_file} missing")
        pins = json.loads(pin_file.read_text())
        self.assertIn("longmemeval", pins)
        cfg = pins["longmemeval"]
        self.assertEqual(cfg.get("status"), "verified", f"Status is {cfg.get('status')}, expected 'verified'")
        self.assertIsNotNone(cfg.get("sha256"), "sha256 hash missing")
        self.assertEqual(len(cfg["sha256"]), 64, f"Invalid sha256 hash length: {len(cfg['sha256'])}")
        dpath = ROOT / cfg["local_path"]
        self.assertTrue(dpath.exists(), f"Dataset file {dpath} missing")
        hasher = hashlib.sha256()
        with open(dpath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        self.assertEqual(hasher.hexdigest(), cfg["sha256"], f"Hash mismatch: got {hasher.hexdigest()}, expected {cfg['sha256']}")

if __name__ == "__main__":
    unittest.main()
