import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from extract import MemoryRecord, extract_turn, parse_ner_response, parse_openie_response, hash_extraction_key
from unittest.mock import patch


class TestExtract(unittest.TestCase):
    def test_parse_ner_response(self):
        raw = '{"named_entities": ["Alice", "Paris", "3 July 2021"]}'
        entities = parse_ner_response(raw)
        self.assertEqual(entities, ["Alice", "Paris", "3 July 2021"])

        # Markdown json wrapping
        raw_md = '```json\n{"named_entities": ["Bob", "London"]}\n```'
        entities_md = parse_ner_response(raw_md)
        self.assertEqual(entities_md, ["Bob", "London"])

        # Fallback on malformed
        self.assertEqual(parse_ner_response("garbage"), [])

    def test_parse_openie_response(self):
        raw = '''{
            "triples": [
                ["Alice", "moved to", "Paris"],
                ["Alice", "bought", "a car"]
            ],
            "propositions": [
                "Alice moved to Paris.",
                "Alice bought a car."
            ]
        }'''
        triples, props = parse_openie_response(raw)
        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0], ("Alice", "moved to", "Paris"))
        self.assertEqual(triples[1], ("Alice", "bought", "a car"))
        self.assertEqual(len(props), 2)
        self.assertEqual(props[0], "Alice moved to Paris.")

    def test_hash_extraction_key(self):
        k1 = hash_extraction_key("modelA", "prompt1", "Hello world")
        k2 = hash_extraction_key("modelA", "prompt1", "Hello world")
        k3 = hash_extraction_key("modelB", "prompt1", "Hello world")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(len(k1), 32)

    @patch("extract.llm.complete")
    def test_extract_turn_cached(self, mock_complete):
        # Mock responses for NER and OpenIE
        mock_complete.side_effect = [
            '{"named_entities": ["Dave", "BMW"]}',
            '{"triples": [["Dave", "owns", "BMW"]], "propositions": ["Dave owns a BMW."]}'
        ]
        records = extract_turn(
            text="Dave bought a BMW yesterday.",
            speaker="User",
            timestamp="2023-05-10",
            sample_id="test_conv_1",
            turn_id=0,
            overrides={"model": "test-model", "provider": "ollama"},
            cache_dir=ROOT / "cache" / "test_extract",
        )
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.sample_id, "test_conv_1")
        self.assertEqual(rec.turn_id, 0)
        self.assertEqual(rec.speaker, "User")
        self.assertEqual(rec.timestamp, "2023-05-10")
        self.assertEqual(rec.fact_text, "Dave owns a BMW.")
        self.assertEqual(rec.entities, ["Dave", "BMW"])
        self.assertEqual(rec.triples, [("Dave", "owns", "BMW")])

        # Second call should read from disk cache, not call mock_complete again
        records2 = extract_turn(
            text="Dave bought a BMW yesterday.",
            speaker="User",
            timestamp="2023-05-10",
            sample_id="test_conv_1",
            turn_id=0,
            overrides={"model": "test-model", "provider": "ollama"},
            cache_dir=ROOT / "cache" / "test_extract",
        )
        self.assertEqual(mock_complete.call_count, 2)  # Still 2, not 4
        self.assertEqual(len(records2), 1)
        self.assertEqual(records2[0].fact_text, "Dave owns a BMW.")

        # Clean up test cache
        test_cache = ROOT / "cache" / "test_extract"
        if test_cache.exists():
            for f in test_cache.glob("*"):
                f.unlink()
            test_cache.rmdir()


if __name__ == "__main__":
    unittest.main()
