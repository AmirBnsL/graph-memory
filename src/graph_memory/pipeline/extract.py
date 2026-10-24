"""Unified Extraction Pipeline for Agent Memory.

Converts dialogue turns into atomic factual propositions and OpenIE relation triples
with strict timestamp and provenance tracking.

Architecture:
  - Step 1: Named Entity Recognition (NER) following HippoRAG Appendix I (1-shot prompt).
  - Step 2: Open Information Extraction (OpenIE) generating RDF triples (head, relation, tail)
    and atomic factual propositions.
  - Caching: Every extraction is content-hashed and cached to disk (cache/extract/),
    so LLM extraction runs exactly once and is shared across all downstream memory backends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from graph_memory.core import llm
except ImportError:
    import llm

def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "conf").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[3]

ROOT = _find_repo_root()
DEFAULT_CACHE_DIR = ROOT / "cache" / "extract"

# HippoRAG Appendix I NER Prompt (1-shot)
NER_SYSTEM = "You are a very effective entity extraction system. Respond only with valid JSON."
NER_TEMPLATE = """Instruction:
Your task is to extract named entities and salient noun phrases from the given paragraph.
Respond with a JSON list of entities in the format: {{"named_entities": ["entity1", "entity2", ...]}}

One-Shot Demonstration:
Paragraph:
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com.
{{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]}}

Input:
Paragraph:
{text}
"""

# HippoRAG Appendix I OpenIE Prompt (1-shot) with atomic proposition generation
OPENIE_SYSTEM = "You are a precise Open Information Extraction system. Respond only with valid JSON."
OPENIE_TEMPLATE = """Instruction:
Your task is to construct relation triples (subject, relation, object) and clear atomic factual propositions from the given text and its entity list.
Requirements:
- Each triple should contain at least one, and preferably two, of the entities in the list.
- Clearly resolve all pronouns to their specific names.
- Provide a list of concise, standalone atomic factual propositions representing the facts in the text.

Respond with JSON format:
{{
    "triples": [
        ["subject", "predicate", "object"]
    ],
    "propositions": [
        "Concise atomic fact 1.",
        "Concise atomic fact 2."
    ]
}}

One-Shot Demonstration:
Paragraph:
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi and English songs.
Entities: ["Radio City", "India", "3 July 2001", "Hindi", "English"]
{{
    "triples": [
        ["Radio City", "is in", "India"],
        ["Radio City", "is", "private FM radio station"],
        ["Radio City", "started on", "3 July 2001"],
        ["Radio City", "plays", "Hindi songs"],
        ["Radio City", "plays", "English songs"]
    ],
    "propositions": [
        "Radio City is a private FM radio station in India.",
        "Radio City was started on 3 July 2001.",
        "Radio City plays songs in Hindi and English."
    ]
}}

Input:
Paragraph:
{text}
Entities: {entities_json}
"""


@dataclass
class MemoryRecord:
    id: str
    sample_id: str
    turn_id: int | str
    speaker: str
    timestamp: str | None
    fact_text: str
    entities: list[str]
    triples: list[tuple[str, str, str]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["triples"] = [list(t) for t in self.triples]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryRecord:
        triples = [tuple(t) for t in d.get("triples", [])]
        return cls(
            id=d["id"],
            sample_id=d["sample_id"],
            turn_id=d["turn_id"],
            speaker=d["speaker"],
            timestamp=d.get("timestamp"),
            fact_text=d["fact_text"],
            entities=d.get("entities", []),
            triples=triples,
        )


def _clean_json_str(raw: str) -> str:
    s = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if match:
        s = match.group(1).strip()
    return s


def parse_ner_response(raw: str) -> list[str]:
    cleaned = _clean_json_str(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            ents = data.get("named_entities", [])
            if isinstance(ents, list):
                return [str(e).strip() for e in ents if e]
        elif isinstance(data, list):
            return [str(e).strip() for e in data if e]
    except Exception:
        pass
    return []


def parse_openie_response(raw: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    cleaned = _clean_json_str(raw)
    try:
        data = json.loads(cleaned)
        triples: list[tuple[str, str, str]] = []
        raw_triples = data.get("triples", [])
        if isinstance(raw_triples, list):
            for t in raw_triples:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    triples.append((str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()))

        props = data.get("propositions", [])
        if not isinstance(props, list) or not props:
            # Fallback to reconstructing from triples if propositions are absent
            props = [f"{s} {p} {o}." for s, p, o in triples]
        return triples, [str(p).strip() for p in props if p]
    except Exception:
        pass
    return [], []


def hash_extraction_key(model: str, prompt_template: str, text: str) -> str:
    payload = f"{model}|{prompt_template}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def extract_turn(
    text: str,
    speaker: str,
    timestamp: str | None,
    sample_id: str,
    turn_id: int | str,
    overrides: dict | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[MemoryRecord]:
    """Extract atomic propositions and OpenIE triples from a single dialogue turn.
    
    Results are cached to disk so repeated runs or multiple downstream backends
    pay zero additional LLM costs.
    """
    if not text.strip():
        return []

    cfg = llm.resolve_config(overrides)
    model_name = cfg["model"]

    # Cache key covers turn content and extraction prompts
    h = hash_extraction_key(model_name, NER_TEMPLATE + OPENIE_TEMPLATE, f"{speaker}: {text}")
    cache_path = cache_dir / f"{h}.json"

    if cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text())
            return [MemoryRecord.from_dict(r) for r in cached_data["records"]]
        except Exception:
            pass

    # Step 1: Named Entity Recognition
    ner_prompt = NER_TEMPLATE.format(text=f"{speaker}: {text}")
    ner_resp = llm.complete(ner_prompt, system=NER_SYSTEM, overrides=overrides)
    entities = parse_ner_response(ner_resp)

    # Step 2: OpenIE Triples + Propositions
    openie_prompt = OPENIE_TEMPLATE.format(
        text=f"{speaker}: {text}",
        entities_json=json.dumps(entities, ensure_ascii=False),
    )
    openie_resp = llm.complete(openie_prompt, system=OPENIE_SYSTEM, overrides=overrides)
    triples, propositions = parse_openie_response(openie_resp)

    # If extraction produced no propositions, fallback to the raw utterance
    if not propositions:
        propositions = [f"{speaker}: {text}"]

    records: list[MemoryRecord] = []
    for idx, prop in enumerate(propositions):
        rec_id = f"{sample_id}_{turn_id}_{idx}"
        records.append(
            MemoryRecord(
                id=rec_id,
                sample_id=sample_id,
                turn_id=turn_id,
                speaker=speaker,
                timestamp=timestamp,
                fact_text=prop,
                entities=entities,
                triples=triples,
            )
        )

    # Save to disk cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({
            "sample_id": sample_id,
            "turn_id": turn_id,
            "hash": h,
            "records": [r.to_dict() for r in records],
        }, indent=2, ensure_ascii=False)
    )

    return records


def extract_conversation(conv: Any, overrides: dict | None = None, workers: int = 4) -> list[MemoryRecord]:
    """Extract all turns in a Conversation object."""
    if not conv.turns:
        return []

    def _do_turn(turn):
        t_id = getattr(turn, "dia_id", getattr(turn, "turn_id", 0))
        ts = getattr(turn, "date_time", getattr(turn, "timestamp", None))
        return extract_turn(
            text=turn.text,
            speaker=turn.speaker,
            timestamp=ts,
            sample_id=conv.sample_id,
            turn_id=t_id,
            overrides=overrides,
        )

    if workers <= 1 or len(conv.turns) == 1:
        all_records: list[MemoryRecord] = []
        for turn in conv.turns:
            all_records.extend(_do_turn(turn))
        return all_records

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_do_turn, conv.turns))

    all_records = []
    for turn_recs in results:
        all_records.extend(turn_recs)
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract conversation facts and triples")
    parser.add_argument("--benchmark", choices=["locomo", "longmemeval"], default="longmemeval")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of conversations")
    parser.add_argument("--sample-per-cat", type=int, default=None, help="Balanced sample per category")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent worker threads per conversation")
    parser.add_argument("--model", type=str, default=None, help="Extractor LLM model override (e.g. qwen2.5:14b-instruct)")
    parser.add_argument("--provider", type=str, default=None, help="Extractor LLM provider override (e.g. ollama)")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.provider:
        overrides["provider"] = args.provider

    from load import load_loco, load_longmemeval
    if args.benchmark == "longmemeval":
        convs = load_longmemeval(limit=args.limit, sample_per_category=args.sample_per_cat)
    else:
        convs = load_loco(limit=args.limit)

    model_label = args.model or "default"
    print(f"Extracting {len(convs)} conversation(s) from {args.benchmark} (model={model_label}, workers={args.workers})...")
    total_records = 0
    for idx, conv in enumerate(convs, 1):
        records = extract_conversation(conv, overrides=overrides or None, workers=args.workers)
        total_records += len(records)
        print(f"[{idx}/{len(convs)}] Conversation {conv.sample_id}: {len(conv.turns)} turns -> {len(records)} memory records")

    print(f"Done. Extracted {total_records} memory records total.")


if __name__ == "__main__":
    main()
