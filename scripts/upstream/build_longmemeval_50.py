#!/usr/bin/env python3
"""Build deterministic 50-conversation stratified LongMemEval subset.

Outputs:
1. Native LongMemEval format: data/longmemeval/longmemeval_s_50.json
2. LoCoMo format: data/longmemeval/longmemeval_s_50_locomo_fmt.json
3. Provenance record: provenance/longmemeval_50_subset.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent


def build_stratified_subset(
    input_file: Path,
    seed: int = 42,
    samples_per_category: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by question_type
    by_cat = defaultdict(list)
    for row in data:
        q_type = str(row.get("question_type") or "unknown")
        by_cat[q_type].append(row)

    # Canonical list of 5 primary categories (collapse single-session variants to 'single-session' if needed)
    # The 6 types in LongMemEval:
    # 'temporal-reasoning', 'multi-session', 'knowledge-update',
    # 'single-session-user', 'single-session-assistant', 'single-session-preference'
    # We select 10 from each of the 5 standard categories:
    # 1. multi-session (10)
    # 2. temporal-reasoning (10)
    # 3. knowledge-update (10)
    # 4. single-session (user + assistant + preference) (10)
    # 5. abstention / unanswerable (or balance across 5 primary buckets)
    # If the input has specific types:
    target_buckets = [
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-user",
        "single-session-assistant",
    ]

    rng = random.Random(seed)
    chosen_rows: list[dict[str, Any]] = []
    category_summary: dict[str, int] = {}

    for bucket in target_buckets:
        pool = sorted(by_cat[bucket], key=lambda x: str(x.get("question_id")))
        picked = rng.sample(pool, min(samples_per_category, len(pool)))
        chosen_rows.extend(picked)
        category_summary[bucket] = len(picked)

    # Sort chosen rows deterministically by question_id
    chosen_rows.sort(key=lambda x: str(x.get("question_id")))

    provenance = {
        "source_file": str(input_file.relative_to(ROOT) if input_file.is_relative_to(ROOT) else input_file.name),
        "total_selected": len(chosen_rows),
        "seed": seed,
        "category_counts": category_summary,
        "question_ids": [r["question_id"] for r in chosen_rows],
    }

    return chosen_rows, provenance


def convert_longmemeval_to_locomo_format(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert LongMemEval rows to LoCoMo-compatible format for A-MEM."""
    locomo_samples = []

    for idx, row in enumerate(rows):
        sample_id = str(idx)
        qid = str(row.get("question_id"))
        q_type = str(row.get("question_type"))
        question = str(row.get("question"))
        answer = str(row.get("answer"))

        haystack_sessions = row.get("haystack_sessions") or []
        haystack_dates = row.get("haystack_dates") or []

        conversation: dict[str, Any] = {
            "speaker_a": "User",
            "speaker_b": "Assistant",
        }

        turn_counter = 1
        for s_idx, session_turns in enumerate(haystack_sessions):
            s_num = s_idx + 1
            date_str = haystack_dates[s_idx] if s_idx < len(haystack_dates) else "2023-01-01"
            conversation[f"session_{s_num}_date_time"] = date_str

            formatted_turns = []
            for t in session_turns:
                role = t.get("role", "user")
                speaker = "User" if role == "user" else "Assistant"
                dia_id = f"D{sample_id}:{turn_counter}"
                formatted_turns.append({
                    "speaker": speaker,
                    "dia_id": dia_id,
                    "text": t.get("content", ""),
                })
                turn_counter += 1

            conversation[f"session_{s_num}"] = formatted_turns

        # Category mapping for LoCoMo schema:
        # 1: multi-hop, 2: temporal, 3: open-domain / knowledge-update, 4: single-hop, 5: adversarial
        cat_code = 1
        if "temporal" in q_type:
            cat_code = 2
        elif "update" in q_type:
            cat_code = 3
        elif "single" in q_type:
            cat_code = 4

        locomo_samples.append({
            "sample_id": sample_id,
            "question_id": qid,
            "conversation": conversation,
            "qa": [
                {
                    "question": question,
                    "answer": answer,
                    "category": cat_code,
                    "evidence": [],
                }
            ],
            "event_summary": [],
            "observation": {},
            "session_summary": {},
        })

    return locomo_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/longmemeval/longmemeval_oracle.json"))
    parser.add_argument("--out-native", default=str(ROOT / "data/longmemeval/longmemeval_s_50.json"))
    parser.add_argument("--out-locomo", default=str(ROOT / "data/longmemeval/longmemeval_s_50_locomo_fmt.json"))
    parser.add_argument("--out-provenance", default=str(ROOT / "provenance/longmemeval_50_subset.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-cat", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    native_rows, prov = build_stratified_subset(input_path, seed=args.seed, samples_per_category=args.samples_per_cat)

    # Compute sha256 of native output
    native_json_str = json.dumps(native_rows, indent=2, ensure_ascii=False)
    native_sha = hashlib.sha256(native_json_str.encode("utf-8")).hexdigest()
    prov["sha256"] = native_sha

    # Save native
    native_out = Path(args.out_native)
    native_out.parent.mkdir(parents=True, exist_ok=True)
    native_out.write_text(native_json_str, encoding="utf-8")

    # Save LoCoMo format
    locomo_rows = convert_longmemeval_to_locomo_format(native_rows)
    locomo_out = Path(args.out_locomo)
    locomo_out.parent.mkdir(parents=True, exist_ok=True)
    locomo_out.write_text(json.dumps(locomo_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save provenance
    prov_out = Path(args.out_provenance)
    prov_out.parent.mkdir(parents=True, exist_ok=True)
    prov_out.write_text(json.dumps(prov, indent=2), encoding="utf-8")

    print(f"Generated {len(native_rows)} samples across {len(prov['category_counts'])} categories.")
    print(f"Native: {native_out} ({native_sha[:16]}...)")
    print(f"LoCoMo fmt: {locomo_out}")
    print(f"Provenance: {prov_out}")


if __name__ == "__main__":
    main()
