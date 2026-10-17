"""Loaders + normalization for LoCoMo (primary secondary benchmark) and LongMemEval.

Everything downstream -- retrieval, answering, scoring, and later the H1 ablations
and H3 structural statistics -- consumes the normalized objects defined here.
Swapping benchmarks must never require touching the harness.

Provenance (do not edit lightly):
  LoCoMo: github.com/snap-research/locomo  commit 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 (2024-08-12)
  The HF dataset `adymaharana/locomo` is EMPTY -- do not use load_dataset() for LoCoMo.

Run:  python src/load.py --stats
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCOMO_PATH = ROOT / "data" / "locomo" / "data" / "locomo10.json"

# LoCoMo QA category numbers as they literally appear in locom10.json.
# NOTE: the semantic mapping (which number means multi-hop / temporal / ...) is
# NOT yet verified against the paper -- treat labels as "TODO verify".
CATEGORY_SIZES = {4: 70, 5: 47, 2: 37, 1: 32, 3: 13}  # observed in conv-26


@dataclass
class Turn:
    dia_id: str
    speaker: str
    text: str
    session_id: int
    date_time: str


@dataclass
class QA:
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    category: int = -1


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[Turn] = field(default_factory=list)
    qa: list[QA] = field(default_factory=list)
    date_times: dict[int, str] = field(default_factory=dict)

    def turn_by_id(self, dia_id: str) -> Turn | None:
        for t in self.turns:
            if t.dia_id == dia_id:
                return t
        return None

    def full_context(self) -> str:
        """All turns rendered as one transcript -- the 'stuff everything' baseline."""
        lines = []
        for t in self.turns:
            lines.append(f"[{t.dia_id}] {t.date_time} | {t.speaker}: {t.text}")
        return "\n".join(lines)


def _session_indices(conv: dict) -> list[int]:
    idx = []
    for k in conv:
        m = re.fullmatch(r"session_(\d+)", k)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def load_loco(limit: int | None = None, path: Path | None = None) -> list[Conversation]:
    """Load LoCoMo conversations. `limit` caps the number of conversations."""
    p = path or LOCOMO_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"LoCoMo not found at {p}\n"
            "Run: git clone --depth 1 https://github.com/snap-research/locomo data/locomo"
        )
    data = json.loads(p.read_text())
    convs: list[Conversation] = []
    for c in data:
        conv = c["conversation"]
        idxs = _session_indices(conv)
        turns: list[Turn] = []
        dates: dict[int, str] = {}
        for i in idxs:
            dt = conv.get(f"session_{i}_date_time", "")
            dates[i] = dt
            for t in conv.get(f"session_{i}", []) or []:
                turns.append(
                    Turn(
                        dia_id=t.get("dia_id", ""),
                        speaker=t.get("speaker", ""),
                        text=t.get("text", ""),
                        session_id=i,
                        date_time=dt,
                    )
                )
        qa = [
            QA(
                question=q.get("question", ""),
                answer=str(q.get("answer", "")),
                evidence=list(q.get("evidence", []) or []),
                category=q.get("category", -1),
            )
            for q in c.get("qa", [])
        ]
        convs.append(
            Conversation(
                sample_id=c.get("sample_id", ""),
                speaker_a=conv.get("speaker_a", ""),
                speaker_b=conv.get("speaker_b", ""),
                turns=turns,
                qa=qa,
                date_times=dates,
            )
        )
        if limit is not None and len(convs) >= limit:
            break
    return convs


# --------------------------------------------------------------------------
# LongMemEval -- loader stub. Source from its official repo/paper distribution
# (NOT a random HF re-upload), and validate against the paper's own numbers.
# Verify the ground-truth arithmetic/mislabeling GitHub issues before trusting it.
# --------------------------------------------------------------------------
def load_longmemeval(*_args, **_kwargs):
    raise NotImplementedError(
        "LongMemEval loader pending: download the official release, then implement "
        "the same Conversation/QA normalization as load_loco()."
    )


def _stats() -> None:
    convs = load_loco()
    print(f"conversations: {len(convs)}")
    total_turns = sum(len(c.turns) for c in convs)
    total_qa = sum(len(c.qa) for c in convs)
    print(f"turns: {total_turns} | qa: {total_qa}")
    cats: Counter[int] = Counter()
    for c in convs:
        cats.update(q.category for q in c.qa)
    print("qa by category:", dict(sorted(cats.items())))
    for c in convs:
        print(
            f"  {c.sample_id}: {c.speaker_a}/{c.speaker_b} "
            f"sessions={len(c.date_times)} turns={len(c.turns)} qa={len(c.qa)}"
        )
    first = convs[0]
    print("\nexample:", first.qa[0])
    print("evidence turn:", first.turn_by_id(first.qa[0].evidence[0]) if first.qa[0].evidence else None)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.stats:
        _stats()
    else:
        ap.print_help()
