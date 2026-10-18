"""Loaders + normalization for LoCoMo (primary secondary benchmark) and LongMemEval.

Everything downstream -- retrieval, answering, scoring, and later the H1 ablations
and H3 structural statistics -- consumes the normalized objects defined here.
Swapping benchmarks must never require touching the harness.

Provenance (do not edit lightly):
  LoCoMo: github.com/snap-research/locomo  commit 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 (2024-08-12)
  The HF dataset `adymaharana/locomo` is EMPTY -- do not use load_dataset() for LoCoMo.

Category 5 is ADVERSARIAL: the question's premise is false (in 75% of the
nameable ones it attributes a fact to the wrong speaker). Its `answer` field is
empty BY DESIGN and the real content lives in `adversarial_answer`. Loading both
is not optional -- dropping it silently turns 446/1986 questions unscorable.

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

# ---------------------------------------------------------------------------
# QA category labels -- VERIFIED against the paper 2026-10-18.
#
# The paper's section 4.1 *enumerates* the five reasoning types as
#     1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain, 5=adversarial
# but the RELEASED locomo10.json does NOT use that order. Two independent
# checks fix the real mapping:
#   (a) structure -- cat1 is 98% multi-evidence-turn, cat2 is 77% "When ...",
#       cat3 is "Would ... likely" commonsense, cat4 is 95% single-turn,
#       cat5 is 100% `adversarial_answer`;
#   (b) proportions -- our per-category shares track paper Table 5 within a few
#       points (multi-hop 14.2% vs 14.6%, open-domain 4.8% vs 3.9%,
#       adversarial 22.5% vs 24.9%); cat1-as-single-hop is off by 22 points.
# Getting this wrong mislabels every per-category number in the ledger.
# Evidence: vault note 03-benchmarks/LoCoMo (2402.17753).md, section F2.
# ---------------------------------------------------------------------------
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# Cat 5: false-premise questions whose `answer` is empty BY DESIGN. Single source
# of truth -- eval.py imports this rather than re-declaring it.
ADVERSARIAL_CATEGORY = 5


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
    # Category 5 only: the reply a system gives if it falls for the false
    # premise. `answer` is empty for these BY DESIGN -- see eval.py.
    adversarial_answer: str = ""


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
                adversarial_answer=str(q.get("adversarial_answer", "")),
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
