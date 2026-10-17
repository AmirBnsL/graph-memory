"""Scoring + reporting.

Metrics (all reported TOGETHER, per the DolphinBench requirement):
    F1   token-level, SQuAD-style with LoCoMo-style normalization (primary)
    EM   normalized exact match
    plus per-category breakdown, latency, and prompt size.

Judge scores are deliberately NOT the headline: an independent audit found a
popular LoCoMo judge config accepted 62.8% of deliberately wrong answers. We
report F1/EM first, and add >=2 judges later with per-judge reporting.

Run:
    python src/eval.py --mode vector --k 10 --n-conv 1 --n-qa 50
    python src/eval.py --mode full   --n-conv 1 --n-qa 20
"""
from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

import answer as answer_mod
import llm
from load import load_loco

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

_ARTICLES = {"a", "an", "the"}


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(" + "|".join(_ARTICLES) + r")\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def run(mode: str, k: int, n_conv: int | None, n_qa: int | None,
        categories: set[int] | None) -> dict:
    convs = load_loco(limit=n_conv)
    rows = []
    t_start = time.time()
    for conv in convs:
        for qa in conv.qa:
            if categories and qa.category not in categories:
                continue
            res = answer_mod.answer(conv, qa, mode=mode, k=k)
            rows.append({
                "sample_id": conv.sample_id,
                "category": qa.category,
                "question": qa.question,
                "gold": qa.answer,
                "pred": res.answer,
                "F1": f1(res.answer, qa.answer),
                "EM": em(res.answer, qa.answer),
                "latency_s": res.latency_s,
                "prompt_chars": res.prompt_chars,
                "evidence": qa.evidence,
                "context_ids": res.context_ids,
            })
            if n_qa and len(rows) >= n_qa:
                break
            print(f"  [{len(rows)}] cat{qa.category} F1={rows[-1]['F1']:.2f} "
                  f"{qa.question[:60]!r} -> {res.answer[:40]!r}", flush=True)
        if n_qa and len(rows) >= n_qa:
            break

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["F1"])

    cfg = llm.resolve_config()
    summary = {
        "mode": mode,
        "k": k,
        "provider": cfg["provider"],
        "model": cfg["model"],
        "n": len(rows),
        "F1": sum(r["F1"] for r in rows) / len(rows) if rows else 0.0,
        "EM": sum(r["EM"] for r in rows) / len(rows) if rows else 0.0,
        "F1_by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        "n_by_category": {c: len(v) for c, v in sorted(by_cat.items())},
        "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows) if rows else 0.0,
        "total_latency_s": time.time() - t_start,
        "mean_prompt_chars": sum(r["prompt_chars"] for r in rows) / len(rows) if rows else 0.0,
        "rows": rows,
    }
    return summary


def save(summary: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model = summary["model"].replace("/", "-")
    p = RESULTS / f"{stamp}_{summary['mode']}_k{summary['k']}_{model}.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "vector"], default="vector")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-conv", type=int, default=None)
    ap.add_argument("--n-qa", type=int, default=None)
    ap.add_argument("--categories", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cats = set(args.categories) if args.categories else None
    s = run(args.mode, args.k, args.n_conv, args.n_qa, cats)
    path = save(s)

    print("\n" + "=" * 60)
    print(f"mode={s['mode']} k={s['k']} model={s['provider']}/{s['model']} n={s['n']}")
    print(f"  F1 = {s['F1']:.3f}   EM = {s['EM']:.3f}")
    print(f"  F1 by category: " + "  ".join(f"c{c}={v:.3f}(n={s['n_by_category'][c]})"
                                            for c, v in s["F1_by_category"].items()))
    print(f"  mean latency = {s['mean_latency_s']:.2f}s | mean prompt = {s['mean_prompt_chars']:.0f} chars")
    print(f"  saved -> {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
