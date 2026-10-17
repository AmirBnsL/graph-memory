"""Scoring + reporting.

Metrics (all reported TOGETHER, per the DolphinBench requirement):
    F1   token-level, SQuAD-style with LoCoMo-style normalization (primary)
    EM   normalized exact match
    plus per-category breakdown, latency, and prompt size.

Judge scores are deliberately NOT the headline: an independent audit found a
popular LoCoMo judge config accepted 62.8% of deliberately wrong answers. We
report F1/EM first, and add >=2 judges later with per-judge reporting.

THE ADVERSARIAL CATEGORY -- the single most important scoring decision here.

LoCoMo category 5 questions have a false premise: in 75% of the ones that name a
speaker, the question attributes a fact to the WRONG speaker (the evidence turn
is spoken by the other person). `answer` is therefore empty BY DESIGN, and the
dataset supplies `adversarial_answer` instead -- the reply a fooled system gives.
The correct behaviour is to decline, so category 5 is scored by refusal
detection, not token overlap. Token F1 against an empty gold scores a *correct*
refusal 0.0, which silently punishes the most capable systems.

Three numbers are always reported, so the convention is never implicit:
    F1                token F1 over every question, adversarial scored 0.
                      This matches the convention the published numbers used --
                      it is what makes cross-paper comparison possible.
    F1_answerable     token F1 over categories 1-4 only.
    refusal_accuracy  fraction of adversarial questions correctly declined.

Run:
    python src/eval.py --mode vector --k 10 --n-conv 1 --n-qa 50
    python src/eval.py --mode full   --n-conv 1 --n-qa 20
    python src/eval.py --rescore results/<file>.json
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
import manifest
from load import load_loco

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# LoCoMo category 5: false-premise questions, `answer` empty by design.
ADVERSARIAL_CATEGORY = 5

_ARTICLES = {"a", "an", "the"}

_REFUSAL_RE = re.compile(
    r"\b(no information|not (?:mentioned|specified|available|stated|provided)"
    r"|cannot (?:be )?(?:determin|answer|find)|no answer|unknown"
    r"|i (?:don'?t|do not) (?:know|have))\b"
)
_REFUSAL_EXACT = {"no", "none", "n/a", "unknown", "nothing"}


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


def is_refusal(pred: str) -> bool:
    """True if the model declined to answer. Used ONLY for the adversarial
    category, where declining is the correct behaviour."""
    p = pred.strip().lower()
    if not p or p in _REFUSAL_EXACT:
        return True
    return bool(_REFUSAL_RE.search(p))


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
            adv = qa.category == ADVERSARIAL_CATEGORY
            rows.append({
                "sample_id": conv.sample_id,
                "category": qa.category,
                "adversarial": adv,
                "question": qa.question,
                "gold": qa.answer,
                "adversarial_answer": qa.adversarial_answer,
                "pred": res.answer,
                "F1": 0.0 if adv else f1(res.answer, qa.answer),
                "EM": 0.0 if adv else em(res.answer, qa.answer),
                "refusal_correct": is_refusal(res.answer) if adv else None,
                "latency_s": res.latency_s,
                "prompt_chars": res.prompt_chars,
                "evidence": qa.evidence,
                "context_ids": res.context_ids,
            })
            if n_qa and len(rows) >= n_qa:
                break
            print(f"  [{len(rows)}] cat{qa.category} "
                  f"{'REF' if adv else f'F1={rows[-1]['F1']:.2f}'} "
                  f"{qa.question[:60]!r} -> {res.answer[:40]!r}", flush=True)
        if n_qa and len(rows) >= n_qa:
            break

    return summarize(rows, mode=mode, k=k, t_start=t_start)


def summarize(rows: list[dict], mode: str, k: int, t_start: float = 0.0) -> dict:
    """Metrics for one run. Always reports all three conventions."""
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["F1"])

    ans = [r for r in rows if not r.get("adversarial")]
    adv = [r for r in rows if r.get("adversarial")]
    cfg = llm.resolve_config()

    return {
        "mode": mode,
        "k": k,
        "provider": cfg["provider"],
        "model": cfg["model"],
        "n": len(rows),
        # Convention A -- comparable to published LoCoMo F1s.
        "F1": sum(r["F1"] for r in rows) / len(rows) if rows else 0.0,
        "EM": sum(r["EM"] for r in rows) / len(rows) if rows else 0.0,
        # Convention B -- the honest number on questions that have an answer.
        "F1_answerable": (sum(r["F1"] for r in ans) / len(ans)) if ans else None,
        "n_answerable": len(ans),
        # Convention C -- adversarial, scored as refusal detection.
        "refusal_accuracy": (
            sum(r["refusal_correct"] for r in adv) / len(adv)
        ) if adv and any(r["refusal_correct"] is not None for r in adv) else None,
        "n_adversarial": len(adv),
        "F1_by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        "n_by_category": {c: len(v) for c, v in sorted(by_cat.items())},
        "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows) if rows else 0.0,
        "total_latency_s": time.time() - t_start,
        "mean_prompt_chars": sum(r["prompt_chars"] for r in rows) / len(rows) if rows else 0.0,
        "rows": rows,
    }


def save(summary: dict, suffix: str = "") -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model = summary["model"].replace("/", "-")
    p = RESULTS / f"{stamp}_{summary['mode']}_k{summary['k']}_{model}{suffix}.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return p


def report(s: dict) -> None:
    print("\n" + "=" * 60)
    print(f"mode={s['mode']} k={s['k']} model={s['provider']}/{s['model']} n={s['n']}")
    print(f"  F1 (all, published convention) = {s['F1']:.3f}   EM = {s['EM']:.3f}")
    if s.get("F1_answerable") is not None:
        print(f"  F1 answerable (cat 1-4, n={s['n_answerable']}) = {s['F1_answerable']:.3f}")
    if s.get("refusal_accuracy") is not None:
        print(f"  refusal accuracy (adversarial, n={s['n_adversarial']}) = {s['refusal_accuracy']:.3f}")
    print("  F1 by category: " + "  ".join(
        f"c{c}={v:.3f}(n={s['n_by_category'][c]})" for c, v in s["F1_by_category"].items()))
    print(f"  mean latency = {s['mean_latency_s']:.2f}s | mean prompt = {s['mean_prompt_chars']:.0f} chars")
    print("=" * 60)


def rescore(path: Path) -> Path:
    """Re-score stored predictions without re-running the model.

    Scoring changes must never require paying for the LLM again -- the rows are
    the evidence, the metrics are just a function of them.
    """
    d = json.loads(path.read_text())
    for r in d["rows"]:
        adv = r.get("category") == ADVERSARIAL_CATEGORY
        r["adversarial"] = adv
        r["F1"] = 0.0 if adv else f1(r["pred"], r["gold"])
        r["EM"] = 0.0 if adv else em(r["pred"], r["gold"])
        r["refusal_correct"] = is_refusal(r["pred"]) if adv else None

    s = summarize(d["rows"], mode=d["mode"], k=d["k"])
    s["rescored_from"] = path.name
    s["created"] = d.get("created")
    s["total_latency_s"] = d.get("total_latency_s")  # keep the original run's wall clock
    out = save(s, suffix="_rescored")

    man = path.with_suffix(".manifest.json")
    if man.exists():
        m = json.loads(man.read_text())
        m["rescored_from"] = path.name
        m["note"] = ("metrics recomputed from stored rows after the adversarial-"
                     "category scoring fix; predictions unchanged")
        out.with_suffix(".manifest.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False))

    report(s)
    print(f"  rescored -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "vector"], default="vector")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-conv", type=int, default=None)
    ap.add_argument("--n-qa", type=int, default=None)
    ap.add_argument("--categories", type=int, nargs="*", default=None)
    ap.add_argument("--rescore", type=str, default=None,
                    help="re-score an existing results JSON instead of running")
    args = ap.parse_args()

    if args.rescore:
        rescore(Path(args.rescore))
        return

    cats = set(args.categories) if args.categories else None
    s = run(args.mode, args.k, args.n_conv, args.n_qa, cats)
    path = save(s)
    mpath = manifest.write(
        manifest.build(
            args.mode, args.k, args.n_conv, args.n_qa, path,
            extra={"summary": {k: v for k, v in s.items() if k != "rows"}},
        ),
        path,
    )
    report(s)
    print(f"  saved -> {path}\n  manifest -> {mpath}")


if __name__ == "__main__":
    main()
