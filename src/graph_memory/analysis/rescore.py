#!/usr/bin/env python3
"""Re-score stored predictions without re-running the model.

Split out of eval.py because it needs no experiment config: the rows are the
evidence, the metrics are just a function of them, so scoring changes must
never require paying for the LLM again.

    python src/rescore.py results/<run>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from graph_memory.pipeline import eval as eval_mod
    from graph_memory.pipeline.load import ADVERSARIAL_CATEGORY
except ImportError:
    import eval as eval_mod
    from load import ADVERSARIAL_CATEGORY

def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "conf").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[3]

ROOT = _find_repo_root()
RESULTS = ROOT / "results"


def rescore(path: Path) -> Path:
    d = json.loads(path.read_text())
    for r in d["rows"]:
        adv = r.get("category") == ADVERSARIAL_CATEGORY
        r["adversarial"] = adv
        r["F1"] = 0.0 if adv else eval_mod.f1(r["pred"], r["gold"])
        r["EM"] = 0.0 if adv else eval_mod.em(r["pred"], r["gold"])
        r["refusal_correct"] = eval_mod.is_refusal(r["pred"]) if adv else None
        r.update(eval_mod.recall(r.get("evidence"), r.get("context_ids")))

    s = eval_mod.summarize(d["rows"], mode=d["mode"], k=d["k"])
    s["rescored_from"] = path.name
    s["created"] = d.get("created")
    s["total_latency_s"] = d.get("total_latency_s")  # keep the original run's wall clock
    # Carry the identity forward from the file being rescored -- no LLM needed here.
    s["provider"] = d.get("provider") or s.get("provider")
    s["model"] = d.get("model") or s.get("model")
    out = eval_mod.save(s, suffix="_rescored")

    man = path.with_suffix(".manifest.json")
    if man.exists():
        m = json.loads(man.read_text())
        m["rescored_from"] = path.name
        m["note"] = ("metrics recomputed from stored rows (paper-convention F1, "
                     "refusal accuracy, recall@k, verified category labels); "
                     "predictions unchanged")
        out.with_suffix(".manifest.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False))

    eval_mod.report(s)
    print(f"  rescored -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=str, help="path to a results JSON (or its manifest)")
    args = ap.parse_args()

    p = Path(args.result)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists() and p.suffix == ".json":
        alt = p.with_suffix(".manifest.json")
        if alt.exists():
            p = alt
    if not p.exists():
        raise SystemExit(f"no such result: {args.result}")
    if p.name.endswith(".manifest.json"):
        p = p.with_name(p.name.replace(".manifest.json", ".json"))
    rescore(p)


if __name__ == "__main__":
    main()
