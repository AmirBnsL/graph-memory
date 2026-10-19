"""Retrieval-only probe: recall@k with ZERO LLM calls.

Why this exists
---------------
Choosing a retriever end-to-end costs a full sweep (hours of T4) and conflates
retrieval with reading. But recall is computable from embeddings + the gold
evidence ids alone -- no generation. So the whole retriever decision costs
minutes on a laptop CPU, and the "retriever is not the culprit" claim becomes a
measured result instead of an assumption.

What it reports, per k in the grid:
    recall_any   >=1 gold evidence turn retrieved   (paper §4.1's headline)
    recall_all   ALL gold evidence turns retrieved  (multi-hop's real bar)
    ...both overall and per category, plus n_with_evidence.

Provenance: the resolved config (retriever spec included) is hashed and stored
with the run, alongside the code commit -- same rule as a full run's manifest.
Output files carry `_probe` in the name, which make_tables skips.

Run:
    python src/recall_probe.py                       # minilm regression check
    python src/recall_probe.py retriever=qwen3-06b
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import config as config_mod
import embed
import hydra
import numpy as np
from load import CATEGORY_NAMES, load_loco

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def probe(convs, k_grid: list[int], s: dict,
          categories: set[int] | None = None) -> dict:
    """Recall@k over all questions with gold evidence."""
    hits_any: dict[int, list] = defaultdict(list)
    hits_all: dict[int, list] = defaultdict(list)
    n_with_evidence = 0

    for conv in convs:
        texts = embed.passage_texts(conv.turns, s)
        if not texts:
            continue
        emb = embed.embed(texts, s)  # cached after the first pass

        qas = [qa for qa in conv.qa
               if (not categories or qa.category in categories)
               and any(e for e in (qa.evidence or []) if e)]
        if not qas:
            continue
        # Batched query encoding: one call per conversation. Single-query
        # encoding is overhead-bound on CPU (~2.6 s/query for a 0.6B model)
        # while batched it is ~100x faster. Same embeddings, same ranking.
        qvecs = embed.encode([s["query_prefix"] + qa.question for qa in qas], s,
                             batch_size=max(1, s["batch_size"]))
        n_with_evidence += len(qas)

        for qa, q in zip(qas, qvecs):
            ev = [e for e in (qa.evidence or []) if e]
            sims = emb @ q
            ranked = [conv.turns[i].dia_id
                      for i in np.argsort(-sims)]
            gold = set(ev)
            for k in k_grid:
                got = gold & set(ranked[:k])
                hits_any[k].append((qa.category, float(bool(got))))
                hits_all[k].append((qa.category, float(got == gold)))

    def agg(pairs) -> dict:
        overall = sum(v for _, v in pairs) / len(pairs) if pairs else None
        by_cat = defaultdict(list)
        for c, v in pairs:
            by_cat[c].append(v)
        return {
            "overall": overall,
            "by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        }

    return {
        "retriever": s["name"],
        "spec": s,
        "k_grid": k_grid,
        "n_with_evidence": n_with_evidence,
        "recall": {str(k): {"any": agg(hits_any[k]), "all": agg(hits_all[k])}
                   for k in k_grid},
    }


@hydra.main(config_path="../conf", config_name="recall", version_base=None)
def main(cfg) -> None:
    values = config_mod.to_plain(cfg)
    s = embed.spec(values["retriever"])
    k_grid = sorted(int(k) for k in (values.get("k_grid") or [10]))
    limits = values.get("limits") or {}
    cats = limits.get("categories")

    convs = load_loco(limit=limits.get("n_conv"))

    t0 = time.time()
    out = probe(convs, k_grid, s, set(cats) if cats else None)
    out["benchmark"] = values.get("benchmark", "locomo")
    out["limits"] = limits
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["code_commit"] = _git_commit()
    out["config_sha256"] = config_mod.hash_dict(values)
    # drop any QA outside limits.categories by recomputing? -- simpler: filter
    # at the source when categories is set
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = s["name"].replace("/", "_").replace("all-", "").lower()
    path = RESULTS / f"{stamp}_recallprobe_{safe}.json"
    path.write_text(json.dumps(out, indent=2))

    _report(out)
    print(f"  saved -> {path}")


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, cwd=ROOT, timeout=10,
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return None


def _report(out: dict) -> None:
    name = out["retriever"]
    print(f"\n=== recall@k -- {name} (n_with_evidence={out['n_with_evidence']}) ===\n")
    cats = sorted(CATEGORY_NAMES)
    print("| k | " + " | ".join(f"{CATEGORY_NAMES[c]}" for c in cats)
          + " | ANY overall | ALL overall |")
    print("|---|" + "---|" * (len(cats) + 2))
    for k in out["k_grid"]:
        r = out["recall"][str(k)]
        cells = []
        for c in cats:
            v = r["any"]["by_category"].get(c)
            cells.append("–" if v is None else f"{v:.3f}")
        print(f"| {k} | " + " | ".join(cells)
              + f" | {r['any']['overall']:.3f} | {r['all']['overall']:.3f} |")


if __name__ == "__main__":
    main()
