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
                      A *floor*, NOT the paper's convention. Use it only for
                      comparison against numbers known to have scored c5 as 0
                      (e.g. Memory-R2's published RAG baseline).
    F1_paper          adversarial scored as correct-rejection. This IS the
                      LoCoMo convention -- the one comparable to Table 2/3.
    F1_answerable     token F1 over categories 1-4 only.
    refusal_accuracy  fraction of adversarial questions correctly declined.
    recall@k          gold-evidence retrieval (paper reports this too, section 4.1).

WHAT THE PAPER ACTUALLY DID (verified 2026-10-18, section 4.1 + Tables 2-3): it
scores adversarial with the same F1 and INCLUDES it in the Overall column, where
it reports Human=89.4. A human cannot score 89.4 token-F1 against an empty gold
unless a correct refusal counts as a match -- so the paper's adversarial column is
effectively refusal accuracy, and F1_paper reproduces that convention. Section 6.1
corroborates: GPT-3.5-16K's adversarial F1 "drops to a mere 2.1%" with a long
context, i.e. it stops declining.

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

import hydra
from pathlib import Path

try:
    import graph_memory.pipeline.answer as answer_mod
    from graph_memory.core import config as config_mod
    from graph_memory.core import embed, llm, manifest
    from graph_memory.pipeline.load import (
        ADVERSARIAL_CATEGORY, CATEGORY_NAMES, LONGMEMEVAL_CATEGORIES,
        load_loco, load_longmemeval
    )
except ImportError:
    import answer as answer_mod
    import config as config_mod
    import embed
    import llm
    import manifest
    from load import (
        ADVERSARIAL_CATEGORY, CATEGORY_NAMES, LONGMEMEVAL_CATEGORIES,
        load_loco, load_longmemeval
    )

def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "conf").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[3]

ROOT = _find_repo_root()
RESULTS = ROOT / "results"

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


def recall(evidence: list[str], context_ids: list[str]) -> dict:
    """Gold-evidence retrieval -- the paper's second QA metric (section 4.1).

    "Each QA sample is also annotated with the turn IDs in the conversation logs
    that contain the answer. We report the accuracy of retrieving the correct
    context for RAG models." The paper does not pin the exact definition, so all
    three readings are reported rather than one guess:
        recall_any         >=1 gold evidence turn is in the retrieved context
        recall_all         every gold evidence turn is in the retrieved context
        evidence_coverage  mean fraction of gold turns retrieved
    Free to compute: every stored row already carries `evidence` + `context_ids`.
    """
    ev = [e for e in (evidence or []) if e]
    if not ev:
        return {"recall_any": None, "recall_all": None, "evidence_coverage": None}
    ctx = set(context_ids or [])
    hit = sum(1 for e in ev if e in ctx)
    return {
        "recall_any": float(hit > 0),
        "recall_all": float(hit == len(ev)),
        "evidence_coverage": hit / len(ev),
    }


def run(mode: str, k: int, n_conv: int | None, n_qa: int | None,
        categories: set[int] | None, reader: dict | None = None,
        retriever: str | None = None, benchmark: str = "locomo",
        mode_cfg: dict | None = None,
        sample_per_category: int | None = None,
        split: str = "oracle") -> dict:
    loaders = {"locomo": load_loco, "longmemeval": load_longmemeval}
    if benchmark not in loaders:
        raise ValueError(f"unknown benchmark {benchmark!r}; have {sorted(loaders)}")
    if benchmark == "longmemeval":
        convs = load_longmemeval(limit=n_conv, split=split, sample_per_category=sample_per_category)
    else:
        convs = loaders[benchmark](limit=n_conv)
    rows = []
    t_start = time.time()
    identity = llm.resolve_identity(reader)
    if mode == "memory":
        answer_mod.clear_memory_cache()
    for conv in convs:
        for qa in conv.qa:
            if categories and qa.category not in categories:
                continue
            res = answer_mod.answer(conv, qa, mode=mode, k=k,
                                    reader=reader, retriever=retriever,
                                    mode_cfg=mode_cfg)
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
                **recall(qa.evidence, res.context_ids),
            })
            if n_qa and len(rows) >= n_qa:
                break
            print(f"  [{len(rows)}] cat{qa.category} "
                  f"{'REF' if adv else f'F1={rows[-1]['F1']:.2f}'} "
                  f"{qa.question[:60]!r} -> {res.answer[:40]!r}", flush=True)
        if n_qa and len(rows) >= n_qa:
            break

    run_mode_name = mode
    if mode == "memory" and mode_cfg:
        b_name = mode_cfg.get("backend", "flat")
        if mode_cfg.get("flatten_structure"):
            b_name += "_flattened"
        elif mode_cfg.get("rewire_structure"):
            b_name += "_rewired"
        run_mode_name = f"memory_{b_name}"

    summary = summarize(rows, mode=run_mode_name, k=k, t_start=t_start, identity=identity,
                        benchmark=benchmark)
    if mode == "memory" and mode_cfg and mode_cfg.get("extractor_model"):
        summary["extractor_model"] = str(mode_cfg["extractor_model"])
    # The retriever MUST be recorded: make_tables keys rows on (mode, retriever,
    # k), so a missing key would make runs from different retrievers collide and
    # silently overwrite each other in the ledger -- exactly the comparison we
    # need to keep. group_name() unwraps the config group to its name; the
    # fallback is the built-in default for pre-config results.
    summary["retriever"] = config_mod.group_name(retriever) or embed.MODEL_NAME

    if mode == "memory":
        try:
            from graph_memory.analysis import graph_metrics
        except ImportError:
            import graph_metrics
        cached = answer_mod.get_cached_memories()
        if cached:
            by_sample: dict[str, dict] = {}
            metrics_list: list[dict] = []
            for cache_key, mem in cached.items():
                if hasattr(mem, "export_graph"):
                    g = mem.export_graph()
                    if g is not None:
                        m = graph_metrics.compute_graph_metrics(g)
                        sample_id = cache_key[0] if isinstance(cache_key, tuple) else str(cache_key)
                        by_sample[sample_id] = m
                        metrics_list.append(m)
            if metrics_list:
                avg_metrics = {}
                for k_m in metrics_list[0]:
                    avg_metrics[k_m] = sum(m[k_m] for m in metrics_list) / len(metrics_list)
                summary["graph_metrics"] = avg_metrics
                summary["graph_metrics_by_sample"] = by_sample

    return summary


def _identity() -> tuple[str | None, str | None]:
    """Provider/model for the record. Rescoring must work with no LLM configured,
    so a missing config degrades to None instead of aborting the rescore."""
    return llm.resolve_identity()


def summarize(rows: list[dict], mode: str, k: int, t_start: float = 0.0,
              identity: tuple[str | None, str | None] | None = None,
              benchmark: str = "locomo") -> dict:
    """Metrics for one run. Always reports every convention, never just one."""
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["F1"])

    ans = [r for r in rows if not r.get("adversarial")]
    adv = [r for r in rows if r.get("adversarial")]
    retr = [r for r in rows if r.get("recall_any") is not None]
    provider, model = identity if identity is not None else _identity()

    # Paper convention: adversarial counts as correct when declined. This is the
    # number comparable to LoCoMo Tables 2-3 (see the module docstring).
    paper = [
        (float(bool(r.get("refusal_correct"))) if r.get("adversarial") else r["F1"])
        for r in rows
    ]
    recap = defaultdict(list)
    for r in retr:
        recap[r["category"]].append(r["recall_any"])

    cat_mapping = LONGMEMEVAL_CATEGORIES if benchmark == "longmemeval" else CATEGORY_NAMES

    return {
        "benchmark": benchmark,
        "mode": mode,
        "k": k,
        "provider": provider,
        "model": model,
        "n": len(rows),
        # Convention A -- adversarial scored 0. A FLOOR, not the paper's number.
        "F1": sum(r["F1"] for r in rows) / len(rows) if rows else 0.0,
        "EM": sum(r["EM"] for r in rows) / len(rows) if rows else 0.0,
        # Convention B -- the LoCoMo paper's convention (section 4.1 + Tables 2-3).
        "F1_paper": sum(paper) / len(paper) if paper else None,
        # Convention C -- the honest number on questions that have an answer.
        "F1_answerable": (sum(r["F1"] for r in ans) / len(ans)) if ans else None,
        "n_answerable": len(ans),
        # Convention D -- adversarial, scored as refusal detection.
        "refusal_accuracy": (
            sum(r["refusal_correct"] for r in adv) / len(adv)
        ) if adv and any(r["refusal_correct"] is not None for r in adv) else None,
        "n_adversarial": len(adv),
        # Retrieval (paper section 4.1). Questions with no gold evidence are excluded.
        "recall_any": (sum(r["recall_any"] for r in retr) / len(retr)) if retr else None,
        "recall_all": (sum(r["recall_all"] for r in retr) / len(retr)) if retr else None,
        "evidence_coverage": (
            sum(r["evidence_coverage"] for r in retr) / len(retr)
        ) if retr else None,
        "n_with_evidence": len(retr),
        # Retrieval-to-response decomposition (MERIT Ignore Rate & LoCoMo-Conv Silent Grounding)
        "F1_given_hit": (
            sum(r["F1"] for r in ans if r.get("recall_any") == 1.0)
            / len([r for r in ans if r.get("recall_any") == 1.0])
        ) if any(r.get("recall_any") == 1.0 for r in ans) else None,
        "ignore_rate": (
            sum(1 for r in ans if r.get("recall_any") == 1.0 and r["F1"] == 0.0)
            / len([r for r in ans if r.get("recall_any") == 1.0])
        ) if any(r.get("recall_any") == 1.0 for r in ans) else None,
        "F1_given_miss": (
            sum(r["F1"] for r in ans if r.get("recall_any") == 0.0)
            / len([r for r in ans if r.get("recall_any") == 0.0])
        ) if any(r.get("recall_any") == 0.0 for r in ans) else None,
        "recall_any_by_category": {c: sum(v) / len(v) for c, v in sorted(recap.items())},
        "F1_by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        "n_by_category": {c: len(v) for c, v in sorted(by_cat.items())},
        "category_names": {str(c): cat_mapping.get(c, "?") for c in sorted(by_cat)},
        "mean_latency_s": sum(r["latency_s"] for r in rows) / len(rows) if rows else 0.0,
        "total_latency_s": time.time() - t_start,
        "mean_prompt_chars": sum(r["prompt_chars"] for r in rows) / len(rows) if rows else 0.0,
        "rows": rows,
    }


def save(summary: dict, suffix: str = "") -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model = (summary.get("model") or "unknown").replace("/", "-")
    ext_model = summary.get("extractor_model")
    ext_tag = ""
    if ext_model and ext_model != summary.get("model"):
        ext_tag = f"_ext-{str(ext_model).replace('/', '-')}"
    benchmark = summary.get("benchmark", "locomo")
    prefix = f"{stamp}_{benchmark}_" if benchmark != "locomo" else f"{stamp}_"
    p = RESULTS / f"{prefix}{summary['mode']}_k{summary['k']}_{model}{ext_tag}{suffix}.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return p


def report(s: dict) -> None:
    print("\n" + "=" * 60)
    b_name = s.get("benchmark", "locomo")
    ext_info = f" extractor={s['extractor_model']}" if s.get("extractor_model") else ""
    print(f"benchmark={b_name} mode={s['mode']} k={s['k']} model={s['provider']}/{s['model']}{ext_info} n={s['n']}")
    print(f"  F1            (adversarial scored 0 -- floor)     = {s['F1']:.3f}")
    if s.get("F1_paper") is not None:
        c5_label = "c5=refusal" if b_name == "longmemeval" else "LoCoMo convention, c5=refusal"
        print(f"  F1_paper      ({c5_label})    = {s['F1_paper']:.3f}")
    print(f"  EM                                                = {s['EM']:.3f}")
    if s.get("F1_answerable") is not None:
        print(f"  F1_answerable (cat 1-4, n={s['n_answerable']})              = {s['F1_answerable']:.3f}")
    if s.get("refusal_accuracy") is not None:
        print(f"  refusal_acc   (adversarial, n={s['n_adversarial']})          = {s['refusal_accuracy']:.3f}")
    if s.get("recall_any") is not None:
        print(f"  recall@k      any={s['recall_any']:.3f}  all={s['recall_all']:.3f}  "
              f"coverage={s['evidence_coverage']:.3f}  (n={s['n_with_evidence']})")
    if s.get("F1_given_hit") is not None:
        f1_miss_str = f"{s['F1_given_miss']:.3f}" if s.get("F1_given_miss") is not None else "–"
        print(f"  retr->resp    F1|hit={s['F1_given_hit']:.3f}  "
              f"ignore_rate={s['ignore_rate']:.3f} (MERIT)  "
              f"F1|miss={f1_miss_str} (silent grounding)")
    print("  by category (verified mapping):")
    cat_mapping = LONGMEMEVAL_CATEGORIES if b_name == "longmemeval" else CATEGORY_NAMES
    recap = s.get("recall_any_by_category", {})
    for c, v in s["F1_by_category"].items():
        name = cat_mapping.get(int(c), "?")
        r = f"  recall_any={recap[c]:.3f}" if c in recap else ""
        print(f"    cat{c} {name:<14} F1={v:.3f}  (n={s['n_by_category'][c]}){r}")
    print(f"  mean latency = {s['mean_latency_s']:.2f}s | mean prompt = {s['mean_prompt_chars']:.0f} chars")
    print("=" * 60)


def find_existing(cfg_values: dict) -> Path | None:
    """A prior result this run would exactly reproduce, or None.

    Resume is a CONTENT-HASH decision, not a filename-pattern one: a run is
    skippable only when a prior manifest matches on config + code commit + env.
    Checking existence alone would silently return stale results after a prompt
    or retriever change -- which is the failure class this harness exists to
    prevent. Pre-refactor manifests carry no config hash, so they are never
    resume-eligible: they were produced by different code.
    """
    want = config_mod.hash_dict({k: v for k, v in cfg_values.items() if k != "hydra"})
    code = manifest.git_state()
    cur_code = code.get("code_commit") or code["commit"]
    env = manifest.env_freeze()["sha256"]
    for mf in sorted(RESULTS.glob("*.manifest.json")):
        try:
            m = json.loads(mf.read_text())
        except json.JSONDecodeError:
            continue
        if m.get("config", {}).get("sha256") != want:
            continue
        c = m.get("code", {})
        prev_code = c.get("code_commit") or c.get("commit")
        if prev_code != cur_code or bool(c.get("dirty")) != bool(code["dirty"]):
            continue
        if m.get("env", {}).get("sha256") != env:
            continue
        rf = RESULTS / m.get("result_file", "")
        if rf.exists():
            return rf
    return None


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg) -> None:
    values = config_mod.to_plain(cfg)
    limits = values.get("limits") or {}

    prev = find_existing(values)
    if prev:
        print(f"RESUME: an identical run already exists -- {prev.name}")
        print("        (config + code + env hashes all match; no LLM calls made)")
        d = json.loads(prev.read_text())
        d.pop("rows", None)
        report(d)
        return

    cats = limits.get("categories")
    mode = config_mod.group_name(values["mode"])
    mode_cfg = values.get("mode") if isinstance(values.get("mode"), dict) else {}
    benchmark = config_mod.group_name(values["benchmark"])
    benchmark_cfg = values.get("benchmark") if isinstance(values.get("benchmark"), dict) else {}
    split = benchmark_cfg.get("split", "oracle")
    sample_per_cat = limits.get("sample_per_category")
    s = run(mode, values["k"], limits.get("n_conv"), limits.get("n_qa"),
            set(cats) if cats else None,
            reader=values.get("reader"), retriever=values.get("retriever"),
            benchmark=benchmark, mode_cfg=mode_cfg,
            sample_per_category=sample_per_cat,
            split=split)
    path = save(s)
    mpath = manifest.write(
        manifest.build(values, path,
                       extra={"summary": {k: v for k, v in s.items() if k != "rows"}}),
        path,
    )
    report(s)
    print(f"  saved -> {path}\n  manifest -> {mpath}")


if __name__ == "__main__":
    main()
