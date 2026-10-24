#!/usr/bin/env python3
"""Generate the ledger's tables from results/*.json.

Hand-editing results/INDEX.md is how prose numbers drift from the JSONs that
produced them -- the same failure class the thesis is about. This regenerates
the run tables from the committed result files, so the ledger can only be
stale, never wrong.

    python src/make_tables.py            # print markdown to stdout

Then paste into results/INDEX.md. Every row carries the file it came from, so
any number in the ledger is traceable in one step.
"""
from __future__ import annotations

from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CAT_NAMES = {
    "locomo": {
        1: "multi-hop",
        2: "temporal",
        3: "open-domain",
        4: "single-hop",
        5: "adversarial",
    },
    "longmemeval": {
        1: "extraction",
        2: "multi-session",
        3: "temporal",
        4: "update",
        5: "abstention",
    },
}


def _extractor_of(d: dict) -> str:
    """Return the extractor model for a memory run, or '' for non-memory runs."""
    mode = str(d.get("mode") or "")
    if not mode.startswith("memory_"):
        return ""
    return str(d.get("extractor_model") or d.get("model") or "")


def load_runs() -> list[dict]:
    """Latest result per (benchmark, mode, retriever, extractor, k, n).

    `n` is part of the key on purpose. The 25-QA balanced subset and the 500-QA
    full split are different measurements; collapsing them (the old key) silently
    replaced the full-split k=10 run with the newer subset run, so the ledger
    showed 0.553 for k=10 and 0.471 for k=5 as if they were one sweep.

    Derived artifacts are skipped: rescored files (same numbers, new name),
    recall probes, and smoke tests (n <= 2)."""
    runs: dict[tuple, dict] = {}
    for f in sorted(glob.glob(str(RESULTS / "*.json"))):
        p = Path(f)
        if p.name.endswith(".manifest.json") or "_probe" in p.name or "_rescored" in p.name:
            continue
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or "mode" not in d or "rows" not in d:
            continue
        n = int(d.get("n") or 0)
        if n <= 2:  # smoke test, not a benchmark run
            continue
        ext = _extractor_of(d)
        key = (
            d.get("benchmark", "locomo"),
            d.get("mode"),
            (d.get("retriever") or ""),
            ext,
            int(d.get("k") or 0),
            n,
        )
        d["_file"] = p.name
        runs[key] = d
    return [runs[k] for k in sorted(runs, key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5]))]


def fmt(v, nd=3) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def fmt_signed(v, nd=3) -> str:
    if v is None:
        return "–"
    return f"{v:+.{nd}f}"


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _paired_cluster_stats(
    run_a: dict,
    run_b: dict,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float | int | tuple[float, float] | None]:
    """Compute paired differences (A - B) on answerable questions (Cat 1-4).

    Uses 10,000-draw conversation-cluster bootstrap (resampling `sample_id`
    clusters with replacement) for 95% CIs on ΔF1_answerable, paired sign-flip
    permutation p-value on ΔF1, and exact McNemar p-values on EM and Recall@k.
    """
    rows_a = {
        (str(r.get("sample_id")), str(r.get("question"))): r
        for r in run_a.get("rows", [])
        if not r.get("adversarial")
    }
    rows_b = {
        (str(r.get("sample_id")), str(r.get("question"))): r
        for r in run_b.get("rows", [])
        if not r.get("adversarial")
    }
    common_keys = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    if not common_keys:
        return {"n_pairs": 0}

    f1_diffs: list[float] = []
    by_cluster: dict[str, list[float]] = defaultdict(list)
    em_b = 0  # A=1, B=0
    em_c = 0  # A=0, B=1
    rec_b = 0  # A hit, B miss
    rec_c = 0  # A miss, B hit

    for k_id in common_keys:
        ra, rb = rows_a[k_id], rows_b[k_id]
        diff = float(ra.get("F1", 0.0)) - float(rb.get("F1", 0.0))
        f1_diffs.append(diff)
        by_cluster[k_id[0]].append(diff)

        ema = 1 if float(ra.get("EM", 0.0)) >= 1.0 else 0
        emb = 1 if float(rb.get("EM", 0.0)) >= 1.0 else 0
        if ema == 1 and emb == 0:
            em_b += 1
        elif ema == 0 and emb == 1:
            em_c += 1

        ha = 1 if float(ra.get("recall_any") or 0.0) >= 1.0 else 0
        hb = 1 if float(rb.get("recall_any") or 0.0) >= 1.0 else 0
        if ha == 1 and hb == 0:
            rec_b += 1
        elif ha == 0 and hb == 1:
            rec_c += 1

    n_pairs = len(f1_diffs)
    mean_diff = sum(f1_diffs) / n_pairs

    # Conversation-cluster bootstrap for 95% CI
    rng = random.Random(seed)
    clusters = list(by_cluster.values())
    n_clusters = len(clusters)
    boot_means: list[float] = []
    for _ in range(n_boot):
        picked = [clusters[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        flat_vals = [x for cl in picked for x in cl]
        if flat_vals:
            boot_means.append(sum(flat_vals) / len(flat_vals))
    boot_means.sort()
    ci_low = boot_means[int(0.025 * len(boot_means))] if boot_means else mean_diff
    ci_high = boot_means[int(0.975 * len(boot_means))] if boot_means else mean_diff

    # Paired cluster sign-flip permutation p-value (two-sided) on ΔF1
    obs_abs = abs(mean_diff)
    exceed = 0
    for _ in range(n_boot):
        signed_sum = 0.0
        for cl in clusters:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            signed_sum += sign * sum(cl)
        if abs(signed_sum / n_pairs) >= obs_abs - 1e-12:
            exceed += 1
    p_perm_f1 = (exceed + 1) / (n_boot + 1)

    return {
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "delta_f1": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_perm_f1": p_perm_f1,
        "em_discordant": (em_b, em_c),
        "p_mcnemar_em": _mcnemar_exact_p(em_b, em_c),
        "rec_discordant": (rec_b, rec_c),
        "p_mcnemar_rec": _mcnemar_exact_p(rec_b, rec_c),
    }


def _group_h1_runs(runs: list[dict]) -> dict[tuple, dict[str, dict]]:
    """Group runs by (benchmark, retriever, extractor, k, n), attaching matching vector run."""
    vec_runs: dict[tuple, dict] = {}
    for d in runs:
        if d.get("mode") == "vector":
            b = d.get("benchmark", "locomo")
            retr = d.get("retriever") or "all-MiniLM-L6-v2"
            k = int(d.get("k") or 0)
            n = int(d.get("n") or 0)
            vec_runs[(b, retr, k, n)] = d

    groups: dict[tuple, dict[str, dict]] = {}
    for d in runs:
        mode = d.get("mode", "")
        if not mode.startswith("memory_"):
            continue
        b = d.get("benchmark", "locomo")
        retr = d.get("retriever") or "all-MiniLM-L6-v2"
        ext = _extractor_of(d) or str(d.get("model") or "qwen2.5:7b-instruct")
        k = int(d.get("k") or 0)
        n = int(d.get("n") or 0)
        gkey = (b, retr, ext, k, n)
        if gkey not in groups:
            groups[gkey] = {}
            v_match = vec_runs.get((b, retr, k, n))
            if v_match:
                groups[gkey]["vector"] = v_match
        groups[gkey][mode] = d
    return groups


def print_h1_decomposition(runs: list[dict]) -> None:
    """Print the 3 H1 tables:
      1. Overall H1 Decomposition (F1_answerable)
      2. Category-Stratified H1 Breakdown (Cat 1-5 F1 & Recall)
      3. Paired Cluster-Bootstrap 95% CIs & McNemar p-values
    """
    groups = _group_h1_runs(runs)
    if not groups:
        return

    print("\n### Hypothesis 1 (Table 1): Overall Structure vs. Extraction Decomposition (`F1_answerable`)\n")
    print("| benchmark | retriever | extractor | k | n | Vector (Raw) | Flat (No Graph) | HippoRAG | Hippo Rewired | Hippo Flattened | Bi-Temporal | Δ_extraction | Δ_structure | Δ_rewire | Δ_ablation | Δ_temporal |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for (b, retr, ext, k, n), mm in sorted(groups.items()):
        v_run = mm.get("vector")
        f_run = mm.get("memory_flat")
        h_run = mm.get("memory_hipporag")
        hr_run = mm.get("memory_hipporag_rewired")
        hf_run = mm.get("memory_hipporag_flattened")
        t_run = mm.get("memory_temporal_bteg")

        f1_metric = "F1_answerable"

        v_val = v_run.get(f1_metric) if v_run else None
        f_val = f_run.get(f1_metric) if f_run else None
        h_val = h_run.get(f1_metric) if h_run else None
        hr_val = hr_run.get(f1_metric) if hr_run else None
        hf_val = hf_run.get(f1_metric) if hf_run else None
        t_val = t_run.get(f1_metric) if t_run else None

        d_ext = (f_val - v_val) if (f_val is not None and v_val is not None) else None
        d_struct = (h_val - f_val) if (h_val is not None and f_val is not None) else None
        d_rewire = (h_val - hr_val) if (h_val is not None and hr_val is not None) else None
        d_ablate = (h_val - hf_val) if (h_val is not None and hf_val is not None) else None
        d_temp = (t_val - f_val) if (t_val is not None and f_val is not None) else None

        print(f"| {b} | {retr} | {ext} | {k} | {n} | {fmt(v_val)} | {fmt(f_val)} | {fmt(h_val)} | "
              f"{fmt(hr_val)} | {fmt(hf_val)} | {fmt(t_val)} | {fmt_signed(d_ext)} | "
              f"{fmt_signed(d_struct)} | {fmt_signed(d_rewire)} | {fmt_signed(d_ablate)} | {fmt_signed(d_temp)} |")

    # Table 2: Category-Stratified H1 Breakdown
    print("\n### Hypothesis 1 (Table 2): Category-Stratified F1 & Abstention Breakdown\n")
    print("Categories 1–4 report token `F1`; Category 5 (`abstention`/`adversarial`) reports `refusal_accuracy`.\n")
    for (b, retr, ext, k, n), mm in sorted(groups.items()):
        cats = CAT_NAMES.get(b, CAT_NAMES["longmemeval"])
        print(f"#### `{b}` · retriever=`{retr}` · extractor=`{ext}` · `k={k}` · `n={n}`\n")
        cat_headers = [f"Cat {c}: {cats[c]}" + (" (refusal)" if c == 5 else "") for c in sorted(cats)]
        print("| Arm / Comparison | " + " | ".join(cat_headers) + " | F1_answerable |")
        print("|---|" + "---|" * (len(cats) + 1))

        arm_order = [
            ("vector", "Vector (Raw Chunks)"),
            ("memory_flat", "Flat Facts (No Graph)"),
            ("memory_hipporag", "HippoRAG (Full Graph)"),
            ("memory_hipporag_rewired", "HippoRAG (Degree-Rewired)"),
            ("memory_hipporag_flattened", "HippoRAG (Flattened)"),
            ("memory_temporal_bteg", "Bi-Temporal Graph (BTEG)"),
        ]
        cat_vals_by_mode: dict[str, dict[int, float | None]] = {}
        for m_key, m_label in arm_order:
            run_d = mm.get(m_key)
            if not run_d:
                continue
            f1_cat = run_d.get("F1_by_category", {})
            c_map: dict[int, float | None] = {}
            cells: list[str] = []
            for c in sorted(cats):
                if c == 5:
                    val = run_d.get("refusal_accuracy")
                else:
                    val = f1_cat.get(str(c) if str(c) in f1_cat else c)
                c_map[c] = val
                cells.append(fmt(val))
            cat_vals_by_mode[m_key] = c_map
            print(f"| {m_label} | " + " | ".join(cells) + f" | {fmt(run_d.get('F1_answerable'))} |")

        # Print Δ rows per category
        delta_specs = [
            ("Δ_extraction (Flat - Raw)", "memory_flat", "vector"),
            ("Δ_structure (HippoRAG - Flat)", "memory_hipporag", "memory_flat"),
            ("Δ_rewire (HippoRAG - Rewired)", "memory_hipporag_rewired", None),
            ("Δ_ablation (HippoRAG - Flattened)", "memory_hipporag", "memory_hipporag_flattened"),
            ("Δ_temporal (BTEG - Flat)", "memory_temporal_bteg", "memory_flat"),
        ]
        for d_label, ma, mb in delta_specs:
            if d_label.startswith("Δ_rewire"):
                ma, mb = "memory_hipporag", "memory_hipporag_rewired"
            if ma not in cat_vals_by_mode or mb not in cat_vals_by_mode:
                continue
            d_cells: list[str] = []
            for c in sorted(cats):
                va = cat_vals_by_mode[ma].get(c)
                vb = cat_vals_by_mode[mb].get(c)
                d_cells.append(fmt_signed(va - vb) if (va is not None and vb is not None) else "–")
            ra_ans = mm[ma].get("F1_answerable")
            rb_ans = mm[mb].get("F1_answerable")
            d_ans = (ra_ans - rb_ans) if (ra_ans is not None and rb_ans is not None) else None
            print(f"| **{d_label}** | " + " | ".join(d_cells) + f" | **{fmt_signed(d_ans)}** |")
        print()

    # Table 3: Paired Cluster-Bootstrap CIs & Exact McNemar Tests
    print("### Hypothesis 1 (Table 3): Paired Statistical Significance (Answerable Qs, Cat 1–4)\n")
    print("95% confidence intervals use 10,000-draw conversation-cluster bootstrap (`sample_id` clusters). "
          "`p_perm (ΔF1)` is the two-sided cluster sign-flip permutation p-value; `p_McNemar` is the exact "
          "two-sided binomial test on discordant pairs `(A+,B- / A-,B+)`.\n")
    print("| benchmark | retriever | extractor | k | Comparison (A vs B) | n_pairs (clusters) | ΔF1_ans | 95% Cluster CI | p_perm (ΔF1) | EM (A+,B- / A-,B+) | p_McNemar (EM) | Recall (A+,B- / A-,B+) | p_McNemar (Rec) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    pair_specs = [
        ("Flat vs Raw (Δ_extraction)", "memory_flat", "vector"),
        ("HippoRAG vs Flat (Δ_structure)", "memory_hipporag", "memory_flat"),
        ("HippoRAG vs Rewired (Δ_rewire)", "memory_hipporag", "memory_hipporag_rewired"),
        ("HippoRAG vs Flattened (Δ_ablation)", "memory_hipporag", "memory_hipporag_flattened"),
        ("Bi-Temporal vs Flat (Δ_temporal)", "memory_temporal_bteg", "memory_flat"),
    ]
    for (b, retr, ext, k, n), mm in sorted(groups.items()):
        for label, mode_a, mode_b in pair_specs:
            ra = mm.get(mode_a)
            rb = mm.get(mode_b)
            if not ra or not rb:
                continue
            st = _paired_cluster_stats(ra, rb)
            if not st.get("n_pairs"):
                continue
            em_b, em_c = st["em_discordant"]  # type: ignore[misc]
            rec_b, rec_c = st["rec_discordant"]  # type: ignore[misc]
            print(
                f"| {b} | {retr} | {ext} | {k} | {label} | "
                f"{st['n_pairs']} ({st['n_clusters']}) | {fmt_signed(st['delta_f1'])} | "
                f"[{fmt_signed(st['ci_low'])}, {fmt_signed(st['ci_high'])}] | "
                f"{fmt(st['p_perm_f1'])} | {em_b}/{em_c} | {fmt(st['p_mcnemar_em'])} | "
                f"{rec_b}/{rec_c} | {fmt(st['p_mcnemar_rec'])} |"
            )


def main() -> None:
    runs = load_runs()
    if not runs:
        print("no results yet")
        return

    print("<!-- GENERATED by src/make_tables.py -- do not edit by hand -->\n")
    print("Rows are keyed by **(benchmark, mode, retriever, extractor, k, n)**. `n` is part of the key: the "
          "25-QA balanced subset and the 500-QA full split are different measurements and are never "
          "merged into one row. The `file` column is the run id; its timestamp prefix is the run start "
          "(UTC). Scoring conventions: `F1` scores adversarial as 0 (floor), `F1_paper` scores correct "
          "refusal as a hit (LoCoMo convention), `F1_answerable` covers categories 1-4 only.\n")
    print("### All runs (latest per benchmark · mode · retriever · extractor · k · n)\n")
    print("| benchmark | mode | k | retriever | extractor | n | F1 | F1_paper | EM | F1_answerable | "
          "refusal_acc | recall_any | recall_all | coverage | prompt chars | latency s | file |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in runs:
        b = d.get("benchmark", "locomo")
        ext = _extractor_of(d) or "–"
        print(f"| {b} | {d['mode']} | {d['k']} | {d.get('retriever') or 'all-MiniLM-L6-v2'} "
              f"| {ext} | {d['n']} | {fmt(d.get('F1'))} | {fmt(d.get('F1_paper'))} "
              f"| {fmt(d.get('EM'))} | {fmt(d.get('F1_answerable'))} "
              f"| {fmt(d.get('refusal_accuracy'))} | {fmt(d.get('recall_any'))} "
              f"| {fmt(d.get('recall_all'))} | {fmt(d.get('evidence_coverage'))} "
              f"| {d.get('mean_prompt_chars', 0):.0f} "
              f"| {d.get('mean_latency_s', 0):.2f} | {d.get('_file', '')} |")

    print_h1_decomposition(runs)

    import predict_utility
    predict_utility.print_h2_diagnostics(runs)

    for b in ("locomo", "longmemeval"):
        b_runs = [d for d in runs if d.get("benchmark", "locomo") == b and d.get("recall_any_by_category")]
        if not b_runs:
            continue
        cats = CAT_NAMES[b]
        print(f"\n### Retrieval recall by category: {b}\n")
        print("| mode | retriever | extractor | k | n | " + " | ".join(cats[c] for c in sorted(cats)) + " | overall |")
        print("|---|---|---|---|---|" + "---|" * (len(cats) + 1))
        for d in b_runs:
            ext = _extractor_of(d) or "–"
            cells = " | ".join(
                fmt(d["recall_any_by_category"].get(str(c) if str(c) in d["recall_any_by_category"] else c))
                for c in sorted(cats))
            print(f"| {d['mode']} | {d.get('retriever') or 'all-MiniLM-L6-v2'} | {ext} | {d['k']} | {d['n']} "
                  f"| {cells} | {fmt(d.get('recall_any'))} |")


if __name__ == "__main__":
    main()
