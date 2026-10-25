#!/usr/bin/env python3
"""Hypothesis 2 (Chapter 2): Predictive Non-Parametric Topological Diagnostics.

Tests whether cheap O(|V|+|E|) structural graph metrics computed without LLM
calls (Degree Gini, Max Degree Ratio, Largest Connected Component Ratio, Edge
Density, Clustering Coefficient, Bridge Ratio) quantitatively predict
downstream retrieval utility and QA accuracy across conversational memory runs.

Formal H2 Falsification Test:
  - Multiple OLS Regression:
      R^2(Utility ~ degree_gini + largest_component_ratio + edge_density + bridge_edge_ratio) >= 0.20
  - Rank Correlation:
      Spearman rho(degree_gini, precision_at_k / F1) <= -0.30

Usage:
    python3 src/predict_utility.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path
import random
from typing import Any

def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "conf").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[3]

ROOT = _find_repo_root()
RESULTS = ROOT / "results"

CORE_PREDICTORS = [
    "degree_gini",
    "max_degree_ratio",
    "largest_component_ratio",
    "edge_density",
    "bridge_edge_ratio",
]

ALL_PREDICTORS = [
    "degree_gini",
    "max_degree_ratio",
    "largest_component_ratio",
    "edge_density",
    "clustering_coefficient",
    "bridge_edge_ratio",
    "temporal_edge_ratio",
    "invalidated_edge_ratio",
]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Sample Pearson correlation coefficient r in [-1, 1]."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / (den_x * den_y)))


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks (1-indexed) with tie handling."""
    n = len(xs)
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for m in range(i, j + 1):
            ranks[indexed[m][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation coefficient rho in [-1, 1]."""
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    return pearson_r(_ranks(xs), _ranks(ys))


def permutation_p_corr(
    xs: list[float],
    ys: list[float],
    metric: str = "spearman",
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test p-value for correlation (Spearman or Pearson)."""
    n = len(xs)
    if n < 4:
        return 1.0
    fn = spearman_rho if metric == "spearman" else pearson_r
    obs = abs(fn(xs, ys))
    if obs < 1e-12:
        return 1.0
    rng = random.Random(seed)
    y_shuf = list(ys)
    exceed = 0
    for _ in range(n_perm):
        rng.shuffle(y_shuf)
        if abs(fn(xs, y_shuf)) >= obs - 1e-12:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b via Gaussian elimination with partial pivoting and Tikhonov ridge."""
    p = len(b)
    aug = [
        [a[i][j] + (1e-8 if i == j and i > 0 else 0.0) for j in range(p)] + [b[i]]
        for i in range(p)
    ]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, p + 1):
            aug[col][j] /= div
        for r in range(p):
            if r != col and abs(aug[r][col]) > 0.0:
                factor = aug[r][col]
                for j in range(col, p + 1):
                    aug[r][j] -= factor * aug[col][j]
    return [aug[i][p] for i in range(p)]


def fit_ols(
    x_cols: dict[str, list[float]],
    y: list[float],
    n_perm: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit standardized Ordinary Least Squares (OLS) regression Y ~ X_1 + ... + X_p.

    Standardizes predictors and target (zero mean, unit variance) so beta_i are
    standardized effect sizes directly comparable across structural dimensions.
    """
    n = len(y)
    active_names = [
        name for name, col in x_cols.items()
        if len(col) == n and _std(col) > 1e-9
    ]
    p = len(active_names)
    if n <= p + 1 or _std(y) < 1e-12 or p == 0:
        return {
            "n": n,
            "p": p,
            "r2": 0.0,
            "adj_r2": 0.0,
            "f_stat": 0.0,
            "p_value": 1.0,
            "betas": {name: 0.0 for name in x_cols},
        }

    # Standardize columns
    z_cols: list[list[float]] = []
    for name in active_names:
        col = x_cols[name]
        m, s = _mean(col), _std(col)
        z_cols.append([(v - m) / s for v in col])

    my, sy = _mean(y), _std(y)
    zy = [(v - my) / sy for v in y]

    def _compute_r2_and_betas(target_z: list[float]) -> tuple[float, list[float]]:
        # Design matrix X has intercept + p standardized columns
        k_dim = p + 1
        xtx = [[0.0] * k_dim for _ in range(k_dim)]
        xty = [0.0] * k_dim
        for i in range(n):
            row_x = [1.0] + [z_cols[c][i] for c in range(p)]
            yi = target_z[i]
            for r_idx in range(k_dim):
                xty[r_idx] += row_x[r_idx] * yi
                for c_idx in range(r_idx, k_dim):
                    xtx[r_idx][c_idx] += row_x[r_idx] * row_x[c_idx]
        for r_idx in range(k_dim):
            for c_idx in range(r_idx):
                xtx[r_idx][c_idx] = xtx[c_idx][r_idx]

        coeffs = _solve_linear_system(xtx, xty)
        preds = [
            coeffs[0] + sum(coeffs[c + 1] * z_cols[c][i] for c in range(p))
            for i in range(n)
        ]
        ss_res = sum((target_z[i] - preds[i]) ** 2 for i in range(n))
        ss_tot = sum(v * v for v in target_z)
        r2_val = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot))) if ss_tot > 1e-12 else 0.0
        return r2_val, coeffs[1:]

    r2, beta_list = _compute_r2_and_betas(zy)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - p - 1)
    f_stat = (r2 / p) / ((1.0 - r2) / max(1, n - p - 1)) if r2 < 1.0 else 999.0

    # Permutation test for overall model R^2
    rng = random.Random(seed)
    zy_shuf = list(zy)
    exceed = 0
    for _ in range(n_perm):
        rng.shuffle(zy_shuf)
        r2_perm, _ = _compute_r2_and_betas(zy_shuf)
        if r2_perm >= r2 - 1e-12:
            exceed += 1
    p_val = (exceed + 1) / (n_perm + 1)

    betas_dict = {name: 0.0 for name in x_cols}
    for name, b_val in zip(active_names, beta_list):
        betas_dict[name] = b_val

    return {
        "n": n,
        "p": p,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "p_value": p_val,
        "betas": betas_dict,
    }


def _row_precision_at_k(row: dict) -> float:
    """Compute per-query Precision@k = |retrieved ∩ gold| / |retrieved|."""
    ev = {str(x) for x in (row.get("evidence") or [])}
    ctx = [str(x) for x in (row.get("context_ids") or [])]
    if not ev or not ctx:
        return 0.0
    uniq_ctx = list(dict.fromkeys(ctx))
    hits = sum(1 for c in uniq_ctx if c in ev)
    return hits / len(uniq_ctx)


def collect_sample_observations(runs: list[dict]) -> list[dict[str, Any]]:
    """Extract per-question observations with graph metrics and paired Flat deltas."""
    # Map (benchmark, retriever, extractor, k, n) -> memory_flat run
    flat_lookup: dict[tuple, dict[tuple[str, str], dict]] = {}
    for d in runs:
        if d.get("mode") != "memory_flat":
            continue
        ext = str(d.get("extractor_model") or d.get("model") or "")
        gkey = (
            d.get("benchmark", "locomo"),
            d.get("retriever") or "all-MiniLM-L6-v2",
            ext,
            int(d.get("k") or 0),
            int(d.get("n") or 0),
        )
        flat_lookup[gkey] = {
            (str(r.get("sample_id")), str(r.get("question"))): r
            for r in d.get("rows", [])
        }

    observations: list[dict[str, Any]] = []
    for d in runs:
        mode = str(d.get("mode") or "")
        if not mode.startswith("memory_") or mode == "memory_flat":
            continue
        by_sample = d.get("graph_metrics_by_sample") or {}
        if not by_sample:
            continue

        ext = str(d.get("extractor_model") or d.get("model") or "")
        retr = d.get("retriever") or "all-MiniLM-L6-v2"
        b = d.get("benchmark", "locomo")
        k = int(d.get("k") or 0)
        n = int(d.get("n") or 0)
        flat_rows = flat_lookup.get((b, retr, ext, k, n), {})

        for r in d.get("rows", []):
            if r.get("adversarial"):
                continue
            sid = str(r.get("sample_id"))
            q = str(r.get("question"))
            gm = by_sample.get(sid)
            if not gm or gm.get("num_nodes", 0) == 0:
                continue

            f1_val = float(r.get("F1", 0.0))
            cov_val = float(r.get("evidence_coverage") if r.get("evidence_coverage") is not None else (r.get("recall_any") or 0.0))
            rec_all = float(r.get("recall_all") if r.get("recall_all") is not None else (r.get("recall_any") or 0.0))
            prec_k = _row_precision_at_k(r)

            flat_r = flat_rows.get((sid, q))
            delta_f1 = (f1_val - float(flat_r.get("F1", 0.0))) if flat_r else 0.0
            flat_cov = float(flat_r.get("evidence_coverage") or 0.0) if flat_r else cov_val
            delta_cov = cov_val - flat_cov

            obs = {
                "benchmark": b,
                "mode": mode,
                "retriever": retr,
                "extractor": ext,
                "k": k,
                "n": n,
                "sample_id": sid,
                "category": int(r.get("category") or 0),
                "F1": f1_val,
                "evidence_coverage": cov_val,
                "recall_all": rec_all,
                "precision_at_k": prec_k,
                "delta_f1_vs_flat": delta_f1,
                "delta_cov_vs_flat": delta_cov,
            }
            for m_name in ALL_PREDICTORS:
                obs[m_name] = float(gm.get(m_name, 0.0))
            observations.append(obs)

    return observations


def analyze_observations(obs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute univariate correlations, OLS regressions, and regime splits."""
    if not obs:
        return {"n": 0}

    targets = ["evidence_coverage", "precision_at_k", "F1", "delta_f1_vs_flat"]
    correlations: dict[str, dict[str, dict[str, float]]] = {}

    for pred in ALL_PREDICTORS:
        xs = [o[pred] for o in obs]
        if _std(xs) < 1e-9:
            continue
        correlations[pred] = {}
        for tgt in targets:
            ys = [o[tgt] for o in obs]
            rho = spearman_rho(xs, ys)
            r = pearson_r(xs, ys)
            p_rho = permutation_p_corr(xs, ys, metric="spearman", n_perm=2500)
            correlations[pred][tgt] = {
                "spearman_rho": rho,
                "pearson_r": r,
                "p_value": p_rho,
            }

    x_core = {p: [o[p] for o in obs] for p in CORE_PREDICTORS}
    ols_models = {
        tgt: fit_ols(x_core, [o[tgt] for o in obs], n_perm=3000)
        for tgt in targets
    }

    # Regime split: High Hub Concentration (degree_gini >= median) vs Low Hub Concentration
    ginis = sorted(o["degree_gini"] for o in obs)
    med_gini = ginis[len(ginis) // 2]
    low_hub = [o for o in obs if o["degree_gini"] <= med_gini]
    high_hub = [o for o in obs if o["degree_gini"] > med_gini]

    return {
        "n": len(obs),
        "correlations": correlations,
        "ols": ols_models,
        "median_gini": med_gini,
        "low_hub_summary": {
            "n": len(low_hub),
            "mean_coverage": _mean([o["evidence_coverage"] for o in low_hub]),
            "mean_precision": _mean([o["precision_at_k"] for o in low_hub]),
            "mean_f1": _mean([o["F1"] for o in low_hub]),
            "mean_delta_f1": _mean([o["delta_f1_vs_flat"] for o in low_hub]),
        },
        "high_hub_summary": {
            "n": len(high_hub),
            "mean_coverage": _mean([o["evidence_coverage"] for o in high_hub]),
            "mean_precision": _mean([o["precision_at_k"] for o in high_hub]),
            "mean_f1": _mean([o["F1"] for o in high_hub]),
            "mean_delta_f1": _mean([o["delta_f1_vs_flat"] for o in high_hub]),
        },
    }


def print_h2_diagnostics(runs: list[dict]) -> None:
    """Print Hypothesis 2 (Topological Diagnostics & Utility Prediction) markdown tables."""
    # Group graph metrics per run first
    graph_runs = [d for d in runs if d.get("graph_metrics")]
    if not graph_runs:
        return

    print("\n### Hypothesis 2 (Table 1): Mean Topological Graph Metrics Across Runs\n")
    print("| benchmark | mode | retriever | extractor | k | n | Nodes | Edges | LCC Ratio (ρ_LCC) | Components | Degree Gini (G_deg) | Max/Avg Deg | Density | Clustering | Bridge Ratio | Temporal Ratio | Superseded Ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in graph_runs:
        gm = d["graph_metrics"]
        ext = str(d.get("extractor_model") or d.get("model") or "–")
        print(
            f"| {d.get('benchmark', 'locomo')} | {d['mode']} | {d.get('retriever') or 'all-MiniLM-L6-v2'} | "
            f"{ext} | {d['k']} | {d['n']} | {gm.get('num_nodes', 0):.1f} | {gm.get('num_edges', 0):.1f} | "
            f"{gm.get('largest_component_ratio', 0.0):.3f} | {gm.get('num_connected_components', 0):.1f} | "
            f"{gm.get('degree_gini', 0.0):.3f} | {gm.get('max_degree_ratio', 0.0):.2f} | "
            f"{gm.get('edge_density', 0.0):.4f} | {gm.get('clustering_coefficient', 0.0):.3f} | "
            f"{gm.get('bridge_edge_ratio', 0.0):.3f} | {gm.get('temporal_edge_ratio', 0.0):.3f} | "
            f"{gm.get('invalidated_edge_ratio', 0.0):.3f} |"
        )

    obs = collect_sample_observations(runs)
    if not obs:
        return

    # Analyze pooled across active graph traversals (HippoRAG + Bi-Temporal, excluding flattened 0-edge control)
    active_obs = [o for o in obs if o["mode"] in ("memory_hipporag", "memory_hipporag_rewired", "memory_temporal_bteg")]
    if len(active_obs) < 6:
        active_obs = obs

    res = analyze_observations(active_obs)

    print("\n### Hypothesis 2 (Table 2): Univariate Topological Correlations (`n="
          f"{res['n']}` active graph QA observations)\n")
    print("Reports Spearman rank correlation `ρ` (and two-sided permutation `p`-value) between `O(|V|+|E|)` "
          "structural metrics and per-query retrieval/QA outcomes.\n")
    print("| Structural Predictor | ρ (Evidence Coverage) | p-val | ρ (Precision@k) | p-val | ρ (Answerable F1) | p-val | ρ (ΔF1 vs Flat) | p-val |")
    print("|---|---|---|---|---|---|---|---|---|")
    for pred, tmap in res["correlations"].items():
        c_cov = tmap["evidence_coverage"]
        c_prec = tmap["precision_at_k"]
        c_f1 = tmap["F1"]
        c_df1 = tmap["delta_f1_vs_flat"]
        print(
            f"| `{pred}` | {c_cov['spearman_rho']:+.3f} | {c_cov['p_value']:.3f} | "
            f"{c_prec['spearman_rho']:+.3f} | {c_prec['p_value']:.3f} | "
            f"{c_f1['spearman_rho']:+.3f} | {c_f1['p_value']:.3f} | "
            f"{c_df1['spearman_rho']:+.3f} | {c_df1['p_value']:.3f} |"
        )

    print("\n### Hypothesis 2 (Table 3): Multivariate OLS Regression (`Y ~ G_deg + MaxDegRatio + ρ_LCC + Density + BridgeRatio`)\n")
    print("Standardized OLS coefficients (`β`) and model fit (`R²`, `F`-stat, permutation `p`-value) testing "
          "the H2 falsification criterion (`R² >= 0.20`).\n")
    print("| Dependent Target (Y) | n | R² | Adj R² | F-stat | p-val (Model) | β (degree_gini) | β (max_deg_ratio) | β (LCC_ratio) | β (edge_density) | β (bridge_ratio) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    target_labels = [
        ("evidence_coverage", "Retrieval Evidence Coverage"),
        ("precision_at_k", "Retrieval Precision@k"),
        ("F1", "End-to-End Answerable F1"),
        ("delta_f1_vs_flat", "Structural Gain (ΔF1 vs Flat)"),
    ]
    for t_key, t_label in target_labels:
        m = res["ols"][t_key]
        b = m["betas"]
        print(
            f"| {t_label} | {m['n']} | **{m['r2']:.3f}** | {m['adj_r2']:.3f} | {m['f_stat']:.2f} | "
            f"{m['p_value']:.4f} | {b.get('degree_gini', 0.0):+.3f} | "
            f"{b.get('max_degree_ratio', 0.0):+.3f} | "
            f"{b.get('largest_component_ratio', 0.0):+.3f} | {b.get('edge_density', 0.0):+.3f} | "
            f"{b.get('bridge_edge_ratio', 0.0):+.3f} |"
        )


def main() -> None:
    try:
        from graph_memory.analysis import make_tables
    except ImportError:
        import make_tables
    runs = make_tables.load_runs()
    if not runs:
        print("No result files found in results/.")
        return
    print_h2_diagnostics(runs)


if __name__ == "__main__":
    main()
