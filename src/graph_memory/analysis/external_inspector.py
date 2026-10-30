"""Topological Sidecar: Non-invasive graph inspection for upstream repositories.

Extracts on-disk persisted graph artifacts from MAGMA, EMem-G, and A-MEM,
projects them to simple undirected networkx.Graph objects, computes the 5
non-parametric H2 topological metrics, and emits canonical run dictionaries.
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from graph_memory.analysis.graph_metrics import compute_graph_metrics


def project_to_simple_undirected(g: nx.Graph | nx.MultiGraph | nx.DiGraph | nx.MultiDiGraph) -> nx.Graph:
    """Project any directed or multi-edge graph to a simple undirected graph without self-loops."""
    simple = nx.Graph()
    for n, data in g.nodes(data=True):
        simple.add_node(n, **data)
    for u, v in g.edges():
        if u != v:
            simple.add_edge(u, v)
    return simple


def extract_magma_graph(cache_path: Path) -> nx.Graph:
    """Load MAGMA's persisted graph (JSON or pickle) and project to simple undirected graph.

    Handles either a direct JSON/pickle file or a sample directory containing graph.json / graph.pickle.
    """
    p = cache_path
    if p.is_dir():
        for candidate in ["graph.json", "graph.pickle", "graph.pkl", "trg_graph.pkl", "magma_graph.pkl"]:
            if (p / candidate).exists():
                p = p / candidate
                break

    if not p.is_file():
        raise FileNotFoundError(f"MAGMA graph artifact not found at {cache_path}")

    if p.name.endswith(".json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        simple = nx.Graph()
        for n in data.get("nodes", []):
            nid = n.get("node_id") or n.get("id")
            if nid is not None:
                simple.add_node(nid)
        for l in data.get("links", []):
            u = l.get("source_node_id") or l.get("source_id") or l.get("source") or l.get("src")
            v = l.get("target_node_id") or l.get("target_id") or l.get("target") or l.get("dst")
            if u is not None and v is not None and u != v:
                simple.add_edge(u, v)
        return simple

    with open(p, "rb") as f:
        raw_g = pickle.load(f)

    if not isinstance(raw_g, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(raw_g)}")

    return project_to_simple_undirected(raw_g)


class _SafeIGraphUnpickler(pickle.Unpickler):
    """Safely unpickle EMem igraph graph even if internal classes are not importable."""
    def find_class(self, module: str, name: str) -> Any:
        if "src.emem" in module or "turns" in module:
            class DummyData:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    pass
                def __setstate__(self, state: Any) -> None:
                    pass
            return DummyData
        return super().find_class(module, name)


def extract_emem_graph(cache_path: Path) -> nx.Graph:
    """Load EMem-G's persisted graph and convert to simple undirected NetworkX graph.

    Handles either a python-igraph graph pickle or an adjacency list/edge list pickle.
    """
    p = cache_path
    if p.is_dir():
        for candidate in ["graph.pickle", "graph.pkl", "emem_graph.pkl", "igraph.pkl"]:
            if (p / candidate).exists():
                p = p / candidate
                break

    if not p.is_file():
        raise FileNotFoundError(f"EMem graph artifact not found at {cache_path}")

    with open(p, "rb") as f:
        try:
            raw_obj = pickle.load(f)
        except (AttributeError, ModuleNotFoundError):
            f.seek(0)
            raw_obj = _SafeIGraphUnpickler(f).load()

    simple = nx.Graph()

    # Check if python-igraph object
    if hasattr(raw_obj, "vcount") and hasattr(raw_obj, "get_edgelist"):
        # igraph.Graph
        n_verts = raw_obj.vcount()
        for i in range(n_verts):
            simple.add_node(i)
        for u, v in raw_obj.get_edgelist():
            if u != v:
                simple.add_edge(u, v)
        return simple

    # If already a NetworkX graph
    if isinstance(raw_obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return project_to_simple_undirected(raw_obj)

    # If dict of nodes/edges or adjacency
    if isinstance(raw_obj, dict):
        if "edges" in raw_obj:
            for u, v in raw_obj["edges"]:
                if u != v:
                    simple.add_edge(u, v)
            return simple

    raise TypeError(f"Unrecognized EMem graph artifact type: {type(raw_obj)}")


class _SafeNoteUnpickler(pickle.Unpickler):
    """Safely unpickle A-MEM memory cache even if MemoryNote is not importable."""
    def find_class(self, module: str, name: str) -> Any:
        if name in ("MemoryNote", "RobustMemoryNote") or "memory_layer" in module:
            class DummyMemoryNote:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)
                def __setstate__(self, state: Any) -> None:
                    if isinstance(state, dict):
                        self.__dict__.update(state)
            return DummyMemoryNote
        return super().find_class(module, name)


def extract_amem_graph(cache_path: Path) -> nx.Graph:
    """Load A-MEM's persisted MemoryNote dictionary and construct simple undirected graph from note.links."""
    p = cache_path
    if p.is_dir():
        for candidate in ["memory_cache.pkl", "memory_cache_sample_0.pkl"]:
            if (p / candidate).exists():
                p = p / candidate
                break

    if not p.is_file():
        raise FileNotFoundError(f"A-MEM memory cache artifact not found at {cache_path}")

    with open(p, "rb") as f:
        try:
            memories = pickle.load(f)
        except (AttributeError, ModuleNotFoundError):
            f.seek(0)
            memories = _SafeNoteUnpickler(f).load()

    simple = nx.Graph()

    if isinstance(memories, dict):
        notes = list(memories.values())
    elif isinstance(memories, list):
        notes = memories
    else:
        raise TypeError(f"Expected dict or list of memory notes, got {type(memories)}")

    id_to_idx: dict[str, int] = {}
    for idx, note in enumerate(notes):
        simple.add_node(idx)
        note_id = getattr(note, "id", None) or (note.get("id") if isinstance(note, dict) else None)
        if note_id is not None:
            id_to_idx[str(note_id)] = idx

    for idx, note in enumerate(notes):
        links = getattr(note, "links", None) or (note.get("links") if isinstance(note, dict) else None) or []
        for target in links:
            t_idx = None
            if isinstance(target, (int, np.integer)):
                if 0 <= target < len(notes):
                    t_idx = int(target)
            elif isinstance(target, str):
                if target.strip().isdigit():
                    val = int(target.strip())
                    if 0 <= val < len(notes):
                        t_idx = val
                elif target in id_to_idx:
                    t_idx = id_to_idx[target]
            if t_idx is not None and t_idx != idx:
                simple.add_edge(idx, t_idx)

    return simple


def _compute_evidence_coverage(retrieved_ids: list[str], gold_evidence: list[str]) -> float:
    if not gold_evidence:
        return 1.0
    gold_set = {str(x) for x in gold_evidence}
    ret_set = {str(x) for x in retrieved_ids}
    return len(gold_set & ret_set) / len(gold_set)


def _compute_precision_at_k(retrieved_ids: list[str], gold_evidence: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    gold_set = {str(x) for x in gold_evidence}
    uniq_ret = list(dict.fromkeys(str(x) for x in retrieved_ids))
    hits = sum(1 for x in uniq_ret if x in gold_set)
    return hits / len(uniq_ret)


def inspect_upstream_run(
    framework: str,
    graph_artifacts_by_sample: dict[str, Path],
    qa_results: list[dict[str, Any]],
    benchmark: str = "locomo",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    extractor_model: str = "Qwen/Qwen2.5-7B-Instruct",
    retriever: str = "all-MiniLM-L6-v2",
    k: int = 10,
    n: int = 10,
) -> dict[str, Any]:
    """Load per-sample graphs, compute H2 topological metrics, and format as CanonicalUpstreamRun."""
    framework = framework.lower()
    metrics_by_sample: dict[str, dict[str, float]] = {}

    for sid, gpath in graph_artifacts_by_sample.items():
        if framework == "magma":
            g = extract_magma_graph(gpath)
        elif framework in ("emem_g", "emem"):
            g = extract_emem_graph(gpath)
        elif framework == "amem":
            g = extract_amem_graph(gpath)
        elif framework == "emem_flat":
            # Zero-edge flat baseline
            g = nx.Graph()
            # If sample path exists, add isolated nodes
            try:
                base_g = extract_emem_graph(gpath)
                for n_idx in base_g.nodes():
                    g.add_node(n_idx)
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown framework: {framework}")

        metrics_by_sample[str(sid)] = compute_graph_metrics(g)

    # Format rows
    rows: list[dict[str, Any]] = []
    for qa in qa_results:
        sid = str(qa.get("sample_id", "0"))
        gold_ev = [str(x) for x in (qa.get("evidence") or [])]
        ctx_ids = [str(x) for x in (qa.get("context_ids") or [])]

        cov = _compute_evidence_coverage(ctx_ids, gold_ev)
        prec = _compute_precision_at_k(ctx_ids, gold_ev)

        ans_str = str(qa.get("answer") or qa.get("reference") or qa.get("ground_truth") or "")
        pred_str = str(qa.get("predicted") or qa.get("prediction") or "")

        f1_candidate = qa.get("F1")
        if f1_candidate is None:
            f1_candidate = qa.get("f1")
        if f1_candidate is None and isinstance(qa.get("metrics"), dict):
            f1_candidate = qa["metrics"].get("f1") or qa["metrics"].get("F1")
        f1_val = float(f1_candidate or 0.0)

        rows.append({
            "sample_id": sid,
            "question": str(qa.get("question", "")),
            "answer": ans_str,
            "predicted": pred_str,
            "evidence": gold_ev,
            "context_ids": ctx_ids,
            "evidence_coverage": cov,
            "precision_at_k": prec,
            "F1": f1_val,
            "category": int(qa.get("category", 1)),
            "adversarial": bool(qa.get("adversarial", False) or qa.get("category") == 5),
        })

    mode_map = {
        "magma": "memory_magma",
        "emem_g": "memory_emem_g",
        "emem": "memory_emem_g",
        "emem_flat": "memory_emem_flat",
        "amem": "memory_amem",
    }
    mode = mode_map.get(framework, f"memory_{framework}")

    # Compute aggregate graph metrics across samples
    agg_metrics: dict[str, float] = {}
    if metrics_by_sample:
        first_gm = next(iter(metrics_by_sample.values()))
        for metric_name in first_gm:
            vals = [gm[metric_name] for gm in metrics_by_sample.values()]
            agg_metrics[metric_name] = sum(vals) / len(vals)

    ans_rows = [r for r in rows if not r.get("adversarial")]
    mean_f1 = sum(r["F1"] for r in rows) / len(rows) if rows else 0.0
    ans_f1 = sum(r["F1"] for r in ans_rows) / len(ans_rows) if ans_rows else 0.0
    mean_em = sum(1.0 for r in rows if r["predicted"].strip().lower() == r["answer"].strip().lower()) / len(rows) if rows else 0.0
    mean_cov = sum(r["evidence_coverage"] for r in rows) / len(rows) if rows else 0.0

    return {
        "benchmark": benchmark,
        "mode": mode,
        "provider": "vllm",
        "model": model,
        "extractor_model": extractor_model,
        "retriever": retriever,
        "k": k,
        "n": n,
        "F1": mean_f1,
        "F1_paper": mean_f1,
        "F1_answerable": ans_f1,
        "EM": mean_em,
        "evidence_coverage": mean_cov,
        "graph_metrics": agg_metrics,
        "graph_metrics_by_sample": metrics_by_sample,
        "rows": rows,
    }
