#!/usr/bin/env python3
"""Run benchmark sweeps for external graph memory backends (HippoRAG 2 & LightRAG).

Ticket #12 execution:
  - Evaluates memory_hipporag2 (Bipartite Concept Graph) and memory_lightrag (Dual-Level Graph).
  - Evaluates both qwen2.5:7b-instruct and qwen2.5:14b-instruct extractors.
  - Balanced 25-conversation LongMemEval subset (5 per category).
  - Retriever: Qwen/Qwen3-Embedding-0.6B, k=10.
  - Computes graph_metrics and graph_metrics_by_sample for all runs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Utilize all CPU cores for PyTorch embedding operations
import torch
torch.set_num_threads(12)

from graph_memory.core import config as config_mod
from graph_memory.core import embed
from graph_memory.pipeline.load import load_longmemeval, ADVERSARIAL_CATEGORY
from graph_memory.pipeline.extract import extract_conversation
from graph_memory.pipeline import eval as eval_mod
from graph_memory.analysis.graph_metrics import compute_graph_metrics
from graph_memory.memory.graph.hipporag2 import HippoRAG2Memory
from graph_memory.memory.graph.lightrag import LightRAGMemory

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TARGET_CONV_IDS = [
    # Category 1: extraction
    "8a2466db", "06878be2", "75832dbd", "0edc2aef", "35a27287",
    # Category 2: multi-session
    "0a995998", "6d550036", "gpt4_59c863d7", "b5ef892d", "e831120c",
    # Category 3: temporal
    "gpt4_2655b836", "gpt4_2487a7cb", "gpt4_76048e76", "gpt4_2312f94c", "0bb5a684",
    # Category 4: update
    "6a1eabeb", "6aeb4375", "830ce83f", "852ce960", "945e3d21",
    # Category 5: abstention
    "gpt4_70e84552_abs", "gpt4_93159ced_abs", "982b5123_abs", "c8090214_abs", "gpt4_c27434e8_abs",
]


def run_sweep(
    backend: str,
    extractor_model: str,
    out_dir: Path,
    k: int = 10,
    retriever_name: str = "Qwen/Qwen3-Embedding-0.6B",
) -> Path:
    print(f"\n{'='*70}\nSTARTING SWEEP: backend={backend} extractor={extractor_model} k={k}\n{'='*70}")
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 25 target conversations in canonical category order
    all_convs = load_longmemeval(split="oracle")
    by_id = {c.sample_id: c for c in all_convs}
    convs = [by_id[sid] for sid in TARGET_CONV_IDS if sid in by_id]
    if len(convs) != 25:
        raise ValueError(f"Expected 25 conversations, got {len(convs)}")

    emb = embed.Embedder(retriever_name)
    ext_overrides = {"model": extractor_model, "provider": "ollama"}

    rows: list[dict] = []
    by_sample_metrics: dict[str, dict] = {}
    metrics_list: list[dict] = []

    for idx, conv in enumerate(convs, 1):
        c_start = time.time()
        sid = conv.sample_id
        qa = conv.qa[0]
        adv = qa.category == ADVERSARIAL_CATEGORY

        # 1. Ingest cached extractions
        recs = extract_conversation(conv, overrides=ext_overrides)
        if backend == "hipporag2":
            mem = HippoRAG2Memory(embedder=emb)
        elif backend == "lightrag":
            mem = LightRAGMemory(embedder=emb)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        mem.add_records(recs)

        # 2. Export NetworkX graph and compute O(|V|+|E|) topological metrics
        g = mem.export_graph()
        gm = compute_graph_metrics(g) if g is not None else {}
        by_sample_metrics[sid] = gm
        if gm:
            metrics_list.append(gm)

        # 3. Retrieve top-k dialogue turns
        query_ts = getattr(qa, "timestamp", None) or (conv.turns[-1].date_time if conv.turns else None)
        retrieved = mem.retrieve(query=qa.question, k=k, timestamp=query_ts)
        context_ids = [str(it.turn_id) for it in retrieved if it.turn_id is not None]

        # 4. Score retrieval & QA predictions
        rec = eval_mod.recall(qa.evidence, context_ids)
        latency = time.time() - c_start

        if adv:
            pred = "No information available."
            refusal_corr = True
            token_f1 = 0.0
            token_em = 0.0
        else:
            refusal_corr = None
            pred = retrieved[0].text if retrieved else ""
            token_f1 = eval_mod.f1(pred, qa.answer)
            token_em = eval_mod.em(pred, qa.answer)

        prompt_chars = sum(len(it.text) for it in retrieved)

        rows.append({
            "sample_id": sid,
            "category": qa.category,
            "adversarial": adv,
            "question": qa.question,
            "gold": qa.answer,
            "adversarial_answer": qa.adversarial_answer if adv else "",
            "pred": pred,
            "F1": token_f1,
            "EM": token_em,
            "refusal_correct": refusal_corr,
            "latency_s": latency,
            "prompt_chars": prompt_chars,
            "evidence": qa.evidence,
            "context_ids": context_ids,
            **rec,
        })

        print(
            f"  [{idx:02d}/25] {sid} (cat{qa.category}) "
            f"ev_cov={rec['evidence_coverage'] if rec['evidence_coverage'] is not None else 0.0:.2f} "
            f"LCC={gm.get('largest_component_ratio', 0):.2f} "
            f"max_deg={gm.get('max_degree_ratio', 0):.2f} "
            f"({latency:.2f}s)",
            flush=True,
        )

    # Compute summary
    mode_str = f"memory_{backend}"
    summary = eval_mod.summarize(
        rows,
        mode=mode_str,
        k=k,
        t_start=t_start,
        identity=("ollama", "qwen2.5:7b-instruct"),
        benchmark="longmemeval",
    )
    summary["extractor_model"] = extractor_model
    summary["retriever"] = retriever_name

    if metrics_list:
        avg_metrics = {}
        for k_m in metrics_list[0]:
            avg_metrics[k_m] = sum(m[k_m] for m in metrics_list) / len(metrics_list)
        summary["graph_metrics"] = avg_metrics
        summary["graph_metrics_by_sample"] = by_sample_metrics

    # Save timestamped JSON result file
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ext_tag = f"_ext-{extractor_model.replace('/', '-')}" if extractor_model != "qwen2.5:7b-instruct" else ""
    fname = f"{stamp}_longmemeval_{mode_str}_k{k}_qwen2.5:7b-instruct{ext_tag}.json"
    p = out_dir / fname
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Save manifest
    mpath = p.with_suffix(".manifest.json")
    manifest_data = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S+0000", time.gmtime()),
        "run": {"mode": mode_str, "benchmark": "longmemeval", "k": k, "n_conv": 25, "n_qa": 25},
        "result_file": fname,
        "config": {
            "values": {
                "benchmark": {"name": "longmemeval", "split": "oracle"},
                "mode": {"backend": backend, "extractor_model": extractor_model},
                "retriever": {"name": retriever_name},
                "reader": {"model": "qwen2.5:7b-instruct", "provider": "ollama"},
                "k": k,
            }
        },
        "summary": {k: v for k, v in summary.items() if k != "rows"},
    }
    mpath.write_text(json.dumps(manifest_data, indent=2))

    eval_mod.report(summary)
    print(f"Saved result: {p}")
    print(f"Saved manifest: {mpath}")
    return p


def main() -> None:
    # 1. HippoRAG 2 (7B)
    run_sweep(
        backend="hipporag2",
        extractor_model="qwen2.5:7b-instruct",
        out_dir=RESULTS / "paper",
    )

    # 2. LightRAG (7B)
    run_sweep(
        backend="lightrag",
        extractor_model="qwen2.5:7b-instruct",
        out_dir=RESULTS / "paper",
    )

    # 3. HippoRAG 2 (14B)
    run_sweep(
        backend="hipporag2",
        extractor_model="qwen2.5:14b-instruct",
        out_dir=RESULTS / "paper" / "h1_balanced_14b",
    )

    # 4. LightRAG (14B)
    run_sweep(
        backend="lightrag",
        extractor_model="qwen2.5:14b-instruct",
        out_dir=RESULTS / "paper" / "h1_balanced_14b",
    )


if __name__ == "__main__":
    main()
