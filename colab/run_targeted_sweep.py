#!/usr/bin/env python3
"""Run targeted balanced H1 decomposition sweep on Colab GPU.

Samples 5 questions from each of the 5 LongMemEval core abilities:
  1. extraction
  2. multi-session
  3. temporal
  4. update
  5. abstention
Total = 25 questions, perfectly balanced.

Runs:
  - Step 1: Pre-extract turns in parallel (workers=4) -> cache/extract/
  - Step 2: Arm 0 - Vector RAG baseline (k=10)
  - Step 3: Arm 1 - Flat Extracted Facts (k=10, H1 non-graph control)
  - Step 4: Arm 2 - HippoRAG OpenIE KG (k=10)
  - Step 5: Arm 3 - HippoRAG Flattened Structure Ablation (k=10)
  - Step 6: Arm 4 - Bi-Temporal Edge Graph (k=10)
  - Step 7: Print H1 decomposition table
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

REPO = "/content/graph-memory"
os.chdir(REPO)

env = os.environ.copy()
env["LLM_PROVIDER"] = "ollama"
env["LLM_MODEL"] = "qwen2.5:7b-instruct"
env["PYTHONUNBUFFERED"] = "1"

def run_cmd(cmd: list[str], desc: str) -> None:
    t0 = time.time()
    print(f"\n============================================================", flush=True)
    print(f"==> {desc}", flush=True)
    print(f"    CMD: {' '.join(cmd)}", flush=True)
    print(f"============================================================", flush=True)
    res = subprocess.run(cmd, env=env, cwd=REPO)
    if res.returncode != 0:
        print(f"ERROR: command failed with code {res.returncode}", flush=True)
        sys.exit(res.returncode)
    print(f"==> Done in {time.time() - t0:.1f}s", flush=True)

def main() -> None:
    print("Starting Targeted Balanced Sweep (25 questions, 5 per ability)...", flush=True)

    # Step 1: Parallel Pre-Extraction (Already cached on disk; will verify in <1s)
    run_cmd(
        [sys.executable, "src/extract.py", "--benchmark", "longmemeval", "--sample-per-cat", "5", "--workers", "4"],
        "Step 1: Parallel Pre-Extraction of 25 Balanced Conversations",
    )

    # Step 2: Arm 0 - Vector Baseline
    run_cmd(
        [sys.executable, "src/eval.py", "benchmark=longmemeval", "mode=vector", "k=10", "+limits.sample_per_category=5"],
        "Step 2: Arm 0 - Vector Baseline (Raw RAG, k=10)",
    )

    # Step 3: Arm 1 - Flat Facts (H1 Control)
    run_cmd(
        [sys.executable, "src/eval.py", "benchmark=longmemeval", "mode=memory", "mode.backend=flat", "k=10", "+limits.sample_per_category=5"],
        "Step 3: Arm 1 - Flat Extracted Facts (H1 Non-Graph Control, k=10)",
    )

    # Step 4: Arm 2 - HippoRAG OpenIE KG
    run_cmd(
        [sys.executable, "src/eval.py", "benchmark=longmemeval", "mode=memory", "mode.backend=hipporag", "k=10", "+limits.sample_per_category=5"],
        "Step 4: Arm 2 - HippoRAG OpenIE KG (k=10)",
    )

    # Step 5: Arm 3 - HippoRAG Flattened Structure Ablation
    run_cmd(
        [sys.executable, "src/eval.py", "benchmark=longmemeval", "mode=memory", "mode.backend=hipporag", "mode.flatten_structure=true", "k=10", "+limits.sample_per_category=5"],
        "Step 5: Arm 3 - HippoRAG Flattened Structure Ablation (k=10)",
    )

    # Step 6: Arm 4 - Bi-Temporal Edge Graph
    run_cmd(
        [sys.executable, "src/eval.py", "benchmark=longmemeval", "mode=memory", "mode.backend=temporal_bteg", "k=10", "+limits.sample_per_category=5"],
        "Step 6: Arm 4 - Bi-Temporal Edge Graph (k=10)",
    )

    # Step 7: Make tables
    run_cmd(
        [sys.executable, "src/make_tables.py"],
        "Step 7: Regenerate Ledger & H1 Decomposition Table",
    )

    print("\nALL 5 ARMS COMPLETED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    main()
