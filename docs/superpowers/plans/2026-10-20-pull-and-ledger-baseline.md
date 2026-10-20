# Ingest LongMemEval Baseline Sweep & Update Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pull the completed LongMemEval vector RAG sweep ($k=5, 10, 25$) from the Colab L4 runtime into the local repository, regenerate the ledger via `src/make_tables.py`, document the benchmark baseline in `results/INDEX.md`, push to GitHub `origin/master`, and stop the Colab GPU VM.

**Architecture:**
- Leverage `colab/pull_results.py` to pull immutable result `.json` and `.manifest.json` files from the remote session `/content/graph-memory/results/` to `results/`.
- Use `src/make_tables.py` to generate the Markdown tables for both LoCoMo and LongMemEval across all $k$.
- Synthesize an empirical analysis in `results/INDEX.md` detailing the 5 LongMemEval ability categories (Extraction, Multi-session, Temporal, Update, Abstention).
- Push all versioned artifacts to GitHub and safely decommission the Colab VM.

**Tech Stack:** Python 3.12, Google Colab CLI, Git, Hydra, Ollama, Qwen2.5-7B-Instruct, Qwen3-Embedding-0.6B.

**Spec:** `docs/superpowers/plans/2026-10-20-longmemeval-loader.md` (Task 5).

---

## Global Constraints

- **Dataset Provenance:** Must align with `provenance/datasets.json` (SHA256: `821a2034...`).
- **Category 5 Invariant:** Preserved as Refusal/Abstention (`_abs`), aligning with LoCoMo refusal accuracy.
- **Traceability:** Every reported metric in `results/INDEX.md` must be traceable to a committed `.json` and `.manifest.json`.
- **GPU Cost Discipline:** Colab VM `gm-dev` must be cleanly stopped upon task completion (`colab stop -s gm-dev`).

---

## Proposed Changes

### Component 1: Result Ingestion & Verification

#### [NEW] `results/20261020-131028_longmemeval_vector_k5_qwen2.5:7b-instruct.json` + manifest
#### [NEW] `results/20261020-131659_longmemeval_vector_k10_qwen2.5:7b-instruct.json` + manifest
#### [NEW] `results/20261020-133126_longmemeval_vector_k25_qwen2.5:7b-instruct.json` + manifest

- Ingest all completed run files from remote `/content/graph-memory/results/` to local `results/`.
- Validate that all 3 runs contain 500 rows, matching the full Oracle dataset.

---

### Component 2: Ledger Documentation & Analysis

#### [MODIFY] `results/INDEX.md`

Add dedicated section for **LongMemEval Vector RAG Baseline**:
- Overall table comparing $k \in \{5, 10, 25\}$ against LoCoMo.
- Per-category recall and F1 breakdown across the 5 core abilities:
  1. Single-session extraction (Cat 1, $n=150$)
  2. Multi-session reasoning (Cat 2, $n=121$)
  3. Temporal reasoning (Cat 3, $n=127$)
  4. Knowledge updates (Cat 4, $n=72$)
  5. Abstention (Cat 5, $n=30$)
- Structural observations:
  - Strong single-session extraction ($F_1 \approx 0.63$ at $k=25$).
  - Retrieval vs reasoning bottleneck: Near-perfect retrieval ($recall@k = 1.000$) yet low answer F1 on multi-session ($0.062$), temporal ($0.097$), and update ($0.030$), highlighting the classical needle-in-haystack synthesis failure of bare vector RAG.

---

### Component 3: Ledger & State Tracking

#### [MODIFY] `.superpowers/sdd/2026-10-20-longmemeval-loader/progress.md`

Mark Task 4 and Task 5 as complete.

---

## Plan Tasks

### Task 1: Pull Results & Validate Integrity

- [ ] Run `python3 colab/pull_results.py --session gm-dev --fresh` to pull all result files and manifests.
- [ ] Confirm presence of $k=5$, $k=10$, and $k=25$ result files and manifests in `results/`.
- [ ] Run `.venv/bin/python src/make_tables.py` and inspect generated Markdown tables for accuracy.

### Task 2: Document LongMemEval Baseline in `results/INDEX.md`

- [ ] Incorporate `make_tables.py` output into `results/INDEX.md`.
- [ ] Add narrative documentation explaining the LongMemEval experiment configuration:
  - Reader: `qwen2.5:7b-instruct` (Q4_K_M via Ollama, 8k context window).
  - Retriever: `Qwen3-Embedding-0.6B` (bare query, last-token pooling).
  - Benchmark split: Oracle (500 questions, 10,960 turns).
- [ ] Document the per-ability findings and the synthesis bottleneck.

### Task 3: Commit, Push, and Decommission Colab VM

- [ ] Stage and commit all new results, manifests, and documentation updates.
- [ ] Push to GitHub `origin/master`.
- [ ] Decommission Colab VM: `colab stop -s gm-dev` and verify `colab ls` reflects 0 active sessions.
- [ ] Update SDD progress ledger `.superpowers/sdd/2026-10-20-longmemeval-loader/progress.md`.

---

## Verification Plan

### Automated Verification:
- `python3 -m unittest discover -s tests` must pass cleanly.
- `python3 src/make_tables.py` must run without error and format both LoCoMo and LongMemEval sections.
- Git status must be clean after push (`git status` -> `nothing to commit, working tree clean`).
- `colab ls` must report no active sessions after stopping the VM.
