# Vault Knowledge Base Expansion & 32k Context LongMemEval k=25 Rerun Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author comprehensive research notes in the Obsidian Vault covering the memory systems landscape (Memory-R2, Mem0, Zep, A-MEM, LangMem), apply the 32k context fix to the Ollama serving stack on Colab L4 (`gm-dev`), and execute an un-truncated $k=25$ LongMemEval evaluation to establish our uncompromised baseline.

**Architecture:**
1. **Vault Notes Layer:** Formalize the structural vs. extraction findings into Obsidian notes (`A-MEM`, `LangMem and Engineering Frameworks`, `Memory Systems Landscape (2026)`), cross-linked to [[00-START-HERE]] and [[00-dashboard]].
2. **Serving Layer (Colab L4):** Define an explicit Ollama Modelfile with `PARAMETER num_ctx 32768` for `qwen2.5:7b-instruct`, eliminating the silent 4096-token front-truncation bug.
3. **Evaluation Layer:** Synchronize configs (`conf/reader/qwen25-7b.yaml` num_ctx 32768), execute LongMemEval $k=25$, pull durable result JSONs, and update the ledger (`results/INDEX.md`).

**Tech Stack:** Python 3.10+, Hydra, Ollama (Llama.cpp backend), PyTorch, Colab CLI (`colab exec`), Obsidian Vault GFM.

---

## Global Constraints

- **Single-SSH Rule:** Colab allows only one active SSH WebSocket tunnel. The user is connected via Zed editor; all remote executions must use `colab exec -s gm-dev` or local orchestration scripts (`colab/pull_results.py`).
- **Zero Hallucination / Scoring Transparency:** Every benchmark metric reported must explicitly declare its scoring convention (`F1_paper` vs `F1_answerable` vs `EM`), retriever backbone, $k$, and evaluation turn count.
- **Cache Integrity:** Never change the model identifier or prompt structure mid-run without updating the provenance manifest (`src/manifest.py`).
- **No Git SSH Keys on VM:** Keep code deployment strictly through git fast-forward from uploaded bundles or remote tracking without storing credentials on the ephemeral VM.

---

## Review Focus

1. **Context Window Verification:** Ensure Ollama actually allocates 32,768 tokens (KV cache ~1.9 GB on L4 GPU) and does not silently fall back to 4096 or 8192 tokens. Test with an explicit synthetic needle prompt (>5000 tokens) before starting the benchmark.
2. **Deterministic Cache Invalidation:** The previous $k=25$ run was clamped at 4096 tokens. Ensure cached LLM responses from the truncated run are cleared or differentiated by `num_ctx: 32768` so new queries evaluate against the full retrieved context.
3. **Category Breakdown Recovery:** In the truncated $k=25$ run, Cat 1 (Extraction) collapsed from 0.677 ($k=10$) to 0.248 ($k=25$). The clean 32k run must recover or exceed Cat 1 performance ($>0.68$).
4. **Vault Note Bi-directional Linking:** Ensure all new vault notes link properly to `[[00-START-HERE]]`, `[[00-dashboard]]`, `[[H1–H4]]`, and `[[Memory-R2 (2605.21768)]]`.

---

## Proposed Changes

### Task 1: Author New Obsidian Vault Notes
- [x] Create `02-primary-papers/A-MEM (2502.12110).md`
- [x] Create `02-primary-papers/LangMem and Engineering Frameworks.md`
- [x] Create `01-surveys/Memory Systems Landscape — Flat Banks vs Graphs (2026).md`
- [x] Update `00-START-HERE.md` and `00-dashboard.md` with pointers to these notes

### Task 2: Configure 32k Context Window in Codebase & Remote VM
- [x] Update `conf/reader/qwen25-7b.yaml` with `num_ctx: 32768`
- [x] Create Modelfile with `PARAMETER num_ctx 32768` on Colab VM (`gm-dev`)
- [x] Build model: `ollama create qwen2.5:7b-instruct -f Modelfile`
- [x] Run preflight context needle test with >5,000 tokens on VM

### Task 3: Execute LongMemEval $k=25$ & Update Ledger
- [x] Run `colab exec -s gm-dev` with LongMemEval $k=25$
- [x] Monitor evaluation to completion
- [x] Pull results via `python3 colab/pull_results.py --session gm-dev`
- [x] Run `python3 src/make_tables.py` and update `results/INDEX.md`
- [x] Commit and push changes to GitHub `origin/master` (commit `380fa9d`)
