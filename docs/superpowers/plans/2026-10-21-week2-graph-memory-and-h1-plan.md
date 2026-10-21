# Week 2 Implementation Plan: Graph Memory Backends & The H1/H3 Decomposition

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transition from the completed vector RAG baseline into **Week 2** of the Master's PFE: build the unified memory extraction pipeline (`src/extract.py`), implement three pluggable memory backends (`flat_extracted`, `hipporag_ppr`, `zep_temporal_bfs`), compute H3 topological statistics (`src/graph_metrics.py`), and execute the first controlled **H1 (Structure vs. Extraction)** decomposition on LongMemEval and LoCoMo.

**Architecture:**
1. **Extraction Layer (`src/extract.py`):** Converts dialogue turns into atomic factual propositions and entity-relation triples with strict timestamp and provenance tracking. Cached deterministically to disk (`cache/extract/`) so LLM extraction runs exactly once.
2. **Unified Memory Interface (`src/memory/base.py`):** An abstract base interface (`ingest`, `retrieve`, `export_graph`) implemented identically across:
   - **Backend A (Flat Extracted / Non-Graph):** Dense retrieval over extracted atomic facts (the ENGRAM / Mem0-flat control).
   - **Backend B (Entity Graph + PPR):** Bipartite entity-chunk graph with Personalized PageRank diffusion (HippoRAG style).
   - **Backend C (Temporal KG + BFS):** Bi-temporal validity windows ($t_{\text{valid}}, t_{\text{recorded}}$) and BFS multi-hop traversal with update invalidation (Zep / Graphiti style).
3. **Topological Metrics Layer (`src/graph_metrics.py`):** Calculates cheap non-parametric structural statistics (largest component ratio, degree variance / hub Gini, edge density, temporal coverage) to test **H3**.
4. **Controlled Ablation Engine (`src/answer.py` & `src/eval.py`):** Integrates the backends into Hydra (`mode=memory memory.backend=...`) with structure-flattened and randomized-edge controls.

---

## Global Constraints

- **The Extractor-Swap Control (H1 Core):** All graph vs. flat comparisons MUST run on the *exact same underlying extracted factual propositions*. Any comparison where the graph system and vector system use different extraction inputs is scientifically invalid.
- **Disk-Caching of Extraction:** LLM extraction is expensive. Every extraction call must be hashed and cached to disk. Rerunning backends or scoring must cost zero additional extraction calls.
- **Local Source of Truth / Ephemeral VM:** Local checkout is source of truth. All code is authored locally, committed, bundled, and run on Colab GPU via `colab exec`.
- **Primary Benchmark:** LongMemEval (500 questions, oracle split) is the primary benchmark; LoCoMo (10 conversations, 1986 questions) is the secondary benchmark.
- **Reporting Standard:** Every table must report token $F_1$, $F_1\text{ paper}$, Exact Match, Recall@$k$, MERIT ignore rate, and per-category metrics.

---

## Review Focus

1. **Extraction Granularity:** Does prompt extraction produce clean atomic facts (e.g. `[2023-05-10] User bought a red Honda Civic (session_1, turn_4)`) and triples, or does it lose crucial temporal anchors?
2. **Deterministic Graph Construction:** Can a NetworkX graph be deterministically constructed from the cached extractions, saved to graphml/JSON, and reloaded without loss of edge attributes?
3. **Separation of Structure vs. Extraction:** Does Backend A (flat vector search over atomic facts) beat raw-turn vector RAG ($0.500 \to ?$), and does Backend B (PPR on graph) beat Backend A?
4. **H3 Metric Predictability:** Do graph topology metrics (e.g., disconnected components or extreme hub concentration) correlate with QA failures and high MERIT ignore rates?

---

## Proposed Changes

### Task 1: The Unified Extraction Pipeline (`src/extract.py`)
- [ ] Create `src/extract.py` with structured schema and LLM disk caching
- [ ] Create `tests/test_extract.py` with parsing and determinism tests
- [ ] Run test suite and verify cache keys

### Task 2: Unified Memory Interface & Backends (`src/memory/`)
- [ ] Create `src/memory/base.py` ABC
- [ ] Create `src/memory/flat.py` (Backend A: Flat Extracted Facts)
- [ ] Create `src/memory/graph_ppr.py` (Backend B: HippoRAG PPR)
- [ ] Create `src/memory/temporal_bfs.py` (Backend C: Zep Temporal BFS)
- [ ] Create `tests/test_memory_backends.py`

### Task 3: H3 Topological Quality Metrics (`src/graph_metrics.py`)
- [ ] Create `src/graph_metrics.py` (LCC, Hub Gini, Density, Temporal Coverage)
- [ ] Create `tests/test_graph_metrics.py`
- [ ] Wire metrics into run manifest output

### Task 4: Harness Wiring & The H1 Decomposition Run
- [ ] Update `conf/config.yaml` to support `mode=memory`
- [ ] Update `src/answer.py` to route through `BaseMemory`
- [ ] Run local smoke test (`limits.n_qa=5`) across all backends
- [ ] Build Colab sweep driver `colab/run_h1_sweep.py`
