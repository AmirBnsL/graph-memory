# Week 2 Implementation Plan: Graph Memory Backends & The H1/H3 Decomposition

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute Week 2 of the Master's PFE by implementing the verified graph memory architectures (HippoRAG NeurIPS 2024 and Bi-Temporal Knowledge Graph), wiring the H3 topological health metrics (`src/graph_metrics.py`), and running the controlled **H1 (Structure vs. Extraction)** decomposition on LongMemEval and LoCoMo.

**Obsidian Source of Truth:**
* Detailed Vault Note: [`06-experiment-design/Week 2 Plan — Graph Backends & H1 Decomposition.md`](file:///home/amirbnsl/Documents/Obsidian%20Vault/graph-memory-agents/06-experiment-design/Week%202%20Plan%20—%20Graph%20Backends%20%26%20H1%20Decomposition.md)
* Linked Roadmap: [`00-dashboard.md`](file:///home/amirbnsl/Documents/Obsidian%20Vault/graph-memory-agents/00-dashboard.md)

---

## 1. Verified Architecture Specifications

### Extraction Layer (`src/extract.py`)
* Implements HippoRAG's 2-step extraction protocol via 1-shot prompts (§2.3 & Appendix I):
  * **Step 1:** Passage NER extracting salient entities (`{"named_entities": [...]}`).
  * **Step 2:** OpenIE extracting relation triples (`{"triples": [["subject", "predicate", "object"], ...]}`) and atomic factual propositions.
* **Deterministic Hashing:** Cached to `cache/extract/<hash>.json` using `sha256(model | prompt_template | turn_text)`. All downstream backends read from this cache without paying for extra LLM calls.

### Backend A: Flat Extracted Facts (The H1 Non-Graph Control)
* Dense vector retrieval over the extracted atomic facts using `Qwen3-Embedding-0.6B`.
* Zero graph edges, zero multi-hop traversal.
* Measures: $\Delta_{\text{extraction}} = \text{F1}_{\text{flat}} - \text{F1}_{\text{raw\_rag}}$.

### Backend B: HippoRAG Concept Graph + PPR (NeurIPS 2024)
* **Graph:** $G = (N, E \cup E')$ where $N$ are entities/noun phrases, $E$ are OpenIE relation triples, and $E'$ are dense synonymy edges where $\text{cosine}(e_i, e_j) > \tau$.
* **Occurrence Matrix ($P \in \mathbb{R}^{|N| \times |P|}$):** Tracks passage co-occurrences.
* **Node Specificity:** Query entity restart probabilities weighted by $s_i = 1 / |P_i|$.
* **Personalized PageRank:** Runs on $G$; stationary distribution $\vec{n}'$ projected to passages via:
  $$\vec{p} = \vec{n}'^\top P$$

### Backend C: Bi-Temporal Knowledge Graph (Zep / Graphiti Formulation)
* **Bi-temporal Edges:** Carries real-world timeline $(t_{\text{valid}}, t_{\text{invalid}})$ and ingestion timeline $(t'_{\text{created}}, t'_{\text{expired}})$.
* **Edge Invalidation:** Detects contradictory facts across turns and invalidates older edges ($t_{\text{invalid}}(\text{old}) = t_{\text{valid}}(\text{new})$).
* **Retrieval:** Multi-hop BFS neighborhood expansion constrained by temporal validity windows.

### H3 Topological Health Calculator (`src/graph_metrics.py`)
* Computes non-parametric graph metrics:
  * Largest connected component ratio ($LCC / |V|$).
  * Degree variance and hub Gini coefficient.
  * Graph density and average clustering coefficient.
  * Temporal validity coverage and conflict edge ratio.

---

## 2. Proposed Changes

### Task 1: Ingestion & Deterministic Extraction
- [ ] Create `src/extract.py` implementing 2-step OpenIE extraction with disk caching
- [ ] Create `tests/test_extract.py` validating extraction formatting, fallback handling, and hash persistence
- [ ] Pre-extract LongMemEval dataset turns on Colab L4/A100

### Task 2: Unified Memory Interface & Backends
- [ ] Create `src/memory/base.py` defining `BaseMemory` abstract class
- [ ] Create `src/memory/flat.py` (Backend A: Flat Extracted Facts)
- [ ] Create `src/memory/hipporag.py` (Backend B: HippoRAG OpenIE KG + PPR + Projection)
- [ ] Create `src/memory/temporal_bteg.py` (Backend C: Bi-Temporal Knowledge Graph with Invalidation)
- [ ] Create `tests/test_memory_backends.py`

### Task 3: H3 Graph Metrics & Structural Controls
- [ ] Create `src/graph_metrics.py` (LCC, Hub Gini, Density, Temporal Coverage)
- [ ] Create `tests/test_graph_metrics.py`
- [ ] Wire metrics into manifest logging and add ablation flags (`memory.flatten_structure`, `memory.randomize_edges`)

### Task 4: Harness Integration & The H1 Decomposition Run
- [ ] Update Hydra configuration (`conf/config.yaml`, `conf/mode/memory.yaml`)
- [ ] Update `src/answer.py` to route through the unified memory interface
- [ ] Run smoke tests locally (`limits.n_qa=5`) across all backends
- [ ] Launch full H1 decomposition sweep on Colab GPU
- [ ] Pull results, regenerate tables via `src/make_tables.py`, and commit to `results/INDEX.md`

---

## 3. Verification Plan

### Automated Tests
1. **Extraction Unit Tests:** `uv run python -m unittest tests/test_extract.py`
2. **Backend Unit Tests:** `uv run python -m unittest tests/test_memory_backends.py`
3. **Graph Metrics Tests:** `uv run python -m unittest tests/test_graph_metrics.py`
4. **End-to-End Local Smoke Test:**
   ```bash
   uv run python src/eval.py benchmark=longmemeval mode=memory memory=flat limits.n_qa=5
   uv run python src/eval.py benchmark=longmemeval mode=memory memory=hipporag limits.n_qa=5
   uv run python src/eval.py benchmark=longmemeval mode=memory memory=temporal_bteg limits.n_qa=5
   ```

### Manual Verification
1. Inspect extracted triples in `cache/extract/` for entity cleanliness and pronoun resolution.
2. Verify in `results/INDEX.md` that the H1 decomposition table cleanly separates:
   $$\Delta_{\text{extraction}} = \text{F1}_{\text{flat}} - \text{F1}_{\text{raw\_rag}}$$
   $$\Delta_{\text{structure}} = \text{F1}_{\text{hipporag}} - \text{F1}_{\text{flat}}$$
