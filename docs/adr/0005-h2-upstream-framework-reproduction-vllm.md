# 0005: Upstream Framework Reproduction & vLLM Serving for Hypothesis 2

We establish Part B (Unmodified Upstream External Validation) of Hypothesis 2, executing three peer-reviewed conversational memory frameworks (MAGMA, EMem-G, A-MEM) from their official author-published repositories via vLLM on NVIDIA L4 (24GB), inspected via a read-only Topological Sidecar.

## Context

Hypothesis 2 posits that non-parametric graph topological metrics ($G_{\text{deg}}$, $\max_v \deg(v)/\bar{d}$, $\rho_{\text{LCC}}$, density, bridge ratio) quantitatively predict downstream retrieval and QA performance.

In Part A (ADR 0002, ADR 0003, ADR 0004), we evaluated four graph memory architectures (`Concept Graph`, `Bi-Temporal Event Graph`, `Bipartite Concept Graph`, and `Dual-Level Graph`) under strict causal isolation over a frozen in-house OpenIE extraction cache. While this rigorously isolated topology from extraction variance, adapting static document RAG models (HippoRAG, LightRAG) to conversational benchmarks (`LoCoMo`, `LongMemEval`) created an external validity risk that topological failures might be perceived as artifacts of in-house adaptation rather than characteristics of published conversational memory systems.

Furthermore, executing unmodified upstream frameworks across multi-session benchmarks requires thousands of LLM extraction and consolidation calls. Our previous Ollama (`llama.cpp`) serving runtime—with contiguous 32k KV slot reservations, GGUF 4-bit `Q4_K_M` quantization, and sequential execution—bottlenecks throughput (18–30+ hours) on a 24GB GPU and risks silent front-of-prompt truncation.

## Decision

We establish four primary methodological decisions for Part B:

1. **Unmodified Official Upstream Repositories:**
   We benchmark three peer-reviewed conversational memory frameworks cloned at pinned commit SHAs:
   - **Multi-Graph Agentic Memory (MAGMA, ACL 2026):** 4-layer typed multi-graph (Temporal, Causal, Entity, Semantic) over segmented event nodes (`FredJiang0324/MAGMA`).
   - **EDU Heterogeneous Graph (EMem-G & EMem, 2025):** Session–EDU–Argument graph traversed via Personalized PageRank (`KevinSRR/EMem`), alongside its native non-PPR flat ablation (`--skip_retrieval_ppr`) for author-grounded Structural Gain ($\Delta\text{F1}$).
   - **Dynamic Zettelkasten Graph (A-MEM, NeurIPS 2025):** Agentic note graph with LLM-evolved semantic tags and explicit associative links (`WujiangXu/AgenticMemory`).

   Closed-source models (such as RippleMem, which lacks a public repository) and document QA repos lacking native conversational loaders (such as `OSU-NLP-Group/HippoRAG`) are excluded from Part B.

2. **Non-Invasive Topological Sidecar Pattern:**
   Zero modifications are made to upstream graph indexing, prompts, or retrieval traversal logic. A read-only Topological Sidecar (`external_inspector.py`) loads the persisted on-disk graph caches written by each framework (`locomo_trg_cache/` in MAGMA, `igraph` graph in EMem, and `memory_cache_sample_{N}.pkl` in A-MEM), projects them to simple undirected `nx.Graph` instances, computes the 5 non-parametric H2 metrics, and formats the output into canonical run records.

3. **vLLM Serving Architecture on NVIDIA L4 (24GB):**
   - High-throughput serving via `vllm serve` on port 8000 using PagedAttention, iteration-level continuous batching, and Automatic Prefix Caching (`--enable-prefix-caching`).
   - Hard `--max-model-len 32768` context window enforcement with XGrammar guided JSON decoding (`--guided-decoding-backend xgrammar`).
   - Explicit `--gpu-memory-utilization 0.80` allocating ~19.2 GB to vLLM, reserving ~4.8 GB of L4 VRAM for local in-process embedding models (`all-MiniLM-L6-v2`, `BAAI/bge-m3`, `Qwen3-Embedding-0.6B`).
   - Native Ada Lovelace W8A8 FP8 Tensor Core execution for 14B models (`Qwen2.5-14B-Instruct-FP8`) and BF16/FP8 for 7B (`Qwen2.5-7B-Instruct`).
   - Full backward compatibility: `"vllm"` is added as a first-class preset in `core/llm.py` while `"ollama"` is preserved.

4. **Standardized Evaluation Scope ($N=360$ Upstream Natural Observations):**
   - **Full LoCoMo:** All 10 conversations (1,986 QA pairs).
   - **Stratified LongMemEval-50:** A deterministic, seed-pinned ($seed=42$) subset of 50 conversations (10 per category across extraction, multi-session, temporal, update, and abstention), exported in both native LongMemEval and LoCoMo schemas.

## Consequences

- H2 validation is partitioned into Part A (Controlled Causal Isolation, $n=160$) and Part B (Unmodified Upstream Validation, $N=360$), resolving reproduction faithfulness.
- Upstream graph caches are directly auditable and reproducible without modifying third-party packages.
- Both Ollama and vLLM serving backends are supported across local and Colab environments.
