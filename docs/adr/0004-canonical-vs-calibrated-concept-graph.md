# 0004: Canonical vs. Calibrated Concept Graph Split

We split our Concept Graph (HippoRAG 1) substrate into a strict canonical baseline and an occurrence-calibrated variant, eliminating hardcoded stop-node lists in favor of vectorized math and data-driven frequency filtering.

## Context

Hypothesis 1 and Hypothesis 2 evaluate whether graph-based memory structures provide utility over flat retrieval on conversational benchmarks (`LongMemEval`). Prior implementations of `HippoRAGMemory` contained ad-hoc English stopword lists (`STOP_NODES = {"user", "assistant", ...}`) and scalar Python cosine loops. 

This introduced two major methodological vulnerabilities:
1. **Fidelity Violation:** Hardcoding an arbitrary stopword set makes the baseline unfaithful to the published HippoRAG 1 specification (Jiménez Gutiérrez et al., NeurIPS 2024), which relied purely on prompt extraction and linear inverse specificity ($s_i = 1 / |P_i|$).
2. **Metric Masking:** Manually pruning dialogue pronouns masks natural topological hub formation, distorting the non-parametric diagnostic metrics ($G_{\text{deg}}$, $\max_v \deg(v)/\bar{d}$) measured under Hypothesis 2.

## Decision

We establish three primary decisions to formalize and calibrate the Concept Graph:

1. **Substrate Separation:**
   - **`Concept Graph (Canonical)` (`memory_hipporag_canonical`):** Implements the strict, unmodified HippoRAG 1 algorithm ($G = (N, E \cup E')$). No entity blacklist or occurrence pruning is applied; initial restart weighting follows $s_i = 1 / |P_i|$.
   - **`Concept Graph (Calibrated)` (`memory_hipporag_calibrated`):** Adapts the concept graph to conversational dialogue by removing structural super-hubs via a non-parametric dataset occurrence filter ($|P_v| / |T| > \theta$).
   
2. **Elimination of Hardcoded Stop-Node Sets:**
   All hardcoded lexical dictionaries (`STOP_NODES`) are purged from the memory modules. Any entity appearing in more than 30% of total observed turns ($\theta = 0.30$) in the conversation history is dynamically suppressed as an uninformative dialogue hub.

3. **Vectorized Mathematical Operations:**
   All vector computations and cosine similarities are standardized on NumPy pre-normalized dot products ($A \cdot B^T$), replacing pure-Python loops across all memory backends.

4. **Calibration Delta Metric:**
   Automated evaluation harnesses and table generators will track:
   $$\Delta_{\text{calibration}} = \text{Concept Graph (Calibrated)} - \text{Concept Graph (Canonical)}$$
   directly measuring the empirical penalty imposed by uncalibrated conversational super-hubs.

## Consequences

- The canonical baseline provides an uncontestable, 100% faithful replication of published HippoRAG 1 on conversational dialogue.
- The calibrated variant provides a principled, language-agnostic mechanism to eliminate conversational dialogue super-hubs without ad-hoc word lists.
- Hypothesis 2 regressions can directly compare the topological degradation between the canonical and calibrated graphs.
