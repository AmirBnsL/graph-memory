# 0001: Three-Way Hypothesis 1 Decomposition

We decompose the performance gap between Flat Memory and Concept Graph into two distinct, measurable phenomena: entity-level bottlenecking and relational topological traversal.

## Context

Hypothesis 1 investigates whether complex graph memory architectures provide genuine utility over flat proposition retrieval and raw vector baselines. Prior work often compares full entity-graph traversal directly against flat retrieval, confounding surface-form loss (restricting retrieval to discrete entity seeds) with the actual benefit of topological multi-hop traversal.

## Decision

We separate the evaluation of memory substrates into three primary contrasts:

1. **Extraction Gain ($\Delta_{\text{extraction}} = \text{Flat Memory} - \text{Vector Baseline}$):**
   Measures the benefit of converting raw dialogue turns into structured atomic propositions with pronoun resolution.
2. **Entity Bottleneck ($\Delta_{\text{entity\_bottleneck}} = \text{Flattened Graph} - \text{Flat Memory}$):**
   Measures the loss incurred by indexing and seeding on discrete extracted entities rather than dense proposition embeddings.
3. **Relational Topology Gain ($\Delta_{\text{topology}} = \text{Concept Graph} - \text{Flattened Graph}$):**
   Measures the pure causal contribution of walking relation and synonym edges via Personalized PageRank, isolated from entity occurrence and specificity weighting.
4. **Relational Validity Gain ($\Delta_{\text{rewire}} = \text{Concept Graph} - \text{Rewired Graph}$):**
   Measures whether relational edges encode genuine semantic connections rather than non-specific graph density.
5. **Temporal Update Gain ($\Delta_{\text{temporal}} = \text{Bi-Temporal Event Graph} - \text{Flat Memory}$):**
   Measures the contribution of valid-time interval tracking and dynamic predicate invalidation on dynamic state updates.

## Consequences

- Automated table generators (`graph-memory tables`) report all five deltas across benchmarks.
- A single overall F1 number cannot falsely mask relational gains on multi-session queries or hub distortion on single-session queries.
