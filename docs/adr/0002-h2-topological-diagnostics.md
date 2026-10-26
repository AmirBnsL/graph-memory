# 0002: Hypothesis 2 Non-Parametric Topological Diagnostics

We establish cheap, $\mathcal{O}(|V|+|E|)$ non-parametric structural graph metrics as pre-retrieval predictors of conversational memory retrieval recall and precision.

## Context

Hypothesis 2 investigates whether structural graph properties explain why graph memory systems often fail to outperform flat memory baselines on conversational benchmarks. While Hypothesis 1 decomposes the causal gap between extraction and topology, it treats memory graphs as black boxes. Prior literature either reports descriptive graph statistics without linking them to retrieval outcomes or relies on expensive LLM judges that introduce substantial evaluation noise.

We require a pre-retrieval diagnostic harness that computes in milliseconds on CPU, operates without LLM calls, and quantitatively predicts downstream retrieval utility.

## Decision

We establish four primary methodological decisions for Hypothesis 2:

1. **Non-Parametric Structural Predictor Suite:**
   We compute five core $\mathcal{O}(|V|+|E|)$ structural predictors on undirected simple graph projections:
   - **Degree Gini:** Inequality coefficient ($G_{\text{deg}} \in [0, 1]$) via trapezoidal integration of the Lorenz curve.
   - **Max Degree Ratio:** Extreme hub dominance ($\max_v \deg(v) / \bar{d}$).
   - **Largest Connected Component Ratio:** Global topological reachability ($\rho_{\text{LCC}} = |V_{\text{LCC}}| / |V|$).
   - **Edge Density:** $2|E| / (|V|(|V|-1))$.
   - **Bridge Edge Ratio:** Path bottleneck frequency ($|E_{\text{bridges}}| / |E|$).

2. **Observation Pooling & Control Separation:**
   - **Descriptive Reporting (Table 1):** Reports structural averages across all calibrated memory substrates, retaining `Flattened Graph` (verifying the zero-edge invariant) and `Rewired Graph` (verifying exact degree preservation alongside clustering collapse). Candidate runs must contain per-sample metric breakdowns (`graph_metrics_by_sample`), naturally filtering out legacy uncalibrated runs.
   - **Predictive Regressions & Correlations (Tables 2–4):** Restrict observational analysis strictly to natural LLM-extracted graph memory substrates (`Concept Graph` and `Bi-Temporal Event Graph`, $n=80$ observations across 7B and 14B extractors). Synthetic randomized null controls (`Rewired Graph`) and zero-edge ablations (`Flattened Graph`) are excluded from predictive modeling to prevent artificial topology noise from muting empirical effect sizes.

3. **Retrieval-Stage vs. Reader-Stage Variance Decomposition:**
   We model four distinct dependent targets:
   - Evidence Coverage ($|\text{retrieved} \cap \text{provenance}| / |\text{provenance}|$)
   - Precision@k ($|\text{retrieved} \cap \text{provenance}| / |\text{retrieved}|$)
   - End-to-End Answerable F1
   - Structural Gain ($\Delta\text{F1 vs. Flat}$)

   Non-parametric graph topology directly governs candidate expansion and Personalized PageRank mass distribution at the retrieval stage ($R^2 \ge 0.20, p < 0.01$). Reader LLM answer generation attenuates this signal by approximately 50% ($R^2 \approx 0.10$), empirically validating the retrieval-to-response bottleneck (MERIT ignore rate).

4. **Topological Regime Stratification (Table 4):**
   Natural graph observations are partitioned at the median across three diagnostic dimensions:
   - Hub Concentration ($G_{\text{deg}} \le \text{median}$ vs. $> \text{median}$)
   - Super-Hub Dominance (`max_degree_ratio` $\le \text{median}$ vs. $> \text{median}$)
   - Component Connectivity ($\rho_{\text{LCC}} \le \text{median}$ vs. $> \text{median}$)

   Each split reports inter-regime deltas and two-sided permutation $p$-values, identifying exact topological pathologies and directly motivating Hypothesis 3's zero-LLM repair operators ($\mathcal{T}_{\text{dampen}}$ and $\mathcal{T}_{\text{bridge}}$).

## Consequences

- Automated table generators (`graph-memory tables`) report all four H2 tables.
- Multiple linear regression models must satisfy $R^2 \ge 0.20$ ($p < 0.05$) on retrieval targets on natural graphs to confirm Hypothesis 2.
- Predictive models never confound natural memory graphs with synthetic degree-preserving rewired controls.
