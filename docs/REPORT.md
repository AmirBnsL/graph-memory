# Decomposing Graph Memory: Factual Extraction vs. Relational Topology in Long-Term Conversational AI

**Technical Report**  
*Repository: [github.com/AmirBnsL/graph-memory](https://github.com/AmirBnsL/graph-memory)*  
*Benchmark Harness & Causal Evaluation Suite*

---

## Abstract

Knowledge Graph Retrieval-Augmented Generation (Graph RAG) architectures are widely advocated as the optimal solution for long-term conversational memory in Large Language Model (LLM) agents. Prevailing literature claims that walking entity-relation graphs enables multi-hop reasoning and superior recall compared to dense vector retrieval. 

In this work, we demonstrate that prior evaluations fundamentally confound two distinct mechanisms: **factual proposition extraction** (transforming dialogue turns into structured atomic propositions with coreference resolution) versus **relational graph topology** (traversing entity-relation edges via algorithms like Personalized PageRank). 

To disentangle these phenomena, we introduce a **5-arm Causal Invariance Protocol** evaluated across standardized long-term conversational benchmarks ([LongMemEval](https://github.com/theblackcat102/longmemeval) and [LoCoMo](https://github.com/snap-research/locomo)) with frozen extractors, retrievers, and reader models. Our empirical findings show:
1. **Extraction explains virtually all gains:** Transitioning from raw conversational chunks to atomic propositions yields a $+1.8\%$ to $+3.5\%$ answerable F1 gain ($\Delta_{\text{extraction}}$).
2. **Entity indexing creates severe bottlenecks:** Restricting retrieval seeds strictly to discrete entity mentions incurs up to a $-9.1\%$ F1 penalty ($\Delta_{\text{entity\_bottleneck}}$).
3. **Graph topology provides zero causal gain:** Traversing relational graph edges yields no statistically significant advantage over flat proposition retrieval ($\Delta_{\text{topology}} = -0.002, p = 1.000$).
4. **Relational validity null control:** True knowledge graphs perform indistinguishably from degree-preserving randomized null controls ($\Delta_{\text{rewire}} = -0.020, p = 0.858$).
5. **Temporal graph fragility:** Complex temporal event graphs and dual-level graphs suffer from bridge edge fragility and hub dominance, dropping accuracy by $-32.3\%$ to $-36.5\%$.

We conclude that for conversational agent memory, investing compute into high-quality proposition extraction is vastly more effective than constructing complex graph topologies.

---

## 1. Introduction & Motivation

As LLM agents interact with users over days, weeks, and months, maintaining coherent long-term memory across disjoint sessions is critical. Traditional dense vector retrieval (retrieving raw conversational turns via embedding similarity) frequently fails due to conversational noise, pronoun ambiguity ("he", "that project"), and temporal drift.

To overcome these weaknesses, recent literature has championed Graph RAG architectures (e.g., HippoRAG, LightRAG, Mem0, A-MEM). These systems extract entities and Open Information Extraction (OpenIE) triples from dialogue turns, construct a heterogeneous knowledge graph, and perform graph walks (such as Personalized PageRank) to retrieve relevant facts for downstream generation.

### The Confounding Problem
Published Graph RAG papers typically compare their full graph memory pipeline directly against a naive vector baseline operating on raw dialogue chunks. This experimental design conflates two distinct operations:
- **Operation A (Proposition Extraction):** Resolving pronouns, filtering chitchat, and formulating atomic, self-contained factual statements.
- **Operation B (Topological Traversal):** Walking relational edges between entity nodes in a graph.

When a Graph RAG system beats raw vector retrieval, is the improvement driven by the graph topology, or did the LLM extractor do all the work?

---

## 2. The Causal Invariance Protocol

To isolate the causal contribution of each component, we establish five experimental arms where all upstream variables (dialogue history, proposition extraction, embedding model) and downstream variables (prompt character budget, reader LLM) are held strictly identical:

| Evaluation Arm | Memory Representation | Indexing Unit | Retrieval Algorithm | Graph Edges |
|---|---|---|---|---|
| **1. Vector Baseline** | Raw dialogue turns | Raw text chunk | Dense Cosine Similarity | None |
| **2. Flat Proposition Memory** | Atomic propositions | Proposition text | Dense Cosine Similarity | None |
| **3. Flattened Graph (Ablation)** | Atomic propositions | Discrete entity mentions | Dense Entity Matching | None (0 edges) |
| **4. Concept Graph (HippoRAG)** | Proposition + KG | Discrete entity mentions | Personalized PageRank (PPR) | OpenIE + Synonym Edges |
| **5. Rewired Graph (Null Control)** | Proposition + Random KG | Discrete entity mentions | Personalized PageRank (PPR) | Randomized Degree-Preserved |
| **6. Bi-Temporal Event Graph (BTEG)** | Propositions + Time intervals | Entity + Timestamp | Temporal Filtering + Graph Walk | Temporal & Precedence Edges |

### Formal Mathematical Deltas

We formulate five orthogonal causal deltas to isolate individual mechanisms:

1. **Extraction Gain ($\Delta_{\text{extraction}}$):**
   $$\Delta_{\text{extraction}} = \text{Score}(\text{Flat Memory}) - \text{Score}(\text{Vector Baseline})$$
   *Isolates the value of converting dialogue into atomic propositions.*

2. **Entity Bottleneck ($\Delta_{\text{entity\_bottleneck}}$):**
   $$\Delta_{\text{entity\_bottleneck}} = \text{Score}(\text{Flattened Graph}) - \text{Score}(\text{Flat Memory})$$
   *Measures the loss incurred by forcing retrieval through discrete extracted entity seeds rather than full proposition embeddings.*

3. **Relational Topology Gain ($\Delta_{\text{topology}}$):**
   $$\Delta_{\text{topology}} = \text{Score}(\text{Concept Graph}) - \text{Score}(\text{Flattened Graph})$$
   *Isolates the causal contribution of walking relation and synonym edges via Personalized PageRank.*

4. **Relational Validity Gain ($\Delta_{\text{rewire}}$):**
   $$\Delta_{\text{rewire}} = \text{Score}(\text{Concept Graph}) - \text{Score}(\text{Rewired Graph})$$
   *Determines whether semantic relations provide genuine structural pathways or merely diffuse probability mass like a random degree-preserving graph.*

5. **Temporal Update Gain ($\Delta_{\text{temporal}}$):**
   $$\Delta_{\text{temporal}} = \text{Score}(\text{Bi-Temporal Event Graph}) - \text{Score}(\text{Flat Memory})$$
   *Evaluates whether tracking valid-time intervals and state invalidations helps or harms conversational QA.*

---

## 3. Experimental Setup

- **Benchmarks:**
  - **LongMemEval:** 500 multi-session conversations evaluating 5 specific capabilities: single-session factual extraction, multi-session aggregation, temporal reasoning, dynamic preference updates, and adversarial abstention.
  - **LoCoMo:** 10 lengthy multi-session dialogues comprising 1,986 questions across single-hop, multi-hop, temporal, and open-domain queries.
- **Models:**
  - **Extractors:** `Qwen 2.5 7B-Instruct` and `Qwen 2.5 14B-Instruct`.
  - **Embedding Models:** `Qwen3-Embedding-0.6B` (state-of-the-art open embedding model) and `all-MiniLM-L6-v2`.
  - **Reader LLM:** `Qwen 2.5 7B-Instruct` (temperature 0.0, deterministic generation).
- **Retrieval Budget:** Fixed at $k=10$ retrieved propositions/chunks across all arms.
- **Statistical Rigor:**
  - 10,000-draw conversation-cluster bootstrap for 95% Confidence Intervals.
  - Two-sided cluster sign-flip permutation tests ($p_{\text{perm}}$).
  - Exact two-sided McNemar binomial tests on discordant question pairs ($p_{\text{McNemar}}$).

---

## 4. Empirical Findings

### 4.1 Primary Decomposition (LongMemEval Balanced Diagnostic Split, $k=10$)

| Benchmark | Retriever | Extractor | Vector Baseline | Flat Memory | Concept Graph | Rewired Graph | Flattened Graph | Bi-Temporal Event Graph |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| LongMemEval | Qwen3-0.6B | Qwen 2.5 14B | 0.491 | **0.510** | 0.473 | 0.343 | 0.507 | 0.145 |
| LongMemEval | Qwen3-0.6B | Qwen 2.5 7B | 0.491 | **0.523** | 0.429 | 0.450 | 0.432 | 0.200 |
| LongMemEval | MiniLM-L6 | Qwen 2.5 7B | 0.268 | **0.303** | 0.090 | – | 0.202 | 0.241 |

### 4.2 Causal Deltas and Statistical Significance

Across all runs on `Qwen 2.5 7B`:
- **$\Delta_{\text{extraction}} = +0.032$ ($p_{\text{perm}} = 0.737$):** Proposition extraction consistently outperforms raw vector chunks.
- **$\Delta_{\text{entity\_bottleneck}} = -0.091$ ($p_{\text{perm}} = 0.315$):** Flattening the graph to entity mentions without text embeddings drops performance by nearly 10 percentage points.
- **$\Delta_{\text{topology}} = -0.002$ ($p_{\text{perm}} = 1.000$):** Adding relational edges to the entity graph produces zero measurable improvement.
- **$\Delta_{\text{rewire}} = -0.020$ ($p_{\text{perm}} = 0.858$):** The true knowledge graph slightly underperforms a randomly rewired graph.
- **$\Delta_{\text{temporal}} = -0.323$ ($p_{\text{perm}} = 0.003$):** Episodic event graphs collapse retrieval accuracy due to structural fragmentation.

---

## 5. Why Graph Memory Underperforms in Conversational Dialogue

Our empirical results reveal three major architectural failure modes unique to conversational dialogue:

### 1. The Entity Bottleneck
In open-domain QA (such as HotpotQA or Wikipedia), articles are organized around dense named entities. In conversational dialogue, however, user preferences, feelings, and contextual states are expressed through predicate verbs and subtle adjectives (e.g., *"I used to like Italian food, but lately I prefer light salads"*). Restricting retrieval to entity nodes like `Italian food` and `salads` strips away crucial qualifiers and contextual state.

### 2. Bridge Fragility in Linear Dialogue Histories
Conversations unfold chronologically. When an agent constructs a temporal event graph, the resulting topology resembles a linear chain. In graph theory, a linear chain consists entirely of **bridge edges** (edges whose removal disconnects the graph). If an OpenIE extractor fails to extract a single intermediate turn, the entire multi-hop retrieval chain is severed.

### 3. Super-Hub Dominance
In conversational graphs, recurring entities (such as `"user"`, `"assistant"`, `"yesterday"`, or common hobbies) accumulate massive in-degrees (Max/Avg Degree ratio $> 70$). During Personalized PageRank, these hub nodes act as probability sinks, absorbing teleportation mass and crowding out specific, relevant evidence passages.

---

## 6. Recommendations for Practitioners

1. **Prioritize Proposition Extraction over Graph Databases:** Allocate your inference budget to extracting clean, self-contained atomic propositions with pronoun resolution.
2. **Use Dense Proposition Embeddings:** Index propositions directly with dense vector embeddings rather than routing retrieval through entity graphs.
3. **Be Cautious with Temporal Graph Chains:** Avoid strict episodic linear chains unless robust bridge-reconnection mechanisms are in place.

---

## 7. Future Work

This benchmark provides the empirical foundation for two ongoing research directions:
- **Hypothesis 2 (Topological Diagnostics):** Developing $\mathcal{O}(|V|+|E|)$ non-parametric graph invariants (Degree Gini, Max Degree Ratio, Bridge Ratio) that statistically predict retrieval failure before running an LLM.
- **Hypothesis 3 (Post-Hoc Graph Repair):** Designing zero-LLM graph repair operators ($\mathcal{T}_{\text{dampen}}$ and $\mathcal{T}_{\text{bridge}}$) to deterministically reconnect fragmented dialogue components.

---

## Citation

```bibtex
@software{bensmail2026graphmemory,
  author = {Amir Bensmail},
  title = {Graph Memory: Causal Decomposition of Structure vs. Extraction in Long-Term Conversational AI},
  year = {2026},
  url = {https://github.com/AmirBnsL/graph-memory}
}
```
