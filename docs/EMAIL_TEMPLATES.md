# Outreach Email Templates: Presenting `graph-memory`

Use these tailored templates when sharing your repository with research labs, professors, recruiters, and engineering hiring managers.

---

## Template 1: For Academic PIs, Professors, & Research Labs

**Best for:** Reaching out to academic advisors, PhD applications, research scientist positions, or lab PIs working on LLMs, agent memory, or Graph RAG.

**Subject:** Empirical evaluation of Graph RAG vs. Flat Proposition Memory on long-term dialogue benchmarks

```text
Dear Professor [Last Name] / Dr. [Last Name],

I have been following your lab's work on [relevant topic, e.g., long-context LLMs / agent memory / retrieval-augmented generation], particularly your recent paper on [Paper Title or Topic].

I recently completed an empirical benchmark investigating a fundamental question in conversational agent memory: Does knowledge graph structure actually improve retrieval, or does factual proposition extraction do all the heavy lifting?

While recent systems (HippoRAG, LightRAG, Mem0) advocate for knowledge graph memory, existing evaluations often confound proposition extraction (pronoun resolution, atomic fact extraction) with topological traversal (Personalized PageRank). 

To isolate these mechanisms, I built Graph Memory (https://github.com/AmirBnsL/graph-memory), a causal evaluation harness evaluated across LongMemEval and LoCoMo:
• We established a 5-arm causal ablation protocol holding extractors (Qwen 2.5 7B/14B) and retrievers (Qwen3-Embedding-0.6B) strictly frozen.
• Key Finding: Converting dialogue into atomic propositions accounts for essentially all retrieval gains (+1.8% to +3.5% F1). Graph topology traversal over extracted knowledge graphs yields no statistically significant improvement over flat proposition retrieval (Δ_topology = -0.002, p = 1.000).
• Null Control: Concept graphs perform indistinguishably from degree-preserving randomized graphs (Δ_rewire = -0.020, p = 0.858), demonstrating that OpenIE relational edges in conversational settings do not provide meaningful multi-hop pathways.
• Temporal Collapse: Linear temporal event graphs suffer from extreme bridge edge fragility (>55% bridges), causing severe retrieval collapse (-32.3% F1).

The repository includes a complete test suite (74 tests passing), deterministic evaluation manifests, and a full technical write-up:
https://github.com/AmirBnsL/graph-memory/blob/master/docs/REPORT.md

I would welcome the opportunity to discuss these findings or explore how this evaluation framework could contribute to ongoing projects in your group.

Best regards,

[Your Name]
[Your Contact Information / LinkedIn / Scholar Profile]
```

---

## Template 2: For AI/ML Engineering Hiring Managers & Technical Recruiters

**Best for:** Applying to Machine Learning Engineer, LLM Systems Engineer, or AI Research Engineer roles.

**Subject:** ML Systems Portfolio: Causal evaluation harness for LLM Agent Memory architectures

```text
Hi [Hiring Manager Name / Recruiter Name],

I am applying for the [Role Title, e.g., Machine Learning Engineer / LLM Systems Engineer] position at [Company]. Given your team’s focus on [relevant product or challenge, e.g., conversational agents / agentic workflows / RAG systems], I wanted to share a recent open-source project I developed:

Graph Memory (https://github.com/AmirBnsL/graph-memory) is an extensible evaluation harness and benchmarking platform for long-term LLM agent memory architectures.

Engineering & Systems Highlights:
• Clean, Modular Python Package: Architected with separation of concerns across core primitives, pluggable memory backends (Flat, HippoRAG, Temporal BTEG, Rewired), and retrieval pipelines.
• Composable Hydra Configuration: Hierarchical YAML experiment configurations enabling declarative evaluation profiles across datasets (LongMemEval, LoCoMo), retrievers, and reader models.
• Deterministic & Auditable Ledger: Every benchmark run is tied to a content-hashed manifest recording exact model parameters, latency, token usage, and per-sample metric breakdowns.
• High-Velocity Tooling & CI: Built with uv for sub-second dependency resolution, accompanied by a comprehensive 74-test verification suite (unit, integration, and statistical tests) with 100% pass rate.
• Empirical Rigor: Implemented paired conversation-cluster bootstrap confidence intervals (10,000 draws) and exact McNemar tests to evaluate statistical significance.

I would love to connect and discuss how my background in building robust, reproducible LLM evaluation pipelines aligns with the technical challenges your team is solving at [Company].

Best regards,

[Your Name]
[Your Portfolio / GitHub / Phone / Email]
```

---

## Template 3: 30-Second Elevator Pitch (LinkedIn InMail / Cold Message)

**Best for:** Short LinkedIn messages, Twitter DMs, or quick email follow-ups.

**Subject:** Quick share: Causal benchmark on Graph RAG vs. Flat Memory for LLMs

```text
Hi [Name],

Quick question: In long-term LLM conversational memory, does knowledge graph structure actually help, or does factual proposition extraction do all the work?

I recently published a causal evaluation harness testing this exact question across LongMemEval and LoCoMo:
👉 https://github.com/AmirBnsL/graph-memory

Key findings across frozen extractors and retrievers:
1. Proposition extraction drives 100% of the gains over raw vectors (+1.8% to +3.5% F1).
2. Knowledge graph traversal (PPR) performs no better than degree-preserving randomized null graphs (p = 0.858).
3. Complex temporal event graphs collapse retrieval (-32% F1) due to bridge bottlenecks in conversational chains.

The codebase is built with uv, composable Hydra configs, and includes 74 passing tests. Would love to hear your thoughts if you're exploring agent memory architectures!

Best,
[Your Name]
```
