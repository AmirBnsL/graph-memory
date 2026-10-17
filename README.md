# graph-memory

Code for a Master's PFE on **graph-based memory for LLM agents**.

Two hypotheses, one harness:

- **H1 — structure vs. extraction.** Does the *graph* contribute, or is it the LLM's
  *note extraction*? Tested by controlled degradation (structure-flattened,
  edge-randomized) and extractor swaps, with a fixed multi-judge protocol.
- **H3 — structural quality metrics.** Do cheap graph statistics (connectivity, hub
  concentration, redundancy, temporal-validity coverage) **predict** downstream utility,
  and does **structurally repairing** a degraded graph **cause** a gain over a matched
  do-nothing control?

H2 (a small GNN reranker) is a stretch goal, judged on the **accuracy–cost Pareto
frontier**, not on accuracy alone.

The research record lives in the Obsidian vault (`graph-memory-agents`).
This repo is the code.

## Layout

| Path | Purpose |
|---|---|
| `data/` | benchmark data (git-cloned, not committed) |
| `src/load.py` | loaders + normalization (LoCoMo now, LongMemEval next) |
| `src/llm.py` | one LLM client for any provider + on-disk response cache |
| `src/answer.py` | retrieval modes + answering *(next)* |
| `src/eval.py` | scoring (EM + judge) + reporting *(next)* |
| `graphs/` | **cached memory graphs — the artifact H1 breaks and H3 measures** |
| `cache/` | every LLM call, saved once, reused forever |
| `results/` | results tables (accuracy + cost + latency, always together) |

## Provenance

- **LoCoMo**: `github.com/snap-research/locomo`
  commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` (2024-08-12).
  Data file: `data/locomo/data/locomo10.json`.
  ⚠️ The HuggingFace dataset `adymaharana/locomo` is **empty** — do not use `load_dataset()`.
  ⚠️ An independent audit found 6.4% of LoCoMo gold answers score-corrupting and a
  popular judge config accepting 62.8% of deliberately wrong answers → we report
  F1/EM alongside any judge score, and never a single judge.
- **LongMemEval**: pending official release (its HF/page distribution, not a re-upload).
  ⚠️ Community reports ground-truth arithmetic errors and mislabeled evidence — verify before trusting.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install datasets openai pandas networkx
```

## Status

- [x] Project scaffolded; LoCoMo cloned and committed by hash
- [x] `src/load.py` — 10 conversations, 5,882 turns, 1,986 QA (categories 1–5)
- [x] `src/llm.py` — provider-agnostic client + disk cache
- [ ] **BLOCKED: no answering model configured** (no local model, no GPU, no API key yet)
- [ ] `src/answer.py` + `src/eval.py`
- [ ] Reproduce one published number within ±2 points → **Week 1 exit gate**
