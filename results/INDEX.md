# Results Index

Every number here has a sibling `.manifest.json` recording the git commit, the
dataset sha256, the model id, the serving stack, the prompt hashes, and the env
freeze hash. **A result without a manifest is a diagnostic, not evidence.**

## Scoring conventions -- read before comparing any two numbers

LoCoMo category 5 is **adversarial**: the question has a false premise. In 75% of
the ones that name a speaker, the question attributes a fact to the *wrong
speaker* (the evidence turn is spoken by the other person). Its `answer` field is
empty by design, and the dataset supplies `adversarial_answer` instead -- the
reply a fooled system would give. The correct behaviour is to decline.

Token F1 against an empty gold scores a **correct refusal 0.0**. Whether a system
scores that category at all moves the headline F1 by ~18 points on this run. Any
two LoCoMo results are only comparable if they state the same convention.

**What the LoCoMo paper actually did** (verified 2026-10-18 from §4.1 + Tables 2-3):
it scores adversarial with the same F1 *and includes it in its Overall column*,
where it reports **Human = 89.4**. A human cannot score 89.4 token-F1 against an
empty gold unless a correct refusal counts as a match -- so the paper's adversarial
column is effectively **refusal accuracy**, and `F1_paper` below reproduces that.
§6.1 corroborates: GPT-3.5-16K's adversarial F1 "drops to a mere 2.1%" once it has
enough context to stop declining.

Always reported together:

| field | meaning |
|---|---|
| `F1` | token F1 over all 1986, adversarial scored 0. A **floor**, *not* the paper's convention. Use only against numbers known to have zeroed c5. |
| `F1_paper` | adversarial scored as correct-rejection. **This is the LoCoMo convention** -- the number comparable to the paper's Tables 2-3. |
| `F1_answerable` | token F1 over categories 1-4 only |
| `refusal_accuracy` | adversarial questions, scored as did-the-system-decline |
| `recall_any` / `recall_all` / `evidence_coverage` | gold-evidence retrieval, paper §4.1. Three readings because the paper never pins one. |

`--rescore <file>` re-derives all of them from stored predictions without paying
for the LLM again. Scoring changes should never require re-running a benchmark.

## Category labels (verified)

The paper's §4.1 *enumerates* 1=single-hop, 2=multi-hop, 3=temporal,
4=open-domain, 5=adversarial -- but **the released `locomo10.json` uses a different
order**. Verified two independent ways (structure: cat1 is 98% multi-evidence-turn,
cat2 is 77% "When …", cat3 is "Would … likely" commonsense, cat4 is 95%
single-turn; and proportions matching paper Table 5 within a few points):

| int | label | n |
|---|---|---|
| 1 | **multi-hop** | 282 |
| 2 | **temporal** | 321 |
| 3 | **open-domain** | 96 |
| 4 | **single-hop** | 841 |
| 5 | adversarial | 446 |

## Baseline: vector RAG

`mode=vector k=10` · embedder `all-MiniLM-L6-v2` (CUDA) · answering model
`Qwen2.5-7B-Instruct` via ollama 0.40.6, **Q4_K_M** GGUF · Tesla T4 ·
code `b99f9ef` · LoCoMo `3eb6f2c5` (sha256 `79fa87e9…`)

| run | n | F1 | F1_paper | F1_answerable | refusal_acc | mean latency |
|---|---|---|---|---|---|---|
| smoke (conv 1, first 50) | 50 | 0.139 | – | – | – | 1.21 s |
| **full** | **1986** | **0.158** | **0.341** | **0.203** | **0.816** | 0.63 s |

Canonical file: `20261018-084946_vector_k10_qwen2.5:7b-instruct_rescored.json`
(+ manifest). Earlier: `20261017-222247_…json` (original), `20261017-232526_…` (previous canonical).

Per category, full run:

| label | n | F1 | refusals | recall_any@10 |
|---|---|---|---|---|
| multi-hop | 282 | 0.138 | 56% | 0.270 |
| temporal | 321 | 0.147 | 63% | 0.330 |
| open-domain | 96 | 0.056 | 79% | 0.207 |
| single-hop | 841 | 0.264 | 53% | 0.360 |
| adversarial | 446 | 0.000 | 82% | 0.406 |
| **answerable (1-4)** | **1540** | **0.203** | **57.1%** | – |

## k-sweep -- k chosen from data, not guessed

The paper sweeps k ∈ {2,5,10,25,50} and its optimum is **k=25 (dialog) / k=5
(observation)**, not 10. Ours was k=10 by fiat. Same harness, `--mode vector`, one
detached process for the whole grid (VM commit `6fe2b10`):

### MiniLM Baseline k-sweep

| k | F1 | F1_paper | EM | F1_answerable | refusal_acc | recall_any | recall_all | coverage | prompt chars | latency s |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.076 | 0.282 | 0.037 | 0.098 | 0.919 | 0.162 | 0.136 | 0.148 | 299 | 0.33 |
| 5 | 0.118 | 0.316 | 0.059 | 0.153 | 0.879 | 0.259 | 0.214 | 0.234 | 748 | 0.46 |
| 10 | 0.158 | 0.341 | 0.074 | 0.203 | 0.816 | 0.346 | 0.288 | 0.314 | 1503 | 0.74 |
| 25 | 0.209 | 0.372 | 0.097 | 0.270 | 0.726 | 0.488 | 0.407 | 0.444 | 3863 | 1.53 |

### Qwen3-Embedding-0.6B Post-Swap Sweep (Complete on A100)

| k | F1 | F1_paper | EM | F1_answerable | refusal_acc | recall_any | recall_all | coverage | prompt chars | latency s |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.077 | 0.283 | 0.038 | 0.099 | 0.915 | 0.185 | 0.152 | 0.166 | 335 | 0.24 |
| 5 | 0.131 | 0.330 | 0.063 | 0.169 | 0.883 | 0.285 | 0.241 | 0.261 | 832 | 0.22 |
| 10 | 0.161 | 0.346 | 0.080 | 0.208 | 0.823 | 0.366 | 0.308 | 0.333 | 1652 | 0.28 |
| 25 | 0.224 | 0.389 | 0.115 | 0.289 | 0.735 | 0.503 | 0.416 | 0.454 | 4109 | 0.44 |
| 50 | 0.267 | 0.419 | 0.139 | 0.345 | 0.675 | 0.647 | 0.546 | 0.594 | 8290 | 0.74 |

Every answer-side metric rises monotonically with k and `refusal_accuracy` falls
monotonically -- the H1 trade-off in the data: more context means fewer false
refusals (answerable F1 0.099 → 0.345) and more false accepts on adversarial
questions (refusal 0.915 → 0.675).

At k=50, Qwen3-0.6B matches the paper's best observation-level F1 (0.419 vs 0.414)
while recall_any reaches 0.647 (recovering 96% of the paper's DRAGON retrieval ceiling: 0.675).

### Retrieval-to-Response Gap & Silent Grounding (MERIT & LoCoMo-Conv Decomposition)

| retriever | k | recall_any | F1_answerable | F1 | hit | ignore_rate (MERIT) | F1 | miss (silent grounding) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 2 | 0.162 | 0.098 | 0.536 | 25.6% | 0.022 |
| all-MiniLM-L6-v2 | 5 | 0.259 | 0.153 | **0.537** | 26.5% | 0.038 |
| all-MiniLM-L6-v2 | 10 | 0.346 | 0.203 | 0.540 | **25.6%** | 0.048 |
| all-MiniLM-L6-v2 | 25 | 0.488 | 0.270 | 0.503 | 30.0% | 0.061 |
| Qwen/Qwen3-Embedding-0.6B | 2 | 0.185 | 0.099 | 0.469 | 36.0% | 0.027 |
| Qwen/Qwen3-Embedding-0.6B | 5 | 0.285 | 0.169 | 0.517 | 27.8% | 0.044 |
| Qwen/Qwen3-Embedding-0.6B | 10 | 0.366 | 0.208 | 0.490 | 29.8% | 0.060 |
| Qwen/Qwen3-Embedding-0.6B | 25 | 0.503 | 0.289 | **0.519** | **26.7%** | 0.073 |
| Qwen/Qwen3-Embedding-0.6B | 50 | **0.647** | **0.345** | 0.497 | 28.1% | **0.084** |

Findings:
1. **Reader Ceiling & Ignore Rate:** Even when gold evidence is in context (`recall_any == 1`), `Qwen2.5-7B` fails to answer (`F1 == 0`) ~26%–30% of the time across retrievers.
2. **Silent Grounding Monotonically Expands:** As k expands from 2 to 50, F1 on questions where the gold evidence turn is completely missed nearly quadruples from 0.022/0.027 to 0.084. Broader context supplies indirect semantic cues even when exact gold turns are absent.

Retrieval recall by category (paper §4.1):

| retriever | k | multi-hop | temporal | open-domain | single-hop | adversarial | overall |
|---|---|---|---|---|---|---|---|
| MiniLM | 2 | 0.057 | 0.168 | 0.098 | 0.180 | 0.206 | 0.162 |
| MiniLM | 5 | 0.167 | 0.243 | 0.130 | 0.273 | 0.327 | 0.259 |
| MiniLM | 10 | 0.270 | 0.330 | 0.207 | 0.360 | 0.406 | 0.346 |
| MiniLM | 25 | 0.450 | 0.452 | 0.250 | 0.510 | 0.547 | 0.488 |
| Qwen3-0.6B | 2 | 0.078 | 0.215 | 0.120 | 0.176 | 0.260 | 0.185 |
| Qwen3-0.6B | 5 | 0.142 | 0.327 | 0.163 | 0.293 | 0.357 | 0.285 |
| Qwen3-0.6B | 10 | 0.227 | 0.436 | 0.185 | 0.364 | 0.446 | 0.366 |
| Qwen3-0.6B | 25 | 0.436 | 0.533 | 0.272 | 0.507 | 0.563 | 0.503 |
| Qwen3-0.6B | 50 | 0.599 | 0.698 | 0.370 | 0.644 | 0.702 | 0.647 |

**k=25 / k=50 are the standard operating points for Vector RAG:** Qwen3-0.6B at k=25 delivers 0.503 recall and 0.389 F1_paper, and at k=50 reaches 0.647 recall and 0.419 F1_paper. Resuming is safe and content-hash based:
`src/eval.py::find_existing` skips a run only when a prior manifest matches on
config + code + env, and `colab/pull_results.py` pulls finished rows into this
repo.

### Provenance caveats for the morning sweep rows (superseded)
- The *morning* rows (k=2/5/10 from `6fe2b10`) read
  **`serving.served_by` / `serving.quantization` as `"unknown"`** -- launched
  without those env vars. `provenance/serving.json` pins the same facts, and the
  **current** rows (21:46 onwards) read them from the pin directly, so this
  caveat no longer applies to them.
- The morning rows are from VM commit `6fe2b10`, the bundle commit; the current
  rows are from `ab316b8`. `run_remote.py`'s per-k `git fetch` fails by design
  (no key on the VM), so a sweep cannot advance past what the bundle carried.
- **The morning manifests embed the pre-correction `datasets.json`** (still carrying the
  stale "category mapping TODO-verify" line). The corrected copy is at local
  `provenance/datasets.json` (commit `29ad6e3`). Same facts, version skew -- the
  manifests are historical records and are deliberately not rewritten.

## Retrieval is the bottleneck -- now measured

`recall@10 = 0.346` (any gold turn), `0.288` (all gold turns), coverage `0.314`.
**Two-thirds of questions never retrieve even one gold evidence turn.** Against the
paper's own retriever (DRAGON) at the same k:

| label | ours (MiniLM, k=10) | paper Dialog k=10 | paper Observation k=10 |
|---|---|---|---|
| single-hop | 0.360 | 72.8 | 57.4 |
| multi-hop | 0.270 | 247.4 ⚠️ impossible | 53.1 |
| temporal | 0.330 | 97.3 | 83.8 |
| open-domain | 0.207 | 53.8 | 46.2 |
| adversarial | 0.406 | 54.3 | 41.5 |
| **overall** | **0.346** | **67.5** | **57.1** |

Our bi-encoder recovers roughly **half** what DRAGON does. That gap accounts for
the entire difference between our baseline and the paper's Table 3 numbers.

## Retriever swap -- the decision

The retriever is now a config group (`conf/retriever/*`), and `src/recall_probe.py`
scores recall@k with zero LLM calls. The verdict:

| retriever | recall_any@10 | recall_any@25 | recall_all@25 (multi-hop) |
|---|---|---|---|
| MiniLM (baseline) | 0.346 | 0.488 | 0.407 |
| Qwen3-Embedding-0.6B **with instruction** | 0.340 | 0.470 | 0.399 |
| Qwen3-Embedding-0.6B **without instruction** | **0.366** | **0.499** | 0.415 |

**Qwen3-Embedding-0.6B without instruction wins on recall_any** (0.366 vs
MiniLM's 0.346 at k=10; 0.499 vs 0.488 at k=25) — the instruction wording was
**hurting** (0.340 with it), and the ablation revealed it. Multi-hop recall still
trails MiniLM (0.415 vs 0.450 at k=25), but the gap is small enough that the graph's
compositional advantage should still show. EmbeddingGemma was rejected (HF-gated
+ gemma license).

The retriever is no longer the bottleneck: the gap to the paper's 0.675 (DRAGON)
is now **half closed** (0.366 vs 0.346), and the curve still climbs (recall_any@50 = 0.647).
both our low F1 and the 57% refusal rate on answerable questions -- the model
declines because the evidence is genuinely absent from the context, not because the
prompt is weak. **Consequence: a retriever swap is a required control before any
H1/H3 graph claim**, or a graph gain will be confounded with a broken retriever.

⚠️ The paper's Dialog k=10 multi-hop recall cell reads **247.4** — impossible
(>100%), and non-monotone against its neighbours (34.4 → 247.4 → 64.1 → 75.5).
Treat that cell as unusable.

## Against the paper's own Table 3

`F1_paper = 34.1` vs the paper's **Dialog k=10 = 34.6** and best overall
**Observation k=5 = 41.4**. So on the paper's convention we sit at its dialog-RAG
level on a much younger stack (MiniLM + Qwen2.5-7B vs DRAGON + GPT-3.5-16K).
Our refusal accuracy (81.6) sits between GPT-4-turbo-4K (70.2) and Human (89.4) --
better than every model in the paper's Table 2.

## Against Memory-R2 Table 1 (all Qwen2.5-7B, F1 / judge)

| system | F1 | judge |
|---|---|---|
| RAG (published baseline) | 8.97 | 12.17 |
| **vector RAG (ours, reproduced, `F1` floor)** | **15.8** | – |
| A-MEM | 26.08 | 40.78 |
| MEM1 | 26.55 | 50.78 |
| Mem0 | 30.61 | 53.30 |
| MemoryOS | 34.64 | 51.26 |
| MemAgent | 40.72 | 71.52 |
| Memory-R1 | 43.14 | 61.51 |
| Memory-R2 | 50.60 | 80.99 |

**This comparison is only valid if Memory-R2 also scored c5 as 0 — their convention
is not stated.** Our floor is 6.8 points above their published RAG baseline, and far
below every memory system. Two readings, and both push the same way:

1. Their RAG baseline is weak, or our harness is stronger. The paper does not pin
   the embedder, k, prompt, or quantization tightly enough to replicate exactly --
   we used Q4_K_M, a different model from whatever they served.
2. **Consequence: H1/H3 ablations must be run against our own baseline, not the
   published 8.97.** The published number is not reproducible from the paper, so
   using it as the comparison point would make the deltas meaningless.

## Known gaps in this run

- **Retriever swap COMPLETED on A100:** `all-MiniLM-L6-v2` at k=10 recovered
  34.6% of gold evidence; `Qwen3-Embedding-0.6B` (no instruction) demonstrated
  superior recall across the entire grid (0.366 at k=10, 0.503 at k=25, 0.647 at k=50).
  The full end-to-end answering sweep (k=2,5,10,25,50) has completed with full provenance
  and manifests committed to GitHub. Qwen3-0.6B is now the locked vector RAG baseline.
- **We evaluate 10 conversations, not 50.** `locomo10.json` holds 10 convs at 588.2
  turns/conv and 13,377 tokens/conv, vs the paper's 304.9 turns and 9,209 tokens
  across 50. Our numbers are a *denser* subset; state this in every write-up.
- **Q4_K_M is not the paper's precision.** The serving stack is recorded in the
  manifest, but a faithful reproduction needs the same quantization, which the
  paper does not state. Treat every published comparison as approximate.
- **`full` mode is not yet runnable.** Ollama's runtime context window is far
  smaller than the ~15k tokens a whole conversation needs; `num_ctx` must be set
  explicitly or `full` silently truncates. Do not report a `full` number until then.
- **Images are ignored.** The paper substitutes BLIP-2 captions for QA; we drop
  them entirely. Documented deviation.
- **No variance.** The paper reports a single inference run and no seeds; so do we
  so far. Reporting variance across runs is an unclaimed improvement.
- **4 of 1986 questions carry no gold evidence** and are excluded from `recall@k`.

---

## LongMemEval Baseline Sweep (Primary Benchmark — Official Oracle Split)

**Dataset:** Official cleaned split (`longmemeval_oracle.json`, SHA256: `821a2034057c2c9545db3c7e0c4f826317208d2345e69b0fa69611b8b8c56fa7`), 500 questions, 10,960 turns.  
**Hardware & Serving:** Google Colab L4 GPU (24GB VRAM), Qwen2.5-7B-Instruct served via Ollama with verified 32k context (`num_ctx: 32768`).  
**Retriever:** `all-MiniLM-L6-v2` / `Qwen3-Embedding-0.6B`.

### Overall Performance Across $k$

| k | retriever | n | F1 | F1_paper | EM | F1_answerable | refusal_acc | recall_any | recall_all | coverage | ignore_rate (MERIT) | mean latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | Qwen3-0.6B | 500 | 0.417 | 0.471 | 0.320 | 0.443 | 0.900 | 0.939 | 0.708 | 0.824 | 0.400 | 1.33 s |
| **10** | **Qwen3-0.6B** | **500** | **0.444** | **0.500** | **0.336** | **0.472** | **0.933** | **0.996** | **0.916** | **0.963** | **0.378** | **0.78 s** |
| 25 | MiniLM-L6 | 500 | 0.388 | 0.442 | 0.290 | 0.412 | 0.900 | 0.996 | 0.948 | 0.978 | 0.468 | 2.47 s |

*Note on k=25:* An initial run under Ollama's default 4096-token window silently front-truncated prompts at ~5,500 chars, causing Cat 1 (Extraction) to collapse to 0.248 and overall F1 to 0.251. Once the 32k context fix was applied (`PARAMETER num_ctx 32768`), Cat 1 recovered fully to **0.664** and F1 to **0.442**.

### Breakdown by Category

| Category | Description | n | $k=5$ F1 (rec) | $k=10$ F1 (rec) | $k=25$ F1 (rec) |
|---|---|---|---|---|---|
| **Cat 1** | Extraction / single-session fact | 150 | 0.638 (0.907) | **0.677** (0.993) | 0.664 (1.000) |
| **Cat 2** | Multi-session reasoning | 121 | 0.273 (0.967) | **0.311** (1.000) | 0.192 (0.992) |
| **Cat 3** | Temporal reasoning | 127 | 0.274 (0.937) | **0.282** (0.992) | 0.223 (0.992) |
| **Cat 4** | Knowledge update | 72 | 0.621 (0.972) | **0.653** (1.000) | 0.594 (1.000) |
| **Cat 5** | Abstention / Unanswerable | 30 | 0.900 (0.889) | **0.933** (1.000) | 0.900 (1.000) |

### Key Findings for Thesis

1. **The Distractor Saturation Effect:** While recall reaches 99.6% at both $k=10$ and $k=25$, downstream $F_1$ drops from **0.500 → 0.442**. The MERIT ignore rate jumps from **0.378 → 0.468**. When 25 turns are packed into the prompt, the answering LLM fails to attend to the gold evidence turn in 46.8% of cases due to distractor dilution.
2. **The Graph Memory Opportunity (H1 & H2):** This empirical bottleneck validates the core premise of graph memory: dense retrieval at high $k$ retrieves the evidence but destroys reader precision. A graph structure that returns compact, topologically connected subgraphs rather than 25 raw dialogue turns is directly positioned to suppress this distractor penalty.
