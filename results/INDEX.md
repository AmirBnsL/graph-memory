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

- **Retriever is weak and k is unjustified.** `all-MiniLM-L6-v2` at k=10 recovers
  34.6% of gold evidence; the paper sweeps k ∈ {2,5,10,25,50} and its optimum is
  **k=25 (dialog) / k=5 (observation)**, not 10. Sweep, and swap the retriever.
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
