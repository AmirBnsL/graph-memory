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

Token F1 against an empty gold therefore scores a **correct refusal 0.0**. Whether
a paper scores that category at all moves the headline F1 by ~18 points on this
run. Any two LoCoMo results are only comparable if they state the same convention.

Always reported together:

| field | meaning |
|---|---|
| `F1` | token F1 over all 1986, adversarial scored 0. **The published convention** -- this is what cross-paper comparison must use. |
| `F1_answerable` | token F1 over categories 1-4 only |
| `refusal_accuracy` | adversarial questions, scored as did-the-system-decline |

`--rescore <file>` re-derives all three from stored predictions without paying
for the LLM again. Scoring changes should never require re-running a benchmark.

## Baseline: vector RAG

`mode=vector k=10` · embedder `all-MiniLM-L6-v2` (CUDA) · answering model
`Qwen2.5-7B-Instruct` via ollama 0.40.6, **Q4_K_M** GGUF · Tesla T4 ·
code `b99f9ef` · LoCoMo `3eb6f2c5` (sha256 `79fa87e9…`)

| run | n | F1 | F1_answerable | refusal_accuracy | mean latency |
|---|---|---|---|---|---|
| smoke (conv 1, first 50) | 50 | 0.139 | – | – | 1.21 s |
| **full** | **1986** | **0.158** | **0.203** | **0.816** | 0.63 s |

Per category (full run): c1 0.138 · c2 0.147 · c3 0.056 · c4 0.264 · c5 0.000
(c5 is 0 by convention -- see `refusal_accuracy` for what it actually scores).

Files: `20261017-222247_…json` (original scoring), `20261017-232526_…_rescored.json`
(**canonical**), plus manifests. The 50-QA smoke run was superseded and its
manifest carried the pre-fix dirty flag.

## Against Memory-R2 Table 1 (all Qwen2.5-7B, F1 / judge)

| system | F1 | judge |
|---|---|---|
| **vector RAG (ours, reproduced)** | **15.8** | – |
| RAG (published baseline) | 8.97 | 12.17 |
| A-MEM | 26.08 | 40.78 |
| MEM1 | 26.55 | 50.78 |
| Mem0 | 30.61 | 53.30 |
| MemoryOS | 34.64 | 51.26 |
| MemAgent | 40.72 | 71.52 |
| Memory-R1 | 43.14 | 61.51 |
| Memory-R2 | 50.60 | 80.99 |

**Our reproduced RAG baseline is 6.8 points above the published RAG baseline**, and
far below every memory system. Two readings, and both push the same way:

1. Their RAG baseline is weak, or our harness is stronger. The paper does not pin
   the embedder, k, prompt, or quantization tightly enough to replicate exactly --
   we used Q4_K_M, which is a different model from whatever they served.
2. **Consequence: H1/H3 ablations must be run against our own baseline, not the
   published 8.97.** The published number is not reproducible from the paper, so
   using it as the comparison point would make the deltas meaningless.

## Retrieval is the bottleneck, not the model

On answerable questions (n=1540), **57.1% of predictions are refusals**:

| category | n | refusals | F1 |
|---|---|---|---|
| 1 | 282 | 56% | 0.138 |
| 2 | 321 | 63% | 0.147 |
| 3 (open-domain) | 96 | **79%** | 0.056 |
| 4 (multi-hop) | 841 | 53% | 0.264 |

The model declines because the evidence is not among the top-10 turns. This is
precisely the gap H1/H3 target: flat top-k cannot compose evidence scattered
across a ~600-turn conversation, and no prompt change fixes it.

## Known gaps in this run

- **Q4_K_M is not the paper's precision.** The serving stack is recorded in the
  manifest, but a faithful reproduction needs the same quantization, which the
  paper does not state. Treat every published comparison as approximate.
- **`full` mode is not yet runnable.** Ollama's runtime context window is far
  smaller than the ~15k tokens a whole conversation needs; `num_ctx` must be set
  explicitly or `full` silently truncates. Do not report a `full` number until then.
- Category semantics (which integer means multi-hop / temporal / open-domain) are
  inferred from the data, not yet verified against the LoCoMo paper.
