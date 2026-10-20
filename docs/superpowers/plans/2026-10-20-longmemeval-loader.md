# LongMemEval Benchmark Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the LongMemEval benchmark (ICLR 2025, 500 questions across 5 core memory abilities) into the graph-memory evaluation harness with pinned dataset provenance, uniform `Conversation`/`Turn`/`QA` data structures, and Hydra configuration, enabling vector RAG baseline evaluation.

**Architecture:** Download and verify the official `longmemeval-cleaned` release, pinning SHA256 hashes in `provenance/datasets.json`. Implement `load_longmemeval()` in `src/load.py` normalizing multi-session chat histories and target questions into existing harness dataclasses, with Category 5 mapped to Abstention (matching LoCoMo's Category 5 convention). Connect the loader to Hydra via `conf/benchmark/longmemeval.yaml` and verify end-to-end execution on the Colab GPU VM before committing results.

**Tech Stack:** Python 3.10+, PyTorch, Hydra, dataclasses, pytest, urllib / requests, Ollama (`qwen2.5:7b-instruct`), HuggingFace sentence-transformers (`Qwen3-Embedding-0.6B`).

**Spec:** [`03-benchmarks/LongMemEval (2410.10813).md`](file:///home/amirbnsl/Documents/Obsidian%20Vault/graph-memory-agents/03-benchmarks/LongMemEval%20(2410.10813).md), [`06-experiment-design/Experiment Plan.md`](file:///home/amirbnsl/Documents/Obsidian%20Vault/graph-memory-agents/06-experiment-design/Experiment%20Plan.md), and official repository [`xiaowu0162/LongMemEval`](https://github.com/xiaowu0162/LongMemEval).

## Global Constraints

- **Reader Model Frozen:** Answering reader must remain strictly `qwen2.5:7b-instruct` (Q4_K_M) served via Ollama.
- **Retriever Model Frozen:** Retriever must remain strictly `Qwen/Qwen3-Embedding-0.6B` without query instruction prefix.
- **Category 5 Invariant:** Category 5 is reserved for Abstention/Refusal questions (`_abs`), aligning with LoCoMo Category 5 adversarial convention where correct refusal is scored under `refusal_accuracy` and `F1_paper`.
- **Zero LoCoMo Regression:** All existing LoCoMo loader functionality, tests, and pinned hashes must remain completely unchanged.
- **Git Push Gate:** Code changes to `origin/master` must only occur after an in-VM smoke test passes with `DRIVER_EXIT=0`.

## Review Focus

1. **Abstention Evidence:** In `_abs` instances, `answer` is an explanation of missing info; ensure `has_answer` turns or empty evidence do not cause division-by-zero in `recall@k`.
2. **Turn ID Collision:** Across 500 questions and ~40 sessions per question, ensure turn `dia_id` values are globally unique per conversation.
3. **Session Timestamp Parsing:** LongMemEval dates use format `YYYY/MM/DD (Day) HH:MM`; ensure session timestamps are preserved for temporal reasoning without parsing exceptions.
4. **Missing Evidence Keys:** Ensure instances with no gold answer turns gracefully assign `evidence=[]` without raising `KeyError`.
5. **Memory Footprint during Load:** `longmemeval_s_cleaned.json` is ~277 MB; loader must load and parse in <5 seconds without excessive RAM bloat.

---

### Task 1: Dataset Provenance & Ingestion Tooling

**Files:**
- Create: `src/download_longmemeval.py`
- Modify: `provenance/datasets.json`
- Test: `tests/test_longmemeval_provenance.py`

**Interfaces:**
- Produces: `data/longmemeval/longmemeval_oracle.json` and `data/longmemeval/longmemeval_s_cleaned.json`
- Pinning: SHA256 hashes recorded in `provenance/datasets.json` under `"longmemeval"`

- [ ] **Step 1: Write the failing provenance test**

```python
# tests/test_longmemeval_provenance.py
import json
import hashlib
from pathlib import Path

def test_longmemeval_dataset_pinned():
    pin_file = Path("provenance/datasets.json")
    assert pin_file.exists()
    pins = json.loads(pin_file.read_text())
    assert "longmemeval" in pins
    cfg = pins["longmemeval"]
    assert cfg["status"] == "verified"
    assert cfg["sha256"] is not None
    assert len(cfg["sha256"]) == 64
    dpath = Path(cfg["local_path"])
    assert dpath.exists(), f"Dataset file {dpath} missing"
    hasher = hashlib.sha256()
    with open(dpath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    assert hasher.hexdigest() == cfg["sha256"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_longmemeval_provenance.py -v`
Expected: FAIL with `AssertionError: assert 'pending' == 'verified'`

- [ ] **Step 3: Implement download script and update `provenance/datasets.json`**

Create `src/download_longmemeval.py` to stream download `longmemeval_oracle.json` and `longmemeval_s_cleaned.json` from the official HuggingFace release (`https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/`), compute their SHA256 hashes, and update `provenance/datasets.json` with exact byte counts and checksums.

- [ ] **Step 4: Execute download script and verify test passes**

Run: `python3 src/download_longmemeval.py --target oracle`
Run: `pytest tests/test_longmemeval_provenance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add provenance/datasets.json src/download_longmemeval.py tests/test_longmemeval_provenance.py
git commit -m "feat(provenance): pin official LongMemEval dataset hashes"
```

---

### Task 2: LongMemEval Loader Normalization in `src/load.py`

**Files:**
- Modify: `src/load.py:163-174`
- Test: `tests/test_load_longmemeval.py`

**Interfaces:**
- Consumes: `data/longmemeval/longmemeval_oracle.json` (or `longmemeval_s_cleaned.json`)
- Produces: `load_longmemeval(limit: int | None = None, split: str = "oracle", path: Path | None = None) -> list[Conversation]`
- Category mapping:
  - 1: `extraction` (single-session-user, assistant, preference)
  - 2: `multi-session` (multi-session reasoning)
  - 3: `temporal` (temporal reasoning)
  - 4: `update` (knowledge updates)
  - 5: `abstention` (questions with `_abs` suffix)

- [ ] **Step 1: Write the failing loader test**

```python
# tests/test_load_longmemeval.py
from pathlib import Path
from src.load import load_longmemeval, Conversation, Turn, QA

def test_load_longmemeval_oracle():
    convs = load_longmemeval(limit=5, split="oracle")
    assert len(convs) == 5
    for c in convs:
        assert isinstance(c, Conversation)
        assert len(c.turns) > 0
        assert len(c.qa) == 1
        qa = c.qa[0]
        assert isinstance(qa, QA)
        assert qa.question
        assert qa.category in {1, 2, 3, 4, 5}
        if c.sample_id.endswith("_abs"):
            assert qa.category == 5
        for t in c.turns:
            assert isinstance(t, Turn)
            assert t.dia_id
            assert t.speaker in {"user", "assistant"}

def test_load_longmemeval_categories():
    convs = load_longmemeval(split="oracle")
    assert len(convs) == 500
    from collections import Counter
    cats = Counter(c.qa[0].category for c in convs)
    # 5 abilities: extraction (150), multi-session (121), temporal (127), update (72), abstention (30)
    assert cats[5] == 30
    assert sum(cats.values()) == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load_longmemeval.py -v`
Expected: FAIL with `NotImplementedError: LongMemEval loader pending`

- [ ] **Step 3: Implement `load_longmemeval` in `src/load.py`**

Replace the stub in `src/load.py` with the full parser:
- Handle `split="oracle"` vs `split="s"`.
- Normalize each question instance into a `Conversation`:
  - `sample_id = d["question_id"]`
  - Build `turns` from `haystack_sessions`:
    `dia_id = f"{d['question_id']}_s{s_idx}_t{t_idx}"`
    `speaker = turn["role"]`
    `text = turn["content"]`
    `session_id = s_idx`
    `date_time = d["haystack_dates"][s_idx]`
    If `turn.get("has_answer")`, append `dia_id` to `evidence`.
  - Determine category:
    If `d["question_id"].endswith("_abs")`: category = 5 (Abstention)
    elif `d["question_type"] == "temporal-reasoning"`: category = 3
    elif `d["question_type"] == "knowledge-update"`: category = 4
    elif `d["question_type"] == "multi-session"`: category = 2
    else: category = 1 (single-session extraction)
  - Create `QA(question=d["question"], answer=str(d["answer"]), evidence=evidence, category=cat)`
- Add `LONGMEMEVAL_CATEGORIES` mapping dict in `src/load.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_load_longmemeval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/load.py tests/test_load_longmemeval.py
git commit -m "feat(load): implement LongMemEval parser and ability normalization"
```

---

### Task 3: Hydra Benchmark Configuration & Evaluator Wiring

**Files:**
- Modify: `conf/benchmark/longmemeval.yaml`
- Modify: `conf/reader/ollama.yaml`
- Modify: `src/eval.py`
- Test: `tests/test_eval_longmemeval.py`

**Interfaces:**
- Consumes: `load_longmemeval`
- Produces: `python src/eval.py benchmark=longmemeval limits.n_qa=5`

- [ ] **Step 1: Write failing pipeline test**

```python
# tests/test_eval_longmemeval.py
from src.eval import run

def test_eval_longmemeval_dry_run():
    # Run 2 questions with vector retriever top-2 on mock reader
    summary = run(
        mode="vector",
        k=2,
        n_conv=2,
        n_qa=2,
        categories=None,
        retriever="qwen3-06b",
        benchmark="longmemeval"
    )
    assert summary["benchmark"] == "longmemeval"
    assert summary["n"] == 2
    assert "F1_paper" in summary
    assert "recall_any" in summary
    assert len(summary["rows"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_longmemeval.py -v`
Expected: FAIL or error on category resolution / configuration

- [ ] **Step 3: Update `conf/benchmark/longmemeval.yaml`, `conf/reader/ollama.yaml`, and `src/eval.py`**

- In `conf/benchmark/longmemeval.yaml`:
  ```yaml
  name: longmemeval
  local_path: data/longmemeval/longmemeval_oracle.json
  split: oracle
  categories: null
  ```
- In `conf/reader/ollama.yaml`:
  Ensure `num_ctx: 8192` is explicitly parameterized.
- In `src/eval.py`:
  Ensure category names look up `LONGMEMEVAL_CATEGORIES` when `benchmark == "longmemeval"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_longmemeval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add conf/benchmark/longmemeval.yaml conf/reader/ollama.yaml src/eval.py tests/test_eval_longmemeval.py
git commit -m "feat(eval): wire LongMemEval through Hydra harness with 8k reader context"
```

---

### Task 4: In-VM Smoke Verification & Git Push Gate

**Files:**
- Modify: `colab/run_setup.py` (add LongMemEval download/setup step if missing)
- Remote execution: `colab exec -s gm-dev ...`

**Interfaces:**
- Pre-condition: Git commits on local `master`
- Gate: `colab exec -s gm-dev --env EVAL_ARGS="benchmark=longmemeval limits.n_qa=5" -f colab/run_remote.py` exits 0
- Output: `git push origin master` executed ONLY upon verification

- [ ] **Step 1: Update `colab/run_setup.py` to ensure dataset availability on VM**

Ensure `colab/run_setup.py` downloads or unbundles `data/longmemeval/` and verifies the SHA256 pin before execution.

- [ ] **Step 2: Create bundle and provision Colab L4 session**

```bash
git bundle create /tmp/gm.bundle master
colab new -s gm-dev --gpu L4
colab upload -s gm-dev /tmp/gm.bundle /content/gm.bundle
colab exec -s gm-dev --timeout 1800 -f colab/run_setup.py
```

- [ ] **Step 3: Run in-VM 5-question smoke test**

```bash
colab exec -s gm-dev --timeout 600 \
  --env EVAL_ARGS="benchmark=longmemeval retriever=qwen3-06b retriever.query_prefix= limits.n_qa=5" \
  -f colab/run_remote.py
```
Expected: `DRIVER_EXIT=0` in log, outputs 5 scored answers and manifest.

- [ ] **Step 4: Clean smoke results and execute Git Push Gate**

Remove smoke run `.json` from VM `/content/graph-memory/results/`, then push verified code to GitHub:
```bash
git push origin master
```

---

### Task 5: Baseline Sweep & Documentation Ledger

**Files:**
- Modify: `results/INDEX.md`
- Output: `results/*_longmemeval_*.json` + manifests

**Interfaces:**
- Executes: `colab/run_sweep.py` with `SWEEP=5,10,25 benchmark=longmemeval`
- Produces: Baseline results table for LongMemEval

- [ ] **Step 1: Launch detached sweep on Colab L4**

```bash
colab exec -s gm-dev --env DETACH=1 --env SWEEP=5,10,25 \
  --env RETRIEVER=qwen3-06b --env EXTRA_ARGS="retriever.query_prefix= benchmark=longmemeval" \
  -f colab/run_sweep.py
```

- [ ] **Step 2: Run keepalive to auto-checkpoint results to GitHub**

```bash
INTERVAL=180 ./colab/keepalive.sh gm-dev
```

- [ ] **Step 3: Update `results/INDEX.md` with LongMemEval baseline numbers**

Run: `python3 src/make_tables.py`
Add LongMemEval baseline section to `results/INDEX.md` documenting per-ability scores:
- Information extraction
- Multi-session reasoning
- Temporal reasoning
- Knowledge updates
- Abstention (Category 5)

- [ ] **Step 4: Commit ledger and stop Colab session**

```bash
git commit -am "docs(results): record initial LongMemEval vector RAG baseline"
git push origin master
colab stop -s gm-dev
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Official LongMemEval ICLR 2025 release used, 5 ability splits mapped, SHA256 pinned.
- [x] **Step scan:** Every step contains explicit tests, exact file targets, and verifiable exit criteria.
- [x] **Type consistency:** Matches existing `Conversation`, `Turn`, and `QA` dataclasses in `src/load.py`.
- [x] **Review Focus addressed:** Category 5 maps to abstention (`_abs`), preserving refusal accuracy and non-zero division safety.
- [x] **Push discipline:** Push to GitHub occurs strictly after in-VM smoke verification passes.
