#!/usr/bin/env bash
#
# Make a freshly-created Colab VM actually runnable.
#
# Runs ON the VM (unlike bootstrap.sh, which orchestrates from the laptop over
# SSH). Send it with:
#
#     colab exec -s gm-dev --timeout 1800 -f colab/vm_setup.sh
#
# Idempotent: safe to re-run on every new session. Answers the three things a
# cold VM always lacks:
#   1. the repo checkout itself (data/ is gitignored, so a clone never brings it)
#   2. LoCoMo, restored at the pinned commit and hash-checked
#   3. an OpenAI-compatible server for the frozen answering model
#
# NEVER install the local requirements.txt here -- it pins torch==+cpu and would
# clobber the CUDA build with a CPU-only wheel.
#
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/graph-memory}"
REPO_SSH="${REPO_SSH:-git@github.com:AmirBnsL/graph-memory.git}"
LOCOMO_REPO="${LOCOMO_REPO:-https://github.com/snap-research/locomo}"
LOCOMO_PIN="${LOCOMO_PIN:-3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376}"
LOCOMO_SHA="${LOCOMO_SHA:-79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4}"
MODEL="${MODEL:-qwen2.5:7b-instruct}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. repo ---------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
  say "cloning $REPO_SSH"
  git clone --quiet "$REPO_SSH" "$REPO_DIR"
fi
git config --global --add safe.directory "$REPO_DIR" 2>/dev/null || true
say "repo at $(git -C "$REPO_DIR" rev-parse --short HEAD) on $(git -C "$REPO_DIR" branch --show-current)"

# 2. dataset ------------------------------------------------------------------
LOCOMO_DIR="$REPO_DIR/data/locomo"
LOCOMO_FILE="$LOCOMO_DIR/data/locomo10.json"
if [ ! -f "$LOCOMO_FILE" ]; then
  say "fetching LoCoMo at $LOCOMO_PIN"
  mkdir -p "$REPO_DIR/data"
  [ -d "$LOCOMO_DIR/.git" ] || git clone --quiet "$LOCOMO_REPO" "$LOCOMO_DIR"
  git -C "$LOCOMO_DIR" checkout --quiet "$LOCOMO_PIN"
fi
GOT_SHA="$(sha256sum "$LOCOMO_FILE" | cut -d' ' -f1)"
if [ "$GOT_SHA" != "$LOCOMO_SHA" ]; then
  die "LoCoMo hash mismatch
     expected $LOCOMO_SHA
     got      $GOT_SHA
     refusing to run against an unpinned dataset"
fi
say "LoCoMo pinned OK (sha256 ${GOT_SHA:0:16}...)"

# 2.2 LongMemEval -------------------------------------------------------------
LONGMEMEVAL_FILE="$REPO_DIR/data/longmemeval/longmemeval_oracle.json"
if [ ! -f "$LONGMEMEVAL_FILE" ]; then
  say "fetching LongMemEval oracle dataset"
  python3 "$REPO_DIR/src/download_longmemeval.py" --target oracle
fi
say "LongMemEval dataset ready"

# 3. extras the stock image lacks ---------------------------------------------
# hydra-core: the config layer (conf/). Not in requirements.txt's install path
# here because that file pins torch==+cpu and must never be installed on the VM.
say "installing cloud delta: bitsandbytes + hydra (config layer)"
uv pip install --system --quiet bitsandbytes hydra-core omegaconf

# 4. model server -------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  # The stock Colab image ships no zstd, and ollama's installer hard-fails
  # without it. This is the single most common cold-VM failure here.
  if ! command -v zstd >/dev/null 2>&1; then
    say "installing zstd (required by the ollama installer)"
    apt-get update -qq
    apt-get install -y -qq zstd
  fi
  say "installing ollama"
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! pgrep -x ollama >/dev/null 2>&1; then
  say "starting ollama"
  nohup ollama serve >/content/ollama-serve.log 2>&1 &
  sleep 8
fi
if ! ollama list 2>/dev/null | grep -q "${MODEL%%:*}"; then
  say "pulling $MODEL (several GB, first time only)"
  ollama pull "$MODEL"
fi
say "model ready:"
ollama list

# 5. prove the harness can reach the model ------------------------------------
say "harness healthcheck"
cd "$REPO_DIR"
LLM_PROVIDER=ollama LLM_MODEL="$MODEL" python3 src/llm.py

say "done."
echo
echo "  run a smoke test:"
echo "    colab exec -s gm-dev --timeout 0 \\"
echo "      --env EVAL_ARGS=\"limits.n_conv=1 limits.n_qa=50\" \\"
echo "      -f colab/run_remote.py"
echo
echo "  sweep:  colab exec -s gm-dev --env DETACH=1 --env SWEEP=2,5,10,25,50 \\"
echo "            --env RETRIEVER=qwen3-06b --env EXTRA_ARGS=\"retriever.query_prefix=\" \\"
echo "            -f colab/run_sweep.py"
echo "          keep the VM alive while it runs (free tier reclaims an unattended"
echo "          VM in ~90 min):  colab/keepalive.sh gm-dev"
echo
echo "  stop the meter:  colab stop -s gm-dev"
