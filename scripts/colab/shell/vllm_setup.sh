#!/usr/bin/env bash
#
# Provision Google Colab Pro NVIDIA L4 (24GB) VM with vLLM serving.
#
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/graph-memory}"
DRIVE_STORAGE="/content/drive/MyDrive/graph-memory-storage"
MODEL_TIER="${MODEL_TIER:-7b}"  # 7b | 14b
PORT="${PORT:-8000}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

# 1. Persistent Drive links for HuggingFace & cache
if [ -d "/content/drive/MyDrive" ]; then
    say "Linking HuggingFace cache to persistent Google Drive: $DRIVE_STORAGE/huggingface"
    mkdir -p "$DRIVE_STORAGE/huggingface"
    mkdir -p /root/.cache
    ln -sfn "$DRIVE_STORAGE/huggingface" /root/.cache/huggingface
    export HF_HOME="$DRIVE_STORAGE/huggingface"
    echo "export HF_HOME=\"$DRIVE_STORAGE/huggingface\"" >> /root/.bashrc
fi

# 2. Install vLLM + dependencies via uv
say "Installing vLLM, litellm, and upstream dependencies"
uv pip install --system --quiet vllm litellm igraph faiss-cpu rank-bm25

# 3. Model selection
if [ "$MODEL_TIER" = "14b" ]; then
    SERVED_MODEL="Qwen/Qwen2.5-14B-Instruct-FP8"
    MODEL_ALIAS="qwen2.5:14b-instruct"
else
    SERVED_MODEL="Qwen/Qwen2.5-7B-Instruct"
    MODEL_ALIAS="qwen2.5:7b-instruct"
fi

# 4. Launch vLLM server in background
if pgrep -f "vllm serve" >/dev/null 2>&1; then
    say "vLLM is already running. Stopping previous instance..."
    pkill -f "vllm serve" || true
    sleep 3
fi

say "Launching vLLM serving $SERVED_MODEL on port $PORT..."
nohup vllm serve "$SERVED_MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --guided-decoding-backend xgrammar \
    --served-model-name "$MODEL_ALIAS" \
    > /content/vllm-serve.log 2>&1 &

say "Waiting for vLLM health check on http://localhost:$PORT/health..."
timeout 180 bash -c "until curl -sf http://localhost:$PORT/health >/dev/null 2>&1; do sleep 3; done" || {
    echo "vLLM failed to start. Last log lines:"
    tail -n 30 /content/vllm-serve.log
    exit 1
}

say "vLLM online and ready for OpenAI-compatible completions on port $PORT"
