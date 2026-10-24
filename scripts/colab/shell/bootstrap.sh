#!/usr/bin/env bash
#
# Turn a cold Colab VM into a ready dev workspace.
#
# Run this from your LOCAL machine after creating (or waking) a Colab session:
#
#   bash colab/bootstrap.sh
#
# Idempotent -- safe to run on every fresh session.
#
# Auth model: your GitHub key stays on your laptop. We run a local ssh-agent and
# forward it over the SSH connection (~/.ssh/config: ForwardAgent + IdentityAgent).
# The VM borrows the agent to reach GitHub. No token, no deploy key, no browser.
#
set -euo pipefail

SSH_HOST="${SSH_HOST:-colab}"                 # ~/.ssh/config alias -> colab ssh --proxy-mode -s gm-dev
REPO_DIR="${REPO_DIR:-/content/graph-memory}"
REPO_SSH="${REPO_SSH:-git@github.com:AmirBnsL/graph-memory.git}"
AGENT_SOCK="${AGENT_SOCK:-$HOME/.ssh/gm-agent.sock}"
AGENT_KEY="${AGENT_KEY:-$HOME/.ssh/id_ed25519}"

# The stock Colab image already ships torch(CUDA), transformers, sentence-transformers,
# datasets, accelerate, networkx, pandas. We add only what is genuinely missing.
# NEVER install our local requirements.txt here -- it pins torch==+cpu and would
# clobber the CUDA build with a CPU-only wheel.
CLOUD_DELTA=(bitsandbytes)

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

ensure_agent() {
  export SSH_AUTH_SOCK="$AGENT_SOCK"
  if ssh-add -l >/dev/null 2>&1; then
    say "ssh-agent already up (key forwarding ready)"
    return 0
  fi
  say "starting ssh-agent at $AGENT_SOCK"
  pkill -f "ssh-agent -a $AGENT_SOCK" 2>/dev/null || true
  rm -f "$AGENT_SOCK"
  ssh-agent -a "$AGENT_SOCK" >/dev/null 2>&1
  ssh-add -q "$AGENT_KEY"
  ssh-add -l
}

say "checking SSH to '$SSH_HOST'"
if ! ssh -o ConnectTimeout=30 "$SSH_HOST" 'echo ok' >/dev/null 2>&1; then
  die "cannot reach '$SSH_HOST'.
       The runtime is probably recycled -- SSH is baked in at creation.
       Create a fresh one:  colab new -s gm-dev --gpu T4"
fi

ensure_agent

say "verifying the VM can reach GitHub through the forwarded agent"
ssh "$SSH_HOST" 'ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -T git@github.com 2>&1 | head -1'

say "syncing repository"
ssh "$SSH_HOST" "
  git config --global --add safe.directory '$REPO_DIR' 2>/dev/null || true
  git config --global user.name  '$(git config --get user.name)'
  git config --global user.email '$(git config --get user.email)'
  if [ ! -d '$REPO_DIR/.git' ]; then
    echo 'cold VM -> cloning'
    git clone --quiet '$REPO_SSH' '$REPO_DIR'
  fi
  cd '$REPO_DIR'
  git remote set-url origin '$REPO_SSH'
  git fetch --all --quiet 2>/dev/null || echo 'WARNING: git fetch failed'
  echo \"branch: \$(git branch --show-current)\"
  git status --short --branch
"

say "installing cloud delta: ${CLOUD_DELTA[*]}"
ssh "$SSH_HOST" "uv pip install --system --quiet ${CLOUD_DELTA[*]}"

say "done. your workspace is live:"
echo
echo "    code --remote ssh-remote+$SSH_HOST $REPO_DIR"
echo
echo "  ssh shell:   ssh $SSH_HOST"
echo "  stop meter:  colab stop -s gm-dev"
echo
