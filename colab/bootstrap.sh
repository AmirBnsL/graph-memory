#!/usr/bin/env bash
#
# Bootstrap the Colab dev VM into a ready workspace.
#
# Run this from your LOCAL machine after creating (or waking) a Colab session.
# It is idempotent: safe to run on every fresh session.
#
#   bash colab/bootstrap.sh
#
# What it does:
#   1. injects the repo-scoped deploy key onto the VM (SSH auth to GitHub)
#   2. clones the repo if the VM is cold, otherwise reports its state
#   3. installs the cloud delta deps on top of the stock Colab image
#   4. prints the VS Code entry point
#
set -euo pipefail

SSH_HOST="${SSH_HOST:-colab}"                 # ~/.ssh/config alias -> colab ssh --proxy-mode -s gm-dev
REPO_DIR="${REPO_DIR:-/content/graph-memory}"
REPO_SSH="${REPO_SSH:-git@github.com:AmirBnsL/graph-memory.git}"
KEY_LOCAL="${KEY_LOCAL:-$HOME/.ssh/gm_deploy}"

# The stock Colab image already ships torch(CUDA), transformers, sentence-transformers,
# datasets, accelerate, networkx, pandas. We only add what is genuinely missing.
# NOTE: never install our local requirements.txt here -- it pins torch==+cpu.
CLOUD_DELTA=(bitsandbytes)

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "$KEY_LOCAL" ] || die "deploy key not found at $KEY_LOCAL"

say "checking SSH to '$SSH_HOST'"
ssh -o ConnectTimeout=30 "$SSH_HOST" 'echo ok' >/dev/null || die "cannot reach $SSH_HOST (is the session running? try: colab new -s gm-dev --gpu T4)"

say "installing deploy key on the VM"
ssh "$SSH_HOST" 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
scp -q "$KEY_LOCAL" "$SSH_HOST:~/.ssh/gm_deploy"
ssh "$SSH_HOST" 'chmod 600 ~/.ssh/gm_deploy
  touch ~/.ssh/config && chmod 600 ~/.ssh/config
  if ! grep -q "Host github.com" ~/.ssh/config; then
    printf "Host github.com\n  IdentityFile ~/.ssh/gm_deploy\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n" >> ~/.ssh/config
  fi
  ssh -o ConnectTimeout=15 -T git@github.com 2>&1 | head -1'

say "syncing repository"
ssh "$SSH_HOST" "
  git config --global --add safe.directory '$REPO_DIR' 2>/dev/null || true
  git config --global user.name  '$(git config --get user.name)'
  git config --global user.email '$(git config --get user.email)'
  if [ ! -d '$REPO_DIR/.git' ]; then
    echo 'cold VM -> cloning'
    git clone '$REPO_SSH' '$REPO_DIR'
  fi
  cd '$REPO_DIR'
  git remote set-url origin '$REPO_SSH'
  if ! git fetch --all --quiet; then
    echo 'WARNING: git fetch failed -- the deploy key is probably not authorized on GitHub yet.'
    echo '         Add ~/.ssh/gm_deploy.pub at https://github.com/AmirBnsL/graph-memory/settings/keys'
    echo '         (tick \"Allow write access\"), then re-run this script.'
  fi
  echo \"branch: \$(git branch --show-current)\"
  git status --short --branch
"

say "installing cloud delta: ${CLOUD_DELTA[*]}"
ssh "$SSH_HOST" "uv pip install --system --quiet ${CLOUD_DELTA[*]}"

say "done. open the workspace with:"
echo
echo "    code --remote ssh-remote+$SSH_HOST $REPO_DIR"
echo
