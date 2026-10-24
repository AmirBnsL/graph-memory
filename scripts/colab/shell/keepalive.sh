#!/usr/bin/env bash
# Keep a detached sweep's Colab session alive AND pull results as they land.
#
# Why this exists: on the free tier a detached background process does NOT count
# as activity. A VM with no attached client is reclaimed (~90 min) mid-sweep, and
# /content is wiped with it -- exactly how the second sweep launch died. Every
# `colab` CLI call re-touches the session, so looping pull_results.py both keeps
# the VM alive and checkpoints the finished k's locally.
#
#     colab exec -s gm-dev --env DETACH=1 --env SWEEP=2,5,10,25,50 -f colab/run_sweep.py
#     colab/keepalive.sh gm-dev          # second terminal; Ctrl-C to stop
#
# Stops on the sweep's terminal marker in results/sweep.log. Change the cadence
# with INTERVAL=120 colab/keepalive.sh gm-dev.
set -u

SESSION="${1:-${SESSION:-gm-dev}}"
INTERVAL="${INTERVAL:-300}"
cd "$(dirname "$0")/.." || exit 1

echo "keepalive: session=$SESSION, every ${INTERVAL}s -- Ctrl-C to stop"
while true; do
    python3 colab/pull_results.py --session "$SESSION" --push | tail -1
    if grep -aqE "sweep (done|FAILED)" results/sweep.log 2>/dev/null; then
        echo "keepalive: terminal marker in results/sweep.log -- stopping"
        break
    fi
    sleep "$INTERVAL"
done
