#!/usr/bin/env python3
"""Sweep ONE variable across its values, on the VM, as a single detached process.

One process for the whole sweep, not one `colab exec` per value: holding a
websocket for an hour is exactly how you get "Timeout waiting for reply".

    colab exec -s gm-dev --env DETACH=1 --env VAR=--k --env SWEEP=2,5,10,25,50 \
        --env BASE_ARGS="--mode vector" -f colab/run_sweep.py

Then poll with colab/tail_log.py. Each value re-enters through run_remote.py, so
it gets the same git fast-forward, dataset hash gate, and LLM preflight as a
normal run -- the sweep is just a loop around that, not a second code path.

Env:
    VAR        the flag to vary            (default --k)
    SWEEP      comma-separated values      (default 2,5,10,25,50 -- the paper's grid)
    BASE_ARGS  everything held fixed       (default --mode vector)
    DETACH=1   fork into the background and return immediately
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

REPO = os.environ.get("REPO_DIR", "/content/graph-memory")
PY = sys.executable or "python3"
DRIVER = os.path.join(REPO, "colab", "run_remote.py")
LOG = os.path.join(REPO, "results", "sweep.log")

VAR = os.environ.get("VAR", "--k")
VALUES = [v.strip() for v in os.environ.get("SWEEP", "2,5,10,25,50").split(",") if v.strip()]
BASE = os.environ.get("BASE_ARGS", "--mode vector")

# Lines worth echoing into the live log from each run's captured output.
KEEP = ("F1", "recall", "refusal", "cat", "ERROR", "WARNING", "DRIVER_EXIT", "exit=")


def main() -> int:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(msg: str = "") -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    say(f"sweep start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"VAR={VAR}  VALUES={VALUES}  BASE_ARGS={BASE!r}  REPO={REPO}")

    for i, v in enumerate(VALUES, 1):
        args = f"{BASE} {VAR} {v}"
        say(f"\n===== [{i}/{len(VALUES)}] {VAR} {v}  args={args!r}  "
            f"at {time.strftime('%H:%M:%S')} =====")
        env = dict(os.environ, EVAL_ARGS=args)
        t0 = time.time()
        r = subprocess.run([PY, DRIVER], env=env, cwd=REPO,
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        log.write("----- raw driver output -----\n" + out + "\n")
        say("\n".join(l for l in out.splitlines() if any(t in l for t in KEEP)))
        say(f"===== [{i}/{len(VALUES)}] {VAR} {v} exit={r.returncode} "
            f"in {time.time() - t0:.0f}s =====")

    say(f"\nsweep done {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    if os.environ.get("DETACH") == "1":
        # Re-exec the committed copy (this one may live outside the repo). Parent
        # returns at once so `colab exec` never holds a long-lived connection.
        # Child stdout goes to a SEPARATE file: if it shared LOG with the child's
        # own handle, the two offsets would fight and corrupt the log.
        child_env = dict(os.environ, DETACH="0")
        stdout_log = LOG.replace(".log", ".stdout.log")
        with open(stdout_log, "w") as lf:
            proc = subprocess.Popen([PY, os.path.join(REPO, "colab", "run_sweep.py")],
                                    env=child_env, cwd=REPO, stdout=lf,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        print(f"detached pid={proc.pid}\nlog={LOG}\nstdout={stdout_log}")
    else:
        print(f"SWEEP_EXIT={main()}")
