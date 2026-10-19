#!/usr/bin/env python3
"""Detach a Hydra multirun on the VM, as one background process.

The sweep is now DATA (a --multirun override on the experiment config), so the
hand-rolled loop -- the VALUES iteration, the skip logic, the per-value log
parsing -- is gone. This shim only does what Colab forces:

  * fork into the background, so `colab exec` never holds a websocket for an
    hour (that is how you get "Timeout waiting for reply")
  * write the start/done markers the local watcher greps for

    colab exec -s gm-dev --env DETACH=1 --env SWEEP=2,5,10,25,50 \
        --env RETRIEVER=qwen3-06b --env EXTRA_ARGS="retriever.query_prefix=" \
        -f colab/run_sweep.py

Then, MANDATORY, start the keepalive in a second terminal. A detached process
does not count as activity on the free tier: an unattended VM is reclaimed
mid-sweep (~90 min) and /content goes with it. This has bitten us once.

    colab/keepalive.sh gm-dev

It pulls results as they land and stops on the sweep's terminal marker. Poll by
hand with colab/tail_log.py if needed.

Resume is automatic and lives in src/eval.py::find_existing: a run whose
manifest matches on config + code + env hashes is skipped, so relaunching the
same command runs only what is missing. Note that a CHANGED config is not
skippable -- by design, since editing conf/ is meant to invalidate results.

Env:
    VAR        the config key to sweep        (default k)
    SWEEP      comma-separated values        (default 2,5,10,25,50 -- the paper's grid)
    RETRIEVER  retriever config group        (default: conf/config.yaml's -- minilm)
    EXTRA_ARGS extra Hydra overrides         (e.g. "retriever.query_prefix=")
    BASE_ARGS  free-form override list       (kept for older launch scripts)
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
RESULTS = os.path.join(REPO, "results")
LOG = os.path.join(RESULTS, "sweep.log")

VAR = os.environ.get("VAR", "k")
VALUES = [v.strip() for v in os.environ.get("SWEEP", "2,5,10,25,50").split(",") if v.strip()]
BASE = os.environ.get("BASE_ARGS", "")
RETRIEVER = os.environ.get("RETRIEVER", "")
EXTRA = os.environ.get("EXTRA_ARGS", "")

# One vocabulary across the shims. run_probe.py takes RETRIEVER/EXTRA_ARGS, so
# using those names here is the obvious thing to do -- and it used to be a silent
# no-op: the sweep would quietly run the DEFAULT retriever (conf/config.yaml's
# minilm) while the launch command looked correct. Accept both spellings;
# retriever first, so EXTRA_ARGS can still override its keys.
overrides = (([f"retriever={RETRIEVER}"] if RETRIEVER else [])
             + (BASE.split() if BASE else [])
             + (EXTRA.split() if EXTRA else []))

# Hydra overrides: --multirun k=2,5,10,25,50 [more overrides]
OVERRIDES = [f"--multirun", f"{VAR}=" + ",".join(VALUES)] + overrides


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    log = open(LOG, "a", buffering=1)

    def say(msg: str = "") -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    say(f"sweep start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"--multirun {OVERRIDES}  (one process; resume is handled by src/eval.py)")
    if not any(o.startswith("retriever=") for o in OVERRIDES):
        say("NOTE: no retriever= override -- using conf/config.yaml's default (minilm)")

    env = dict(os.environ, EVAL_ARGS=" ".join(OVERRIDES))
    proc = subprocess.Popen([PY, DRIVER], env=env, cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = proc.wait()
    say(f"DRIVER_EXIT={rc}")
    # The marker must reflect the outcome: the local watcher greps for
    # "sweep done", and a failure that prints "done" would false-trigger it.
    say(f"\nsweep done {time.strftime('%Y-%m-%d %H:%M:%S')}"
        if rc == 0 else
        f"\nsweep FAILED (DRIVER_EXIT={rc}) {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return rc


if __name__ == "__main__":
    if os.environ.get("DETACH") == "1":
        # Re-exec the committed copy (this one may live outside the repo). The
        # parent returns at once so `colab exec` never holds the connection, and
        # the child's stdout gets its own file: sharing it with the wrapper's
        # log handle would make the two offsets fight and corrupt the log.
        child_env = dict(os.environ, DETACH="0")
        stdout_log = LOG.replace(".log", ".stdout.log")
        with open(stdout_log, "w") as lf:
            proc = subprocess.Popen([PY, os.path.join(REPO, "colab", "run_sweep.py")],
                                    env=child_env, cwd=REPO, stdout=lf,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        print(f"detached pid={proc.pid}\nlog={LOG}\nstdout={stdout_log}")
    else:
        print(f"SWEEP_EXIT={main()}")
