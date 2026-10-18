#!/usr/bin/env python3
"""Detach a Hydra multirun on the VM, as one background process.

The sweep is now DATA (a --multirun override on the experiment config), so the
hand-rolled loop -- the VALUES iteration, the skip logic, the per-value log
parsing -- is gone. This shim only does what Colab forces:

  * fork into the background, so `colab exec` never holds a websocket for an
    hour (that is how you get "Timeout waiting for reply")
  * write the start/done markers the local watcher greps for

    colab exec -s gm-dev --env DETACH=1 --env SWEEP=2,5,10,25,50 -f colab/run_sweep.py

Then poll with colab/tail_log.py and pull results with colab/pull_results.py.

Resume is automatic and lives in src/eval.py::find_existing: a run whose
manifest matches on config + code + env hashes is skipped, so relaunching the
same command runs only what is missing. Note that a CHANGED config is not
skippable -- by design, since editing conf/ is meant to invalidate results.

Env:
    VAR        the config key to sweep        (default k)
    SWEEP      comma-separated values        (default 2,5,10,25,50 -- the paper's grid)
    BASE_ARGS  extra Hydra overrides held fixed (default "")
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

# Hydra overrides: --multirun k=2,5,10,25,50 [more overrides]
OVERRIDES = [f"--multirun", f"{VAR}=" + ",".join(VALUES)] + (BASE.split() if BASE else [])


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    log = open(LOG, "a", buffering=1)

    def say(msg: str = "") -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    say(f"sweep start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"--multirun {OVERRIDES}  (one process; resume is handled by src/eval.py)")

    env = dict(os.environ, EVAL_ARGS=" ".join(OVERRIDES))
    r = subprocess.run([PY, DRIVER], env=env, cwd=REPO,
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    log.write("----- raw driver output -----\n" + out + "\n")
    say(f"DRIVER_EXIT={r.returncode}")
    say(f"\nsweep done {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return r.returncode


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
