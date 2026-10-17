#!/usr/bin/env python3
"""Run the benchmark ON the Colab VM, driven from a local file.

Sent to the remote kernel by `colab exec -f`, so the local repo stays the only
place code is authored:

    colab exec -s gm-dev --timeout 0 --env EVAL_ARGS="--mode vector -k 10" \
        -f colab/run_remote.py

Steps, in order:
    1. fast-forward the VM checkout to origin/master (git is the record, not sync)
    2. verify the pinned dataset hash BEFORE spending GPU time
    3. run src/eval.py inside that checkout
    4. report the result path so it can be pulled back with `colab download`

Why a driver instead of `-f src/eval.py` directly: the kernel writes the sent
file outside the repo, and src/ uses flat __file__-anchored imports that only
resolve when the script sits beside its siblings. The driver re-enters through
subprocess so `src/eval.py` runs exactly as it does locally.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys

REPO = os.environ.get("REPO_DIR", "/content/graph-memory")
PY = sys.executable or "python3"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)


def main() -> int:
    if not os.path.isdir(os.path.join(REPO, ".git")):
        print(f"ERROR: no git checkout at {REPO}. Run colab/vm_setup.sh first.")
        return 2

    print(git("fetch", "--quiet", "origin").returncode, "fetch")
    git("checkout", "--quiet", "master")
    ff = git("merge", "--ff-only", "--quiet", "origin/master")
    if ff.returncode != 0:
        print("WARNING: fast-forward failed -- VM tree has local commits or edits")
        print(ff.stderr.strip()[:400])

    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    dirty = git("status", "--porcelain").stdout.strip()
    print(f"HEAD={head} dirty={'YES: ' + dirty.replace(chr(10), '; ') if dirty else 'no'}")

    pin_file = os.path.join(REPO, "provenance", "datasets.json")
    if not os.path.exists(pin_file):
        print("ERROR: provenance/datasets.json missing")
        return 3
    pin = json.loads(open(pin_file).read())["locomo"]
    dpath = os.path.join(REPO, pin["local_path"])
    if os.path.exists(dpath):
        h = hashlib.sha256(open(dpath, "rb").read()).hexdigest()
        ok = h == pin["sha256"]
        print(f"dataset sha256 {'OK' if ok else 'MISMATCH'}: {h}")
        if not ok:
            print(f"expected {pin['sha256']}", file=sys.stderr)
            print("refusing to run against an unpinned dataset", file=sys.stderr)
            return 4
    else:
        print(f"WARNING: {dpath} missing -- the loader will raise FileNotFoundError")

    # The kernel's own sys.argv is IPython's launch flags, NOT ours. Never read it.
    argv = shlex.split(os.environ.get("EVAL_ARGS", ""))
    print(f"EVAL_ARGS={os.environ.get('EVAL_ARGS')!r} -> {argv or '(none)'}")
    if not argv:
        print("WARNING: no eval args; eval.py will use its own defaults")

    print("-- preflight: one LLM call --", flush=True)
    pf = subprocess.run(
        [PY, os.path.join(REPO, "src", "llm.py")], cwd=REPO,
        capture_output=True, text=True, timeout=600,
    )
    print((pf.stdout or pf.stderr).strip()[:400])
    if pf.returncode != 0:
        print("preflight FAILED -- not starting the run", file=sys.stderr)
        return 5

    cmd = [PY, os.path.join(REPO, "src", "eval.py"), *argv]
    print("$", " ".join(cmd), flush=True)
    logf = os.path.join(REPO, "results", "eval.log")

    # DETACH=1 for long runs: `colab exec` holding a websocket for an hour is how
    # you get "Timeout waiting for reply". Detach, then poll the log.
    if os.environ.get("DETACH") == "1":
        os.makedirs(os.path.dirname(logf), exist_ok=True)
        with open(logf, "w") as lf:
            proc = subprocess.Popen(
                cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"detached pid={proc.pid}\nlog={logf}")
        return 0

    # Capture: a child inheriting fd 1 writes straight past the kernel's output
    # capture, and every line is silently lost (this is exactly what happened on
    # the first run -- eval exited 0 and we saw none of its output).
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    open(logf, "w").write(r.stdout + r.stderr)

    lines = r.stdout.splitlines()
    if len(lines) > 60:
        print(f"[{len(lines)} output lines -- showing the last 40]")
        print("\n".join(lines[-40:]))
    else:
        print(r.stdout)
    if r.stderr.strip():
        print("--- stderr ---")
        print(r.stderr[-2000:])

    print(f"eval exit={r.returncode}  (full log on VM: {logf})")
    return r.returncode


if __name__ == "__main__":
    # Deliberately not SystemExit: inside a Jupyter kernel that only produces a
    # traceback, and `colab exec` does not propagate the code anyway. Parse the
    # printed DRIVER_EXIT line instead.
    print(f"DRIVER_EXIT={main()}")
