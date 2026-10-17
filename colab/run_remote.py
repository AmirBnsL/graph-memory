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

    argv = sys.argv[1:] or shlex.split(os.environ.get("EVAL_ARGS", ""))
    print("eval args:", argv or "(none)")
    rc = subprocess.run(
        [PY, os.path.join(REPO, "src", "eval.py"), *argv], cwd=REPO
    ).returncode
    print(f"eval exit={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
