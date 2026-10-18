#!/usr/bin/env python3
"""Run colab/vm_setup.sh on the VM, with live output.

`colab exec -f` executes a PYTHON file in the kernel, so the bash setup script
cannot be sent directly. This shim exists for one reason: the repo has to exist
before `vm_setup.sh` can be run from it, and `vm_setup.sh` itself is what clones
the repo. So: clone (idempotent, same URL the setup script uses), then bash.

    colab new -s gm-dev --gpu T4
    colab exec -s gm-dev --timeout 1800 -f colab/run_setup.py

Output is captured line-by-line and re-printed: a child that inherits fd 1 writes
straight past the kernel's output capture and is silently lost.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.environ.get("REPO_DIR", "/content/graph-memory")
REPO_SSH = os.environ.get("REPO_SSH", "git@github.com:AmirBnsL/graph-memory.git")
SETUP = os.path.join(REPO, "colab", "vm_setup.sh")


def run(cmd: list[str], cwd: str | None = None) -> int:
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:                      # live, and captured
        print(line.rstrip(), flush=True)
    return p.wait()


def main() -> int:
    if not os.path.isdir(os.path.join(REPO, ".git")):
        print(f"cloning {REPO_SSH} -> {REPO}", flush=True)
        rc = run(["git", "clone", "--quiet", REPO_SSH, REPO])
        if rc != 0:
            print("ERROR: clone failed", file=sys.stderr)
            return rc
    if not os.path.exists(SETUP):
        print(f"ERROR: {SETUP} missing after clone", file=sys.stderr)
        return 2
    rc = run(["bash", SETUP])
    print(f"SETUP_EXIT={rc}")
    return rc


if __name__ == "__main__":
    print(f"DRIVER_EXIT={main()}")
