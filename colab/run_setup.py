#!/usr/bin/env python3
"""Run colab/vm_setup.sh on the VM, with live output.

`colab exec -f` executes a PYTHON file in the kernel, so the bash setup script
cannot be sent directly. This shim exists for one reason: the repo has to exist
before `vm_setup.sh` can be run from it, and `vm_setup.sh` itself is what clones
the repo.

    git bundle create /tmp/gm.bundle --all
    colab upload -s gm-dev /tmp/gm.bundle /content/gm.bundle
    colab exec -s gm-dev --timeout 1800 -f colab/run_setup.py

Source order, deliberately:
  1. a git bundle at $BUNDLE, if one was uploaded -- a bundle carries the exact
     commit with NO credential on the VM at all. Preferred.
  2. the SSH remote otherwise -- requires a key on the VM, which a fresh runtime
     does not have (and which we do not want: the previous session carried a
     full-account key, the opposite of this repo's "no secrets on the VM" rule).

Output is captured line-by-line and re-printed: a child that inherits fd 1 writes
straight past the kernel's output capture and is silently lost.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.environ.get("REPO_DIR", "/content/graph-memory")
REPO_SSH = os.environ.get("REPO_SSH", "git@github.com:AmirBnsL/graph-memory.git")
BUNDLE = os.environ.get("BUNDLE", "/content/gm.bundle")
SETUP = os.path.join(REPO, "colab", "vm_setup.sh")


def run(cmd: list[str], cwd: str | None = None) -> int:
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:                      # live, and captured
        print(line.rstrip(), flush=True)
    return p.wait()


def main() -> int:
    if not os.path.isdir(os.path.join(REPO, ".git")):
        if os.path.exists(BUNDLE):
            print(f"cloning from bundle {BUNDLE} -> {REPO} (no credential on the VM)",
                  flush=True)
            rc = run(["git", "clone", "--quiet", BUNDLE, REPO])
            if rc == 0:
                # Point origin at the real remote: a fetch that fails because the
                # VM has no key is then LOUD, instead of silently serving the
                # bundle forever while looking like an up-to-date checkout.
                run(["git", "-C", REPO, "remote", "set-url", "origin", REPO_SSH])
        else:
            print(f"no bundle at {BUNDLE}; falling back to {REPO_SSH}", flush=True)
            rc = run(["git", "clone", "--quiet", REPO_SSH, REPO])
        if rc != 0:
            print("ERROR: clone failed", file=sys.stderr)
            return rc
    elif os.path.exists(BUNDLE):
        # Existing checkout: fast-forward it from the uploaded bundle. git fetches
        # from a bundle path directly, so updating code costs an upload -- no
        # session recreation, no re-pulling the model, and still no credential.
        # Without this, a fix after setup could only reach the VM by wiping it.
        print(f"checkout exists -- updating from bundle {BUNDLE}", flush=True)
        rc = run(["git", "-C", REPO, "fetch", "--quiet", BUNDLE, "master"])
        if rc == 0:
            ff = run(["git", "-C", REPO, "merge", "--ff-only", "--quiet", "FETCH_HEAD"])
            if ff.returncode != 0:
                print("WARNING: fast-forward from bundle failed "
                      "(local commits or edits?)", flush=True)
        else:
            print("WARNING: fetch from bundle failed", flush=True)
    if not os.path.exists(SETUP):
        print(f"ERROR: {SETUP} missing after clone", file=sys.stderr)
        return 2
    rc = run(["bash", SETUP])
    print(f"SETUP_EXIT={rc}")
    return rc


if __name__ == "__main__":
    print(f"DRIVER_EXIT={main()}")
