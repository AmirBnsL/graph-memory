#!/usr/bin/env python3
"""Run the LLM-free recall probe on the VM (src/recall_probe.py).

The probe needs NO LLM server -- retrieval recall is computable from embeddings
plus the gold evidence ids -- but it runs on the VM anyway: the T4 encodes a
0.6B model ~30-80x faster than a laptop CPU, and an index built here is the
same artifact the end-to-end sweep will reuse.

    colab exec -s gm-dev --timeout 1800 -f colab/run_probe.py

Env:
    RETRIEVER   config group name under conf/retriever/   (default minilm)
    EXTRA_ARGS  extra Hydra overrides held after the group override
                (e.g. "retriever.query_prefix=" to ablate the instruction)
    LIMITS      optional limits overrides, e.g. "limits.n_conv=1"
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.environ.get("REPO_DIR", "/content/graph-memory")
PY = sys.executable or "python3"
RETRIEVER = os.environ.get("RETRIEVER", "minilm")
EXTRA = os.environ.get("EXTRA_ARGS", "")
LIMITS = os.environ.get("LIMITS", "")

CMD = ([PY, "src/recall_probe.py", f"retriever={RETRIEVER}"]
       + (EXTRA.split() if EXTRA else [])
       + (LIMITS.split() if LIMITS else []))


def main() -> int:
    # run_setup.py's lesson: a child that inherits fd 1 writes straight past the
    # kernel's output capture and is silently lost -- capture and re-print.
    p = subprocess.Popen(CMD, cwd=REPO, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    print(f"PROBE_EXIT={p.wait()}")
    return 0


if __name__ == "__main__":
    print(f"cmd: {' '.join(CMD)}", flush=True)
    main()
