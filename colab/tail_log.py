#!/usr/bin/env python3
"""Tail a log on the VM without holding a long `colab exec` connection.

    colab exec -s gm-dev --env LOG=/content/graph-memory/results/sweep.log \
        --env TAIL=4000 -f colab/tail_log.py
"""
from __future__ import annotations

import os

REPO = os.environ.get("REPO_DIR", "/content/graph-memory")
LOG = os.environ.get("LOG", os.path.join(REPO, "results", "sweep.log"))
TAIL = int(os.environ.get("TAIL", "4000"))

if not os.path.exists(LOG):
    print(f"no log at {LOG}")
else:
    size = os.path.getsize(LOG)
    with open(LOG) as f:
        f.seek(max(0, size - TAIL))
        text = f.read()
    print(f"[{LOG} -- {size} bytes, showing last {len(text)}]")
    print(text)
