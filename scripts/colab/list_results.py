#!/usr/bin/env python3
"""Recursively list all result files on the VM for pull_results.py."""
import os

base = os.environ.get("REMOTE_DIR", "/content/graph-memory/results")
if os.path.isdir(base):
    for r, _, fs in os.walk(base):
        for f in fs:
            if f.endswith(".json") or f.endswith(".log") or f == "INDEX.md":
                print(os.path.relpath(os.path.join(r, f), base))
