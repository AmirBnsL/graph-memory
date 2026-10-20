#!/usr/bin/env python3
import glob
import os
import subprocess
import sys
import time

REPO = "/content/graph-memory"
os.chdir(REPO)

# 1. Remove any old truncated k=25 result file
old_k25 = glob.glob(os.path.join(REPO, "results", "*longmemeval*k25*.json"))
for f in old_k25:
    print(f"Removing old truncated k=25 run: {f}")
    os.remove(f)

# 2. Launch eval.py
cmd = [
    sys.executable,
    os.path.join(REPO, "src", "eval.py"),
    "benchmark=longmemeval",
    "k=25",
]
print(f"Starting execution: {' '.join(cmd)}", flush=True)

log_path = os.path.join(REPO, "results", "eval_k25_32k.log")
with open(log_path, "w", buffering=1) as lf:
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        lf.write(line)
        if any(marker in line for marker in ["[", "cat", "F1", "recall", "Saved", "rows", "benchmark="]):
            print(line.rstrip(), flush=True)
    rc = proc.wait()

print(f"Execution finished with exit code {rc}", flush=True)

# 3. Find and print the newly written result file
new_k25 = glob.glob(os.path.join(REPO, "results", "*longmemeval*k25*.json"))
print(f"New k=25 result files: {new_k25}", flush=True)
if new_k25:
    import json
    data = json.load(open(new_k25[0]))
    print("=== SUMMARY METRICS ===")
    print("F1_paper:      ", data.get("F1_paper"))
    print("F1_answerable: ", data.get("F1_answerable"))
    print("EM:            ", data.get("EM"))
    print("recall@k:      ", data.get("recall@k"))
    print("by_category:   ", data.get("by_category"))

sys.exit(rc)
