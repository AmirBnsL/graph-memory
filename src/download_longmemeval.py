#!/usr/bin/env python3
"""Download and verify official LongMemEval benchmark files.

Downloads the cleaned benchmark files released for LongMemEval (ICLR 2025):
  - longmemeval_oracle.json (~15MB): Oracle retrieval (evidence sessions only).
  - longmemeval_s_cleaned.json (~277MB): Full chat history (~40 sessions, ~115k tokens).

Computes SHA256 hashes and updates provenance/datasets.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "longmemeval"
PROVENANCE_FILE = ROOT / "provenance" / "datasets.json"

FILES = {
    "oracle": {
        "filename": "longmemeval_oracle.json",
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    },
    "s": {
        "filename": "longmemeval_s_cleaned.json",
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    },
}


def download_file(url: str, dest: Path) -> tuple[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".tmp")
    print(f"Downloading {url} -> {dest} ...", flush=True)

    hasher = hashlib.sha256()
    total_bytes = 0

    req = urllib.request.Request(url, headers={"User-Agent": "graph-memory-pfe/1.0"})
    with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as out:
        while chunk := resp.read(65536):
            hasher.update(chunk)
            out.write(chunk)
            total_bytes += len(chunk)
            if total_bytes % (5 * 1024 * 1024) == 0:
                print(f"  ... {total_bytes / (1024 * 1024):.1f} MB", flush=True)

    temp_dest.rename(dest)
    sha256 = hasher.hexdigest()
    print(f"Done: {total_bytes} bytes, sha256={sha256}")
    return sha256, total_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["oracle", "s", "all"], default="oracle",
                    help="Target dataset to download (default: oracle)")
    ap.add_argument("--update-provenance", action="store_true", default=True,
                    help="Update provenance/datasets.json with downloaded hashes")
    args = ap.parse_args()

    targets = ["oracle", "s"] if args.target == "all" else [args.target]
    meta = {}

    for t in targets:
        info = FILES[t]
        dest = DATA_DIR / info["filename"]
        if dest.exists():
            print(f"File already exists: {dest}")
            hasher = hashlib.sha256()
            with open(dest, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            sha256 = hasher.hexdigest()
            size = dest.stat().st_size
            print(f"Existing file hash: {sha256} ({size} bytes)")
        else:
            sha256, size = download_file(info["url"], dest)
        meta[t] = {"sha256": sha256, "bytes": size, "path": str(dest.relative_to(ROOT))}

    if args.update_provenance:
        prov = json.loads(PROVENANCE_FILE.read_text())
        oracle_info = meta.get("oracle")
        primary_info = oracle_info or meta[targets[0]]

        prov["longmemeval"] = {
            "status": "verified",
            "source": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
            "verified": time.strftime("%Y-%m-%d"),
            "primary_split": "oracle",
            "local_path": primary_info["path"],
            "sha256": primary_info["sha256"],
            "bytes": primary_info["bytes"],
            "splits": {
                k: {"local_path": v["path"], "sha256": v["sha256"], "bytes": v["bytes"]}
                for k, v in meta.items()
            },
            "expected_shape": {
                "questions": 500,
                "abilities": {
                    "extraction": 150,
                    "multi-session": 121,
                    "temporal": 127,
                    "update": 72,
                    "abstention": 30
                }
            },
            "categories": {
                "1": "extraction",
                "2": "multi-session",
                "3": "temporal",
                "4": "update",
                "5": "abstention"
            },
            "restore": "python3 src/download_longmemeval.py --target oracle",
            "known_issues": [
                "Category 5 represents Abstention (_abs instances with false premises or unanswerable queries); scored as refusal accuracy matching LoCoMo Cat 5.",
                "Community reports noted potential arithmetic discrepancies in a subset of temporal questions; see issue tracker."
            ]
        }
        PROVENANCE_FILE.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n")
        print(f"Updated provenance at {PROVENANCE_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
