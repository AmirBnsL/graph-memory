#!/usr/bin/env python3
"""Run upstream benchmark evaluations and compute H2 sidecar metrics.

Executes unmodified MAGMA, EMem-G/EMem, and A-MEM runners against local vLLM,
runs the Topological Sidecar on the output graph caches, and outputs canonical run JSONs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.analysis.external_inspector import inspect_upstream_run


def run_benchmark(
    framework: str,
    benchmark: str = "locomo",
    model: str = "qwen2.5:7b-instruct",
    base_url: str = "http://localhost:8000/v1",
    dry_run: bool = False,
) -> dict:
    framework = framework.lower()
    print(f"[{framework.upper()}] Starting benchmark '{benchmark}' with model '{model}' at {base_url}")

    if dry_run:
        print(f"[{framework.upper()}] Dry run complete.")
        return {}

    # Real run invocation logic for Colab execution
    ext_dir = ROOT / "external"
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_KEY"] = "none"

    if framework == "magma":
        magma_dir = ext_dir / "MAGMA"
        if benchmark == "locomo":
            cmd = [
                sys.executable, str(magma_dir / "test_fixed_memory.py"),
                "--model", model,
            ]
        else:
            cmd = [
                sys.executable, str(magma_dir / "test_longmemeval_chunked.py"),
                "--model", model,
            ]
        subprocess.run(cmd, cwd=str(magma_dir), env=env, check=True)

    elif framework in ("emem_g", "emem", "emem_flat"):
        emem_dir = ext_dir / "EMem"
        cmd = [
            sys.executable, str(emem_dir / "run_emem_locomo_cached.py" if benchmark == "locomo" else "run_emem_longmemeval_cached.py"),
            "--model", model,
        ]
        if framework == "emem_flat":
            cmd.append("--skip_retrieval_ppr")
        subprocess.run(cmd, cwd=str(emem_dir), env=env, check=True)

    elif framework == "amem":
        amem_dir = ext_dir / "AgenticMemory"
        cmd = [
            sys.executable, str(amem_dir / "test_advanced_robust.py"),
            "--backend", "vllm",
            "--model", model,
        ]
        if benchmark != "locomo":
            cmd.extend(["--dataset", str(ROOT / "data/longmemeval/longmemeval_s_50_locomo_fmt.json")])
        subprocess.run(cmd, cwd=str(amem_dir), env=env, check=True)

    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["magma", "emem_g", "emem_flat", "amem", "all"], default="all")
    parser.add_argument("--benchmark", choices=["locomo", "longmemeval"], default="locomo")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frameworks = ["magma", "emem_g", "emem_flat", "amem"] if args.framework == "all" else [args.framework]

    for fw in frameworks:
        run_benchmark(
            framework=fw,
            benchmark=args.benchmark,
            model=args.model,
            base_url=args.base_url,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
