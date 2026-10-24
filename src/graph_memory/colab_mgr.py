#!/usr/bin/env python3
"""Unified Colab Manager for graph-memory.

Provides a clean, reproducible interface to operate remote Google Colab VMs
using the `colab` CLI tool.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "conf").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[2]


ROOT = _find_repo_root()
SCRIPTS_DIR = ROOT / "scripts" / "colab"


def run_cmd(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a shell command with real-time output forwarding."""
    print(f"==> {' '.join(cmd)}")
    return subprocess.run(cmd, timeout=timeout)


def do_setup(args: argparse.Namespace) -> None:
    session = args.session
    bundle_path = "/tmp/gm.bundle"
    print(f"Creating git bundle at {bundle_path}...")
    run_cmd(["git", "bundle", "create", bundle_path, "--all"])
    
    print(f"Uploading bundle to Colab session {session}...")
    run_cmd(["colab", "upload", "-s", session, bundle_path, "/content/gm.bundle"])
    
    print("Executing remote VM setup script...")
    run_cmd([
        "colab", "exec", "-s", session, "--timeout", "1800",
        "-f", str(SCRIPTS_DIR / "run_setup.py"),
    ])


def do_run(args: argparse.Namespace) -> None:
    session = args.session
    eval_args = f"--mode {args.mode} -k {args.k}"
    if args.benchmark:
        eval_args += f" benchmark={args.benchmark}"
    if args.extra:
        eval_args += " " + " ".join(args.extra)

    print(f"Launching evaluation on Colab session {session}...")
    run_cmd([
        "colab", "exec", "-s", session, "--timeout", str(args.timeout),
        "--env", f"EVAL_ARGS={eval_args}",
        "-f", str(SCRIPTS_DIR / "run_remote.py"),
    ])


def do_sweep(args: argparse.Namespace) -> None:
    session = args.session
    env_args = [
        "--env", "DETACH=1",
        "--env", f"SWEEP={args.sweep}",
        "--env", f"RETRIEVER={args.retriever}",
    ]
    if args.extra:
        env_args.extend(["--env", f"EXTRA_ARGS={' '.join(args.extra)}"])

    print(f"Launching detached sweep on Colab session {session} (values: {args.sweep})...")
    cmd = ["colab", "exec", "-s", session] + env_args + ["-f", str(SCRIPTS_DIR / "run_sweep.py")]
    run_cmd(cmd)
    print("\nSweep launched! Remember to keep alive or pull results:")
    print(f"  uv run colab-mgr pull --session {session}")
    print(f"  uv run colab-mgr tail --session {session}")


def do_tail(args: argparse.Namespace) -> None:
    session = args.session
    cmd = [
        "colab", "exec", "-s", session,
        "--env", f"TAIL={args.lines * 100}",
        "-f", str(SCRIPTS_DIR / "tail_log.py"),
    ]
    run_cmd(cmd)


def do_pull(args: argparse.Namespace) -> None:
    # Ensure root is in sys.path
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.colab.pull_results import pull
    print(f"Pulling results from Colab session {args.session}...")
    pull(args.session, args.remote_dir, fresh=args.fresh)


def do_keepalive(args: argparse.Namespace) -> None:
    session = args.session
    keepalive_sh = SCRIPTS_DIR / "shell" / "keepalive.sh"
    print(f"Starting keepalive loop for session {session}...")
    run_cmd(["bash", str(keepalive_sh), session])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="colab-mgr",
        description="Reproducible Google Colab VM Operations Manager for graph-memory",
    )
    parser.add_argument("--session", "-s", default="gm-dev", help="Colab session name (default: gm-dev)")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # setup
    subparsers.add_parser("setup", help="Upload bundle and set up remote Colab environment")

    # run
    p_run = subparsers.add_parser("run", help="Run a single evaluation condition remotely")
    p_run.add_argument("--mode", "-m", default="vector", help="Evaluation mode (default: vector)")
    p_run.add_argument("--k", "-k", type=int, default=10, help="k context items (default: 10)")
    p_run.add_argument("--benchmark", "-b", default=None, help="Benchmark override")
    p_run.add_argument("--timeout", type=int, default=0, help="Execution timeout in seconds (0 = infinite)")
    p_run.add_argument("extra", nargs="*", help="Extra evaluation arguments")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Launch detached sweep on the remote VM")
    p_sweep.add_argument("--sweep", default="2,5,10,25,50", help="Comma-separated grid (default: 2,5,10,25,50)")
    p_sweep.add_argument("--retriever", default="qwen3-06b", help="Retriever group (default: qwen3-06b)")
    p_sweep.add_argument("extra", nargs="*", help="Extra Hydra arguments")

    # tail
    p_tail = subparsers.add_parser("tail", help="Tail the sweep log from the remote VM")
    p_tail.add_argument("--lines", "-n", type=int, default=40, help="Approximate lines to tail (default: 40)")

    # pull
    p_pull = subparsers.add_parser("pull", help="Pull results and manifests to local results/")
    p_pull.add_argument("--remote-dir", default="/content/graph-memory/results", help="Remote results directory")
    p_pull.add_argument("--fresh", action="store_true", help="Force re-download of existing results")

    # keepalive
    subparsers.add_parser("keepalive", help="Run background keepalive script")

    args = parser.parse_args()

    if args.action == "setup":
        do_setup(args)
    elif args.action == "run":
        do_run(args)
    elif args.action == "sweep":
        do_sweep(args)
    elif args.action == "tail":
        do_tail(args)
    elif args.action == "pull":
        do_pull(args)
    elif args.action == "keepalive":
        do_keepalive(args)


if __name__ == "__main__":
    main()
