#!/usr/bin/env python3
"""Unified Google Colab Manager and CLI for graph-memory.

Provides a clean, reproducible interface to operate remote Google Colab VMs
using the `colab` CLI tool.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent.parent


def run_cmd(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a shell command with real-time output forwarding."""
    print(f"==> {' '.join(cmd)}")
    return subprocess.run(cmd, timeout=timeout)


def do_mount(args: argparse.Namespace) -> None:
    session = args.session
    path = args.path or "/content/drive"
    print(f"Mounting Google Drive in Colab session {session} at {path}...")
    run_cmd(["colab", "drivemount", "-s", session, path])
    print("Initializing persistent storage directories & symlinks...")
    try:
        run_cmd([
            "colab", "exec", "-s", session, "--timeout", "60",
            "--env", "EVAL_ARGS=mount",
            "-f", str(SCRIPTS_DIR / "run_remote.py"),
        ])
    except Exception as e:
        print(f"Note: Symlink initialization will complete during setup: {e}")


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
    profiles = args.profiles if hasattr(args, "profiles") and args.profiles else ["smoke"]
    eval_args = "run " + " ".join(profiles)
    if hasattr(args, "extra") and args.extra:
        eval_args += " " + " ".join(args.extra)

    env_args = ["--env", f"EVAL_ARGS={eval_args}"]
    detach = getattr(args, "detach", False)
    if detach:
        env_args.extend(["--env", "DETACH=1"])

    timeout_val = str(getattr(args, "timeout", 0) or 0)
    print(f"Launching evaluation on Colab session {session} ({eval_args}, detach={detach})...")
    cmd = [
        "colab", "exec", "-s", session, "--timeout", timeout_val,
        *env_args,
        "-f", str(SCRIPTS_DIR / "run_remote.py"),
    ]
    run_cmd(cmd)
    if detach:
        print("\nRun launched in detached mode! Monitor or pull results with:")
        print(f"  python3 scripts/colab/cli.py tail --session {session}")
        print(f"  python3 scripts/colab/cli.py pull --session {session}")


def do_smoke(args: argparse.Namespace) -> None:
    args.profiles = ["smoke"]
    args.extra = []
    args.detach = False
    do_run(args)


def do_extract(args: argparse.Namespace) -> None:
    session = args.session
    eval_args = f"extract --benchmark {args.benchmark}"
    if args.limit:
        eval_args += f" --limit {args.limit}"
    if args.sample_per_cat:
        eval_args += f" --sample-per-cat {args.sample_per_cat}"
    if args.workers:
        eval_args += f" --workers {args.workers}"
    if args.extra:
        eval_args += " " + " ".join(args.extra)

    env_args = ["--env", f"EVAL_ARGS={eval_args}"]
    if args.detach:
        env_args.extend(["--env", "DETACH=1"])

    timeout_val = str(args.timeout or 0)
    print(f"Launching extraction on Colab session {session} ({eval_args})...")
    cmd = [
        "colab", "exec", "-s", session, "--timeout", timeout_val,
        *env_args,
        "-f", str(SCRIPTS_DIR / "run_remote.py"),
    ]
    run_cmd(cmd)


def do_tail(args: argparse.Namespace) -> None:
    session = args.session
    cmd = [
        "colab", "exec", "-s", session,
        "--env", f"TAIL={args.lines * 100}",
        "-f", str(SCRIPTS_DIR / "tail_log.py"),
    ]
    run_cmd(cmd)


def do_pull(args: argparse.Namespace) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.colab.pull_results import pull
    print(f"Pulling results from Colab session {args.session}...")
    pull(args.session, args.remote_dir, fresh=args.fresh)


def do_status(args: argparse.Namespace) -> None:
    session = args.session
    run_cmd(["colab", "status", "-s", session])


def do_keepalive(args: argparse.Namespace) -> None:
    session = args.session
    keepalive_sh = SCRIPTS_DIR / "shell" / "keepalive.sh"
    print(f"Starting keepalive loop for session {session}...")
    run_cmd(["bash", str(keepalive_sh), session])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="colab-cli",
        description="Reproducible Google Colab VM Operations Manager for graph-memory",
    )
    parser.add_argument("--session", "-s", default="gm-dev", help="Colab session name (default: gm-dev)")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # mount
    p_mount = subparsers.add_parser("mount", help="Mount Google Drive and configure persistent cache")
    p_mount.add_argument("--path", default="/content/drive", help="Mount path (default: /content/drive)")

    # setup
    subparsers.add_parser("setup", help="Upload bundle and set up remote Colab environment")

    # run
    p_run = subparsers.add_parser("run", help="Run one or more declarative experiment YAML profiles remotely")
    p_run.add_argument("profiles", nargs="*", default=["smoke"], help="YAML experiment profile names in conf/experiment/ (e.g. smoke, h1_balanced)")
    p_run.add_argument("--detach", "-d", action="store_true", help="Run detached in background on the VM")
    p_run.add_argument("--timeout", type=int, default=0, help="Execution timeout in seconds (0 = infinite)")
    p_run.add_argument("extra", nargs="*", help="Extra Hydra overrides (e.g. limits.n_conv=5)")

    # smoke
    subparsers.add_parser("smoke", help="Shortcut: run smoke test experiment profile on VM")

    # extract
    p_extract = subparsers.add_parser("extract", help="Pre-extract dialogue turns into facts and triples on the VM")
    p_extract.add_argument("--benchmark", choices=["locomo", "longmemeval"], default="longmemeval")
    p_extract.add_argument("--limit", type=int, default=None, help="Limit number of conversations")
    p_extract.add_argument("--sample-per-cat", type=int, default=None, help="Balanced sample per category")
    p_extract.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    p_extract.add_argument("--detach", "-d", action="store_true", help="Run detached in background")
    p_extract.add_argument("--timeout", type=int, default=0, help="Execution timeout (0 = infinite)")
    p_extract.add_argument("extra", nargs="*", help="Extra extract arguments")

    # tail
    p_tail = subparsers.add_parser("tail", help="Tail the execution log from the remote VM")
    p_tail.add_argument("--lines", "-n", type=int, default=40, help="Approximate lines to tail (default: 40)")

    # pull
    p_pull = subparsers.add_parser("pull", help="Pull results and manifests to local results/")
    p_pull.add_argument("--remote-dir", default="/content/graph-memory/results", help="Remote results directory")
    p_pull.add_argument("--fresh", action="store_true", help="Force re-download of existing results")

    # status
    subparsers.add_parser("status", help="Show remote session status")

    # keepalive
    subparsers.add_parser("keepalive", help="Run background keepalive script")

    args = parser.parse_args()

    if args.action == "mount":
        do_mount(args)
    elif args.action == "setup":
        do_setup(args)
    elif args.action == "run":
        do_run(args)
    elif args.action == "smoke":
        do_smoke(args)
    elif args.action == "extract":
        do_extract(args)
    elif args.action == "tail":
        do_tail(args)
    elif args.action == "pull":
        do_pull(args)
    elif args.action == "status":
        do_status(args)
    elif args.action == "keepalive":
        do_keepalive(args)


if __name__ == "__main__":
    main()
