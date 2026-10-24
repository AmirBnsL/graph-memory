"""Unified CLI for graph-memory.

Allows running benchmark evaluations, diagnostics, and table generation
with a single declarative interface modeled after leading research repositories.

Usage:
    # Run evaluation via Hydra
    python -m graph_memory.cli eval benchmark=longmemeval mode=memory/hipporag k=10

    # Simplified direct flags
    python -m graph_memory.cli run --benchmark longmemeval --mode memory/hipporag --k 10 --limit 5

    # Generate summary tables
    python -m graph_memory.cli tables

    # Compute graph metrics
    python -m graph_memory.cli metrics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def get_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "conf").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def run_eval(args: list[str]) -> None:
    """Delegate to the Hydra-configured evaluation pipeline."""
    # Ensure src and repo root are in sys.path
    root = get_repo_root()
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from graph_memory.pipeline.eval import main as eval_main
    eval_main()


def run_simple(args: argparse.Namespace) -> None:
    """Run an evaluation using direct argparse flags, translated into Hydra overrides."""
    root = get_repo_root()
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    overrides = []
    if args.benchmark:
        overrides.append(f"benchmark={args.benchmark}")
    if args.mode:
        overrides.append(f"mode={args.mode}")
    if args.k is not None:
        overrides.append(f"k={args.k}")
    if args.limit is not None:
        overrides.append(f"limit={args.limit}")
    if args.reader:
        overrides.append(f"reader={args.reader}")
    if args.retriever:
        overrides.append(f"retriever={args.retriever}")
    if args.extra:
        overrides.extend(args.extra)

    # Set sys.argv for hydra
    prog = sys.argv[0]
    sys.argv = [prog] + overrides
    from graph_memory.pipeline.eval import main as eval_main
    eval_main()


def run_tables(args: argparse.Namespace) -> None:
    root = get_repo_root()
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from graph_memory.analysis import make_tables
    make_tables.main()


def run_metrics(args: argparse.Namespace) -> None:
    root = get_repo_root()
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from graph_memory.analysis import graph_metrics
    graph_metrics.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="graph-memory",
        description="Graph Memory: Long-Term Conversational Memory and Evaluation",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to run")

    # 'run' subcommand: user-friendly flags
    run_parser = subparsers.add_parser("run", help="Run benchmark evaluation with simple flags")
    run_parser.add_argument("--benchmark", "-b", choices=["longmemeval", "locomo"], default="longmemeval",
                            help="Benchmark dataset (default: longmemeval)")
    run_parser.add_argument("--mode", "-m", default="vector",
                            help="Memory/retrieval mode (e.g. vector, full, memory/flat, memory/hipporag, memory/temporal_bteg)")
    run_parser.add_argument("--k", "-k", type=int, default=10, help="Top-k retrieved context items")
    run_parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of conversations")
    run_parser.add_argument("--reader", default=None, help="Reader model override (e.g. qwen25-7b)")
    run_parser.add_argument("--retriever", default=None, help="Retriever override (e.g. minilm, gemma-300m, qwen3-06b)")
    run_parser.add_argument("extra", nargs="*", help="Extra Hydra overrides (key=value)")

    # 'eval' subcommand: direct Hydra pass-through
    eval_parser = subparsers.add_parser("eval", help="Run evaluation passing raw Hydra overrides")
    eval_parser.add_argument("hydra_args", nargs="*", help="Hydra override arguments (e.g. mode=memory/hipporag k=10)")

    # 'tables' subcommand
    subparsers.add_parser("tables", help="Generate summary Markdown / LaTeX leaderboard tables")

    # 'metrics' subcommand
    subparsers.add_parser("metrics", help="Compute graph topology and connectivity metrics")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # If first argument looks like a hydra override (e.g., benchmark=locomo or mode=vector), default to eval
    if "=" in sys.argv[1]:
        cmd_args = sys.argv[1:]
        sys.argv = [sys.argv[0]] + cmd_args
        run_eval(cmd_args)
        return

    parsed, remaining = parser.parse_known_args()

    if parsed.subcommand == "run":
        parsed.extra = remaining
        run_simple(parsed)
    elif parsed.subcommand == "eval":
        sys.argv = [sys.argv[0]] + (parsed.hydra_args or remaining)
        run_eval(parsed.hydra_args or remaining)
    elif parsed.subcommand == "tables":
        run_tables(parsed)
    elif parsed.subcommand == "metrics":
        run_metrics(parsed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
