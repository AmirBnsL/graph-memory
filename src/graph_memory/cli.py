"""Unified CLI for graph-memory.

Allows running benchmark evaluations from declarative experiment profiles,
generating summary tables, and computing graph topology metrics.

Usage:
    # Run declarative experiment profiles (supports multiple)
    uv run graph-memory run smoke
    uv run graph-memory run h1_balanced
    uv run graph-memory run h1_balanced vector_grid

    # Run direct Hydra command
    uv run graph-memory eval benchmark=longmemeval mode=memory/hipporag k=10

    # Generate summary Markdown & LaTeX tables
    uv run graph-memory tables

    # Compute graph topology metrics
    uv run graph-memory metrics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml


def get_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "conf").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def run_eval(overrides: list[str]) -> None:
    """Run evaluation with Hydra overrides using initialize_config_dir and execute_config."""
    root = get_repo_root()
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    conf_dir = str(root / "conf")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from graph_memory.core import config as config_mod
    from graph_memory.pipeline.eval import execute_config

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=overrides)
        values = config_mod.to_plain(cfg)
        execute_config(values)


def load_profile(name_or_path: str, root: Path) -> dict:
    if name_or_path.endswith(".yaml") or name_or_path.endswith(".yml"):
        p = Path(name_or_path)
        if not p.is_file():
            p = root / "conf" / "experiment" / name_or_path
    else:
        p = root / "conf" / "experiment" / f"{name_or_path}.yaml"

    if not p.is_file():
        raise FileNotFoundError(f"Experiment profile not found at {p}")
    return yaml.safe_load(p.read_text())


def run_experiments(args: argparse.Namespace) -> None:
    root = get_repo_root()
    conf_dir = root / "conf"

    profiles = args.profiles
    if not profiles:
        print("ERROR: Please specify one or more experiment profiles to run (e.g. `graph-memory run smoke`)")
        available = [f.stem for f in (conf_dir / "experiment").glob("*.yaml")]
        if available:
            print(f"Available profiles: {', '.join(sorted(available))}")
        sys.exit(1)

    for prof_name in profiles:
        prof = load_profile(prof_name, root)
        exp_name = prof.get("name", prof_name)
        category = prof.get("category", "experiments")
        benchmark = prof.get("benchmark", "longmemeval")
        reader = prof.get("reader", "qwen25-7b")
        retriever = prof.get("retriever", "qwen3-06b")
        modes = prof.get("modes", ["vector"])
        if isinstance(modes, str):
            modes = [modes]
        k_list = prof.get("k", [10])
        if isinstance(k_list, int):
            k_list = [k_list]
        limits = prof.get("limits") or {}

        print(f"\n=======================================================")
        print(f"Executing Experiment Profile: {exp_name} ({category})")
        print(f"Benchmark: {benchmark} | Reader: {reader} | Retriever: {retriever}")
        print(f"Modes: {modes} | k: {k_list}")
        print(f"=======================================================\n")

        for mode in modes:
            for k in k_list:
                overrides = [
                    f"benchmark={benchmark}",
                    f"mode={mode}",
                    f"k={k}",
                    f"reader={reader}",
                    f"retriever={retriever}",
                    f"+category={category}",
                    f"+experiment={exp_name}",
                ]
                for limit_k, limit_v in limits.items():
                    if limit_v is not None:
                        overrides.append(f"limits.{limit_k}={limit_v}")
                if args.extra:
                    overrides.extend(args.extra)

                print(f"\n---> Condition: mode={mode} k={k}")
                run_eval(overrides)


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

    # 'run' subcommand: declarative profiles
    run_parser = subparsers.add_parser("run", help="Run one or more declarative experiment YAML profiles")
    run_parser.add_argument("profiles", nargs="*", help="Experiment profile name(s) in conf/experiment/ (e.g. smoke, h1_balanced)")
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
        run_eval(cmd_args)
        return

    parsed, remaining = parser.parse_known_args()

    if parsed.subcommand == "run":
        parsed.extra = remaining
        run_experiments(parsed)
    elif parsed.subcommand == "eval":
        run_eval(parsed.hydra_args or remaining)
    elif parsed.subcommand == "tables":
        run_tables(parsed)
    elif parsed.subcommand == "metrics":
        run_metrics(parsed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
