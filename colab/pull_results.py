#!/usr/bin/env python3
"""Pull results off the VM into the local repo -- the checkpoint that survives a
dead Colab session.

Runs LOCALLY: it shells out to the `colab` CLI, which the VM cannot call. This is
the missing half of the sweep. /content is wiped when a session ends, so a finished
k is not safe until its result and manifest are local. The sweep (colab/run_sweep.py)
makes itself resumable; this makes the work durable.

Policy, deliberately asymmetric:
  * result JSONs  -- immutable once written, downloaded only if ABSENT
  * logs          -- grow while a run is in flight, ALWAYS refreshed
  * INDEX.md      -- authored locally and never overwritten by the VM copy
  * eval.log      -- overwritten per run and large, skipped (it is a working log)

    python3 colab/pull_results.py                      # incremental
    python3 colab/pull_results.py --fresh              # re-download results too
    python3 colab/pull_results.py --session gm-dev \
        --remote-dir /content/graph-memory/results

Keep the repo current while a sweep runs by looping it locally:
    while sleep 300; do python3 colab/pull_results.py; done
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# Refresh every time -- they change while a run is in flight.
ALWAYS = {"sweep.log", "sweep.stdout.log"}
# Owned elsewhere -- never clobber.
NEVER = {"INDEX.md", "eval.log"}


def remote_ls(session: str, remote_dir: str) -> list[str]:
    """Filenames in a remote dir. `colab ls` prints bare names, one per line."""
    p = subprocess.run(["colab", "ls", "-s", session, remote_dir],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"colab ls failed: {p.stderr.strip() or p.stdout.strip()}")
    names = []
    for line in p.stdout.splitlines():
        n = line.strip()
        if not n or " " in n:                      # skip banner / box-drawing lines
            continue
        if n.endswith(".json") or n.endswith(".log") or n == "INDEX.md":
            names.append(n)
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="gm-dev")
    ap.add_argument("--remote-dir", default="/content/graph-memory/results")
    ap.add_argument("--local-dir", default="results")
    ap.add_argument("--fresh", action="store_true",
                    help="re-download results too, not just logs")
    args = ap.parse_args()

    local = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)

    pulled, skipped, failed = 0, 0, []
    for name in remote_ls(args.session, args.remote_dir):
        if name in NEVER:
            skipped += 1
            continue
        dest = local / name
        if dest.exists() and not (args.fresh or name in ALWAYS):
            skipped += 1
            continue
        p = subprocess.run(
            ["colab", "download", "-s", args.session,
             f"{args.remote_dir}/{name}", str(dest)],
            capture_output=True, text=True,
        )
        if p.returncode == 0:
            pulled += 1
            print(f"  pulled  {name}")
        else:
            failed.append(name)
            print(f"  FAILED  {name}: {(p.stderr or p.stdout).strip()[:200]}")

    print(f"\npulled={pulled}  skipped={skipped}  failed={len(failed)}  -> {local}/")
    for f in failed:
        print(f"  still on the VM only: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
