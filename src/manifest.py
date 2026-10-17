"""Run manifest: the record that makes a number auditable.

Every result written by eval.py gets a sibling manifest answering the only
question a reproduction cares about: *what exactly produced this number?*

Recorded per run:
    code       git commit, branch, dirty flag
    data       pinned datasets + sha256 (provenance/datasets.json, embedded whole)
    model      provider, model id, base_url, what served it, at what precision
    prompts    sha256 of the system prompt and the answer template
    env        sha256 of the frozen package list, so "same environment" is checkable
    hardware   python / torch / CUDA / GPU name
    result     the result file this describes

Deliberately NOT recorded: API keys. base_url + model id are enough to
re-identify a served model without writing a credential into results/.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import answer
import llm

ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = ROOT / "provenance" / "datasets.json"


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance must never crash a run
        return ""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


# Output directories are excluded from the dirty check. results/ is deliberately
# committed, so the moment a run writes its own output the tree looks dirty --
# every manifest would claim dirty=True, which makes the field useless.
_OUTPUT_DIRS = ("results", "cache", "graphs")

# The code-state snapshot is taken the FIRST time this module is imported, which
# -- because eval.py imports manifest at module load -- is before the run has
# written anything. That ordering is the whole point.
_GIT: dict | None = None


def git_state() -> dict:
    global _GIT
    if _GIT is None:
        spec = ["--", ".", *(f":(exclude){d}" for d in _OUTPUT_DIRS)]
        _GIT = {
            "commit": _sh(["git", "rev-parse", "HEAD"]) or "unknown",
            "branch": _sh(["git", "branch", "--show-current"]) or "unknown",
            "dirty": bool(_sh(["git", "status", "--porcelain", *spec])),
            "dirty_excludes": list(_OUTPUT_DIRS),
        }
    return _GIT


def datasets() -> dict:
    if not PROVENANCE.exists():
        return {"status": "MISSING", "expected_at": str(PROVENANCE)}
    return json.loads(PROVENANCE.read_text())


def prompts() -> dict:
    return {
        "system_sha256": _sha(answer.SYSTEM),
        "template_sha256": _sha(answer.PROMPT_TEMPLATE),
    }


def env_freeze() -> dict:
    out = _sh(["uv", "pip", "freeze", "--system"]) or _sh(
        [sys.executable, "-m", "pip", "freeze"]
    )
    return {"sha256": _sha(out), "n_packages": len(out.splitlines())}


def hardware() -> dict:
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        info["gpu_count"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)
    return info


def build(mode: str, k: int, n_conv: int | None, n_qa: int | None,
          result_path: Path, extra: dict | None = None) -> dict:
    cfg = llm.resolve_config()
    m: dict = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": {"mode": mode, "k": k, "n_conv": n_conv, "n_qa": n_qa},
        "result_file": result_path.name,
        "code": git_state(),
        "data": datasets(),
        "model": {
            "provider": cfg["provider"],
            "model": cfg["model"],
            "base_url": cfg["base_url"],
        },
        "serving": {
            "served_by": os.environ.get("LLM_SERVED_BY", "unknown"),
            "quantization": os.environ.get("LLM_QUANTIZATION", "unknown"),
        },
        "prompts": prompts(),
        "env": env_freeze(),
        "hardware": hardware(),
    }
    if extra:
        m.update(extra)
    return m


def write(m: dict, result_path: Path) -> Path:
    """Write <result>.manifest.json next to the result it describes."""
    p = result_path.with_suffix(".manifest.json")
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    return p
