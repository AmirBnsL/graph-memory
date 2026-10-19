"""Run manifest: the record that makes a number auditable.

Every result written by eval.py gets a sibling manifest answering the only
question a reproduction cares about: *what exactly produced this number?*

Recorded per run:
    code       git commit, branch, dirty flag
    data       pinned datasets + sha256 (provenance/datasets.json, embedded whole)
    model      provider, model id, base_url
    serving    what served it, at what precision, and its MEASURED context window
               (provenance/serving.json, embedded whole) -- not env-var dependent
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
import config as config_mod
import llm

ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = ROOT / "provenance" / "datasets.json"
SERVING = ROOT / "provenance" / "serving.json"


def config_hash(cfg: dict) -> str:
    """The hash resume compares. Excluded keys would make two different configs
    collide, so everything except Hydra's own machinery is included."""
    return config_mod.hash_dict({k: v for k, v in cfg.items() if k != "hydra"})


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
            "code_commit": _sh(["git", "log", "-1", "--format=%H", *spec]) or "unknown",
            "branch": _sh(["git", "branch", "--show-current"]) or "unknown",
            "dirty": bool(_sh(["git", "status", "--porcelain", *spec])),
            "dirty_excludes": list(_OUTPUT_DIRS),
        }
    return _GIT


def datasets() -> dict:
    if not PROVENANCE.exists():
        return {"status": "MISSING", "expected_at": str(PROVENANCE)}
    return json.loads(PROVENANCE.read_text())


def serving_pin() -> dict:
    """Measured serving-stack facts (context window, precision, server version).

    Embedded whole rather than read from env vars: the effective context window is
    a constraint on which k values are valid, and it must be recorded even for a
    run launched without LLM_SERVED_BY / LLM_QUANTIZATION set.
    """
    if not SERVING.exists():
        return {"status": "MISSING", "expected_at": str(SERVING)}
    return json.loads(SERVING.read_text())


def serving() -> dict:
    """What served the model, from the PIN, not from env vars.

    The pin is the authority: reading LLM_SERVED_BY / LLM_QUANTIZATION instead is
    what produced a sweep's worth of manifests reading "unknown" -- the sweep was
    launched without those exported, and a fact that depends on the shell is not
    a fact. The pin is embedded whole regardless, so the measured context window
    (a constraint on which k values are even valid) is always on record. Env is
    used only as a fallback when provenance/serving.json is absent.
    """
    pin = serving_pin()
    if "answering_model" in pin:
        am = pin["answering_model"]
        return {
            "served_by": am.get("served_by", "unknown"),
            "quantization": am.get("quantization", "unknown"),
            "pinned": pin,
        }
    return {
        "served_by": os.environ.get("LLM_SERVED_BY", "unknown"),
        "quantization": os.environ.get("LLM_QUANTIZATION", "unknown"),
        "pinned": pin,
    }


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


def build(cfg: dict, result_path: Path, extra: dict | None = None) -> dict:
    """Build a manifest from the RESOLVED experiment config.

    `cfg` is the plain dict form of the Hydra config (src/config.py::to_plain).
    It is hashed into the manifest, which is what makes resume a content-hash
    decision rather than a filename-pattern guess: a run is skippable only when
    a prior manifest matches on config + code + env.
    """
    limits = cfg.get("limits") or {}
    provider, model = llm.resolve_identity(cfg.get("reader"))
    cfg_values = {k: v for k, v in cfg.items() if k != "hydra"}
    m: dict = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": {
            "mode": config_mod.group_name(cfg.get("mode")),
            "benchmark": config_mod.group_name(cfg.get("benchmark")),
            "k": cfg.get("k"),
            "n_conv": limits.get("n_conv"),
            "n_qa": limits.get("n_qa"),
        },
        "result_file": result_path.name,
        "code": git_state(),
        "config": {"sha256": config_hash(cfg_values), "values": cfg_values},
        "data": datasets(),
        "model": {
            "provider": provider,
            "model": model,
            "base_url": cfg.get("base_url") or (llm.resolve_config((cfg.get("reader") or {})).get("base_url")),
        },
        "serving": serving(),
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
