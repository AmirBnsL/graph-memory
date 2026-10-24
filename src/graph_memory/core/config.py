"""Config helpers: plain-dict conversion + the hash that makes resume correct.

The resolved Hydra config is hashed into every run manifest
(``manifest.config.sha256``). Resume compares that hash -- plus the code commit
and the env hash -- against prior manifests, so editing anything in conf/
invalidates the affected results, on purpose.

This is the content-hash rule that workflow engines (Snakemake, DVC, Make)
apply to their inputs, implemented directly rather than by depending on one:
file *existence* is not enough to skip a run, because it cannot tell whether
the prompt or the retriever changed underneath it. Only the hashes can.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from omegaconf import OmegaConf


def _register_resolvers() -> None:
    """conf/config.yaml uses ${now:...} in its hydra.output block, which only
    Hydra's own runtime registers. Standalone loads (tests, rescore, anything
    that reads the config without going through @hydra.main) would raise
    UnsupportedInterpolationType -- so register it here, idempotently."""
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now", lambda fmt: datetime.now(timezone.utc).strftime(fmt), replace=True)


_register_resolvers()


def group_name(v) -> str:
    """A config GROUP resolves to a dict ({name: ...}); pull the name out.

    `mode` and `benchmark` come from defaults-list groups, so cfg.mode is
    {'name': 'vector'} -- not the string the run logic wants. Accepts either
    form so a group can be flattened to a plain value without breaking callers.
    """
    if isinstance(v, str):
        return v
    return (v or {}).get("name", "")


def to_plain(cfg) -> dict:
    """Resolve interpolations and return a JSON-serialisable plain dict.

    The `hydra` block is dropped before resolving: keys like `hydra.job.num`
    exist only inside Hydra's runtime, so resolving them from a standalone
    OmegaConf raises InterpolationKeyError. The block is Hydra machinery, never
    hashed (see manifest.config_hash) and never read by the harness.
    """
    _register_resolvers()
    container = OmegaConf.to_container(cfg, resolve=False)  # type: ignore[union-attr]
    container.pop("hydra", None)
    return OmegaConf.to_container(OmegaConf.create(container), resolve=True)  # type: ignore[return-value]


def hash_dict(d: dict) -> str:
    """Stable hash of a dict -- sort_keys so key order never changes the hash."""
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8", "ignore")).hexdigest()
