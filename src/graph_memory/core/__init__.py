"""Core package: config, manifests, LLM and embedding interfaces."""

from graph_memory.core.config import group_name, to_plain, hash_dict
from graph_memory.core.manifest import build as build_manifest, write as write_manifest, git_state, config_hash
from graph_memory.core.llm import complete
from graph_memory.core.embed import embed, encode, encode_query

__all__ = [
    "group_name",
    "to_plain",
    "hash_dict",
    "build_manifest",
    "write_manifest",
    "git_state",
    "config_hash",
    "complete",
    "embed",
    "encode",
    "encode_query",
]
