"""Base Memory Interface.

All memory backends (flat vector, HippoRAG, temporal graph) implement this identical
interface to guarantee fair, controlled ablations for H1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import networkx as nx

from extract import MemoryRecord


@dataclass
class RetrievedItem:
    id: str
    text: str
    score: float
    turn_id: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """Abstract base class for all memory backends."""

    @abstractmethod
    def add_records(self, records: list[MemoryRecord]) -> None:
        """Ingest extracted memory records."""
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int, timestamp: str | None = None) -> list[RetrievedItem]:
        """Retrieve top-k relevant items given a natural language query."""
        pass

    @abstractmethod
    def export_graph(self) -> nx.Graph | None:
        """Export the internal graph representation, or None if non-graph."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Reset internal memory state."""
        pass
