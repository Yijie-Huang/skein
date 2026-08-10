"""Exporter interface for agent trace outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from skein.core.trace import TraceEvent


class Exporter(ABC):
    """Base class for trace exporters."""

    @abstractmethod
    def emit(self, trace: TraceEvent) -> None:
        """Emit a trace event"""


class NoOpExporter(Exporter):
    """Default exporter that discards every event."""

    def emit(self, event: TraceEvent) -> None:
        return None
