from __future__ import annotations

from skein.exporters.base import Exporter
from skein.core.trace import TraceEvent


class InMemoryExporter(Exporter):
    def __init__(self):
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)