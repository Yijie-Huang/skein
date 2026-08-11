from __future__ import annotations

from skein.core.trace import TraceEvent
from skein.exporters.base import Exporter


class InMemoryExporter(Exporter):
    def __init__(self):
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)