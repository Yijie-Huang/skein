from __future__ import annotations

from pathlib import Path

from skein.core.trace import TraceEvent
from skein.exporters.base import Exporter


class JSONLExporter(Exporter):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
            