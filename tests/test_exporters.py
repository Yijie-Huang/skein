"""Tests for trace exporters."""

from __future__ import annotations

import json

from skein import InMemoryExporter, JSONLExporter, NoOpExporter
from skein.core.trace import TaskStatus, TraceEvent


def make_event(node_name: str = "node-a") -> TraceEvent:
    return TraceEvent(trace_id="t-1", node_name=node_name, status=TaskStatus.COMPLETED)


def test_in_memory_exporter_collects_events():
    exporter = InMemoryExporter()
    exporter.emit(make_event("a"))
    exporter.emit(make_event("b"))

    assert [event.node_name for event in exporter.events] == ["a", "b"]


def test_no_op_exporter_discards_events():
    assert NoOpExporter().emit(make_event()) is None


def test_jsonl_exporter_appends_one_json_object_per_event(tmp_path):
    path = tmp_path / "nested" / "trace.jsonl"
    exporter = JSONLExporter(path)
    exporter.emit(make_event("a"))
    exporter.emit(make_event("b"))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["node_name"] for line in lines] == ["a", "b"]
    assert json.loads(lines[0])["status"] == TaskStatus.COMPLETED.value
