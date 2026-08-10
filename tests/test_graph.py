"""Tests for the Graph execution engine."""

from __future__ import annotations

import asyncio

import pytest

from skein import BaseState, Graph, InMemoryExporter, Node, NoOpExporter
from skein.core.state import GraphStatus
from skein.core.trace import TaskStatus


class RecordingNode(Node):
    """Node that appends its own name to a shared list on the state."""

    async def run(self, state: TrackedState) -> TrackedState:
        state.visited.append(self.name)
        return state


class TrackedState(BaseState):
    visited: list[str] = []


def make_fn(name: str):
    async def node_fn(state: TrackedState) -> TrackedState:
        state.visited.append(name)
        return state

    return node_fn


def run_graph(graph: Graph, state: BaseState) -> BaseState:
    return asyncio.run(graph.run(state))


def test_node_function_and_node_class_can_be_mixed():
    graph = Graph("mixed", InMemoryExporter())
    graph.add_node("fn", make_fn("fn"))
    graph.add_node("cls", RecordingNode("cls"))
    graph.add_edge("fn", "cls")
    graph.set_entry_point("fn")

    result = run_graph(graph, TrackedState(trace_id="t-mixed"))

    assert result.visited == ["fn", "cls"]
    assert result.status == GraphStatus.COMPLETED


def test_default_exporter_is_usable_without_arguments():
    graph = Graph("default-exporter")
    assert isinstance(graph.exporter, NoOpExporter)

    graph.add_node("only", make_fn("only"))
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-default"))
    assert result.status == GraphStatus.COMPLETED


def test_add_node_rejects_non_callable():
    graph = Graph("bad-node")
    with pytest.raises(TypeError):
        graph.add_node("nope", 42)


def test_builder_methods_support_chaining():
    graph = (
        Graph("chained", InMemoryExporter())
        .add_node("a", make_fn("a"))
        .add_node("b", make_fn("b"))
        .add_edge("a", "b")
        .set_entry_point("a")
    )

    result = run_graph(graph, TrackedState(trace_id="t-chain"))
    assert result.visited == ["a", "b"]


def test_failing_node_stops_execution_and_marks_graph_failed():
    async def boom(state: TrackedState) -> TrackedState:
        raise RuntimeError("node exploded")

    exporter = InMemoryExporter()
    graph = Graph("failing", exporter)
    graph.add_node("ok", make_fn("ok"))
    graph.add_node("boom", boom)
    graph.add_node("never", make_fn("never"))
    graph.add_edge("ok", "boom")
    graph.add_edge("boom", "never")
    graph.set_entry_point("ok")

    result = run_graph(graph, TrackedState(trace_id="t-fail"))

    assert result.status == GraphStatus.FAILED
    assert result.visited == ["ok"]
    assert [event.node_name for event in exporter.events] == ["ok", "boom"]
    assert exporter.events[-1].status == TaskStatus.FAILED
    assert "node exploded" in exporter.events[-1].error


def test_max_steps_halts_early_and_marks_graph_failed():
    graph = Graph("capped", InMemoryExporter())
    graph.add_node("a", make_fn("a"))
    graph.add_node("b", make_fn("b"))
    graph.add_edge("a", "b")
    graph.set_entry_point("a")

    result = asyncio.run(graph.run(TrackedState(trace_id="t-cap"), max_steps=1))

    assert result.visited == ["a"]
    assert result.status == GraphStatus.FAILED


def test_cycle_is_rejected():
    graph = Graph("cyclic")
    graph.add_node("a", make_fn("a"))
    graph.add_node("b", make_fn("b"))
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")

    with pytest.raises(ValueError, match="cycle"):
        run_graph(graph, TrackedState(trace_id="t-cycle"))


def test_multiple_entry_points_require_explicit_entry_point():
    graph = Graph("forked")
    graph.add_node("a", make_fn("a"))
    graph.add_node("b", make_fn("b"))

    with pytest.raises(ValueError, match="multiple entry points"):
        run_graph(graph, TrackedState(trace_id="t-forked"))

    graph.set_entry_point("b")
    result = run_graph(graph, TrackedState(trace_id="t-forked-2"))
    assert result.visited == ["b", "a"]


def test_edges_require_registered_nodes():
    graph = Graph("dangling")
    graph.add_node("a", make_fn("a"))

    with pytest.raises(ValueError):
        graph.add_edge("a", "missing")


def test_duplicate_node_name_is_rejected():
    graph = Graph("dup")
    graph.add_node("a", make_fn("a"))

    with pytest.raises(ValueError):
        graph.add_node("a", make_fn("a2"))
