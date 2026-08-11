"""Tests for the Graph execution engine."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import Field, ValidationError

from skein import BaseState, Graph, InMemoryExporter, Node, NoOpExporter, StateDelta
from skein.core.state import GraphStatus, SkeinStateError
from skein.core.trace import TaskStatus


class TrackedState(BaseState):
    visited: list[str] = Field(default_factory=list)


class RecordingNode(Node[TrackedState]):
    """Node that records its own name by returning a delta, never mutating state."""

    async def run(self, state: TrackedState) -> StateDelta:
        return {"visited": [*state.visited, self.name]}


def make_fn(name: str):
    async def node_fn(state: TrackedState) -> StateDelta:
        return {"visited": [*state.visited, name]}

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
    async def boom(state: TrackedState) -> StateDelta:
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


# --- delta contract ---------------------------------------------------------


def single_node_graph(node) -> Graph:
    graph = Graph("delta", InMemoryExporter())
    graph.add_node("only", node)
    graph.set_entry_point("only")
    return graph


def test_returning_none_leaves_state_unchanged():
    async def touches_nothing(state: TrackedState) -> None:
        return None

    result = run_graph(single_node_graph(touches_nothing), TrackedState(trace_id="t-none"))

    assert result.visited == []
    assert result.status == GraphStatus.COMPLETED


def test_delta_only_replaces_the_fields_it_names():
    async def writes_one_field(state: TrackedState) -> StateDelta:
        return {"visited": ["only-this"]}

    state = TrackedState(trace_id="t-partial", visited=["pre-existing"])
    result = run_graph(single_node_graph(writes_one_field), state)

    assert result.visited == ["only-this"]
    assert result.trace_id == "t-partial"          # untouched fields survive
    assert result.created_at == state.created_at


def test_unknown_field_in_delta_is_rejected():
    async def writes_garbage(state: TrackedState) -> StateDelta:
        return {"nope": 1, "also_nope": 2}

    exporter = InMemoryExporter()
    graph = Graph("bad-delta", exporter)
    graph.add_node("only", writes_garbage)
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-unknown"))

    assert result.status == GraphStatus.FAILED
    error = exporter.events[-1].error
    assert "unknown field(s)" in error
    assert "'also_nope', 'nope'" in error


def test_returning_a_state_object_gives_a_clear_error():
    """The most likely migration mistake: returning state instead of a delta."""

    async def returns_state(state: TrackedState) -> TrackedState:
        return state

    exporter = InMemoryExporter()
    graph = Graph("legacy-node", exporter)
    graph.add_node("only", returns_state)
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-legacy"))

    assert result.status == GraphStatus.FAILED
    assert "must return a dict of changed fields" in exporter.events[-1].error


def test_delta_values_are_validated():
    async def writes_wrong_type(state: TrackedState) -> StateDelta:
        return {"visited": "not-a-list"}

    result = run_graph(single_node_graph(writes_wrong_type), TrackedState(trace_id="t-type"))
    assert result.status == GraphStatus.FAILED


def test_node_can_mark_the_run_failed_through_the_delta():
    exporter = InMemoryExporter()
    graph = Graph("self-fail", exporter)

    async def gives_up(state: TrackedState) -> StateDelta:
        return {"status": GraphStatus.FAILED}

    graph.add_node("first", gives_up)
    graph.add_node("second", make_fn("second"))
    graph.add_edge("first", "second")
    graph.set_entry_point("first")

    result = run_graph(graph, TrackedState(trace_id="t-selffail"))

    assert result.status == GraphStatus.FAILED
    assert result.visited == []                                   # second never ran
    assert exporter.events[-1].status == TaskStatus.FAILED


def test_the_caller_state_object_is_never_mutated():
    graph = Graph("immutable", InMemoryExporter())
    graph.add_node("a", make_fn("a"))
    graph.add_node("b", make_fn("b"))
    graph.add_edge("a", "b")
    graph.set_entry_point("a")

    state = TrackedState(trace_id="t-immutable")
    result = run_graph(graph, state)

    assert result.visited == ["a", "b"]
    assert state.visited == []
    assert state.current_node is None
    assert state.status == GraphStatus.RUNNING
    assert result is not state


# --- declared writes --------------------------------------------------------


class DeclaringNode(Node[TrackedState]):
    """Writes exactly what it declares."""

    def __init__(self, name: str, writes, delta: StateDelta):
        super().__init__(name, writes=writes)
        self._delta = delta

    async def run(self, state: TrackedState) -> StateDelta:
        return self._delta


def test_declared_write_is_allowed():
    node = DeclaringNode("ok", ["visited"], {"visited": ["a"]})

    result = run_graph(single_node_graph(node), TrackedState(trace_id="t-declared"))

    assert result.visited == ["a"]
    assert result.status == GraphStatus.COMPLETED


def test_undeclared_write_is_rejected():
    node = DeclaringNode("sneaky", ["visited"], {"visited": ["a"], "current_node": "elsewhere"})

    exporter = InMemoryExporter()
    graph = Graph("undeclared", exporter)
    graph.add_node("only", node)
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-undeclared"))

    assert result.status == GraphStatus.FAILED
    error = exporter.events[-1].error
    assert "undeclared field(s): ['current_node']" in error
    assert "writes=['visited']" in error


def test_declaring_a_field_the_state_lacks_is_rejected():
    """Catches a typo in `writes` even when the node never writes that field."""
    node = DeclaringNode("typo", ["visted"], {})

    exporter = InMemoryExporter()
    graph = Graph("bad-declaration", exporter)
    graph.add_node("only", node)
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-typo"))

    assert result.status == GraphStatus.FAILED
    assert "declares writes to field(s) ['visted']" in exporter.events[-1].error


def test_empty_writes_forbids_every_write():
    node = DeclaringNode("read-only", [], {"visited": ["a"]})

    exporter = InMemoryExporter()
    graph = Graph("read-only", exporter)
    graph.add_node("only", node)
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-readonly"))

    assert result.status == GraphStatus.FAILED
    assert "undeclared field(s): ['visited']" in exporter.events[-1].error


def test_add_node_can_declare_writes_for_a_bare_function():
    """A plain async function has nowhere to hang the declaration."""
    exporter = InMemoryExporter()
    graph = Graph("fn-writes", exporter)
    graph.add_node("only", make_fn("a"), writes=["current_node"])
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-fnwrites"))

    assert result.status == GraphStatus.FAILED
    assert "undeclared field(s): ['visited']" in exporter.events[-1].error


def test_add_node_writes_overrides_the_nodes_own_declaration():
    """Wrapping someone else's node, the graph gets the last word."""
    node = DeclaringNode("theirs", ["current_node"], {"visited": ["a"]})

    graph = Graph("override", InMemoryExporter())
    graph.add_node("only", node, writes=["visited"])
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-override"))

    assert result.visited == ["a"]                      # allowed by the override
    assert node.writes == frozenset({"current_node"})   # the node itself is untouched


def test_add_node_keeps_the_nodes_declaration_when_no_override_is_given():
    node = DeclaringNode("theirs", ["visited"], {"visited": ["a"]})

    graph = Graph("inherit", InMemoryExporter())
    graph.add_node("only", node)

    assert graph.node_writes["only"] == frozenset({"visited"})


def test_a_bare_string_is_rejected_as_a_write_set():
    graph = Graph("stringly")

    with pytest.raises(TypeError, match='did you mean \\["verdict"\\]'):
        graph.add_node("only", make_fn("a"), writes="verdict")

    with pytest.raises(TypeError, match="not the string"):
        RecordingNode("n", writes="verdict")


def test_status_may_be_written_without_declaring_it():
    """Ending a run early is control flow open to every node."""
    node = DeclaringNode("bail", ["visited"], {"status": GraphStatus.FAILED})

    exporter = InMemoryExporter()
    graph = Graph("bail", exporter)
    graph.add_node("only", node)
    graph.add_node("next", make_fn("next"))
    graph.add_edge("only", "next")
    graph.set_entry_point("only")

    result = run_graph(graph, TrackedState(trace_id="t-bail"))

    assert result.status == GraphStatus.FAILED
    assert result.visited == []
    assert exporter.events[-1].error is None      # a declared-writes violation, not an error


def test_writes_is_optional():
    """Nodes that do not declare writes keep working unchecked."""
    node = DeclaringNode("undeclared", None, {"visited": ["anything"]})
    assert node.writes is None

    result = run_graph(single_node_graph(node), TrackedState(trace_id="t-optional"))
    assert result.visited == ["anything"]


def test_repr_shows_declared_writes():
    assert repr(DeclaringNode("d", ["visited"], {})) == (
        "DeclaringNode(name='d', writes=['visited'])"
    )
    assert repr(DeclaringNode("d", None, {})) == "DeclaringNode(name='d')"


def test_state_is_frozen():
    """Mutating state in place is a contract violation, so it must not be possible."""
    state = TrackedState(trace_id="t-frozen")

    with pytest.raises(ValidationError):
        state.current_node = "sneaky"
    with pytest.raises(ValidationError):
        state.visited = ["sneaky"]


def test_invalid_delta_raises_skein_state_error_directly():
    from skein.core.graph import _invoke_node

    async def writes_garbage(state: TrackedState) -> StateDelta:
        return {"nope": 1}

    with pytest.raises(SkeinStateError, match="unknown field"):
        asyncio.run(_invoke_node("only", writes_garbage, TrackedState(trace_id="t-direct")))
