"""Tests for the Graph execution engine."""

from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import Field, ValidationError

from skein import BaseState, Graph, InMemoryExporter, Node, NoOpExporter, StateDelta
from skein.core.graph import SkeinGraphError
from skein.core.state import GraphStatus, SkeinStateError
from skein.core.trace import TaskStatus


class TrackedState(BaseState):
    visited: list[str] = Field(default_factory=list)
    marker: str | None = None


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


async def set_marker(state: TrackedState) -> StateDelta:
    return {"marker": "set"}


def test_multiple_roots_share_the_first_wave():
    """Nothing to disambiguate: independent roots simply run in the same wave."""
    graph = Graph("forked")
    graph.add_node("a", make_fn("a"), writes=["visited"])
    graph.add_node("b", set_marker, writes=["marker"])

    assert graph.waves == [["a", "b"]]

    result = run_graph(graph, TrackedState(trace_id="t-forked"))

    assert result.visited == ["a"]
    assert result.marker == "set"
    assert result.status == GraphStatus.COMPLETED


def test_undeclared_nodes_are_split_into_their_own_groups(caplog):
    """A node that declares nothing cannot be cleared of a conflict, so it runs alone.

    Run together, both nodes would read the same snapshot and rewrite `visited`,
    and merging would keep only the last. Split apart, the second sees the first's
    result and the writes chain.
    """
    exporter = InMemoryExporter()
    graph = Graph("undeclared", exporter)
    graph.add_node("a", make_fn("a"))
    graph.add_node("b", make_fn("b"))

    with caplog.at_level(logging.WARNING):
        result = run_graph(graph, TrackedState(trace_id="t-undeclared"))

    assert result.visited == ["a", "b"]
    assert "is split into 2 groups" in caplog.text
    # Same wave, different groups — visibly split apart rather than run together.
    assert [(e.node_name, e.wave, e.group) for e in exporter.events] == [
        ("a", 0, 0),
        ("b", 0, 1),
    ]


def test_the_trace_records_which_wave_and_group_a_node_ran_in():
    """Whether nodes really ran together is recorded, not left to be inferred."""
    exporter = InMemoryExporter()
    graph = Graph("observable", exporter)
    graph.add_node("root", make_fn("root"), writes=["visited"])
    graph.add_node("mark", set_marker, writes=["marker"])
    graph.add_node("tail", set_marker, writes=["marker"])
    graph.add_edge("root", "tail")
    graph.add_edge("mark", "tail")

    run_graph(graph, TrackedState(trace_id="t-observable"))

    recorded = [(e.node_name, e.wave, e.group) for e in exporter.events]
    assert recorded == [
        ("root", 0, 0),
        ("mark", 0, 0),   # same wave and group -> really ran together
        ("tail", 1, 0),
    ]


def test_a_split_wave_keeps_the_groups_that_already_succeeded():
    """All-or-nothing is per group, the unit that actually runs together.

    Splitting orders these nodes, so an earlier group's result surviving a later
    failure is the same behaviour a linear graph has always had.
    """

    async def boom(state: TrackedState) -> StateDelta:
        raise RuntimeError("second exploded")

    async def never_runs(state: TrackedState) -> StateDelta:
        raise AssertionError("the run should have stopped before this")

    exporter = InMemoryExporter()
    graph = Graph("split-atomic", exporter)
    graph.add_node("first", make_fn("first"))     # undeclared → each runs alone
    graph.add_node("second", boom)
    graph.add_node("third", never_runs)

    result = run_graph(graph, TrackedState(trace_id="t-split-atomic"))

    assert result.status == GraphStatus.FAILED
    assert result.visited == ["first"]            # its group completed before the failure
    assert [e.node_name for e in exporter.events] == ["first", "second"]
    assert "second exploded" in exporter.events[-1].error


def test_a_failing_group_discards_its_own_members():
    """Within one group it is still all or nothing."""

    async def boom(state: TrackedState) -> StateDelta:
        raise RuntimeError("sibling exploded")

    exporter = InMemoryExporter()
    graph = Graph("group-atomic", exporter)
    graph.add_node("ok", make_fn("ok"), writes=["visited"])
    graph.add_node("boom", boom, writes=["marker"])

    result = run_graph(graph, TrackedState(trace_id="t-group-atomic"))

    assert result.status == GraphStatus.FAILED
    assert result.visited == []                   # the sibling's delta went with it
    assert {e.wave for e in exporter.events} == {0}
    assert {e.group for e in exporter.events} == {0}


def test_entry_point_with_incoming_edges_is_rejected():
    graph = Graph("bad-entry")
    graph.add_node("a", make_fn("a"))
    graph.add_node("b", make_fn("b"))
    graph.add_edge("a", "b")
    graph.set_entry_point("b")

    with pytest.raises(ValueError, match="cannot start the graph"):
        graph.build()


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


# --- build ------------------------------------------------------------------


def test_waves_group_nodes_by_dependency_depth():
    graph = Graph("diamond")
    for name in ("root", "left", "right", "join"):
        graph.add_node(name, make_fn(name))
    graph.add_edge("root", "left")
    graph.add_edge("root", "right")
    graph.add_edge("left", "join")
    graph.add_edge("right", "join")

    assert graph.build().waves == [["root"], ["left", "right"], ["join"]]


def test_two_nodes_in_one_wave_writing_the_same_field_is_refused():
    graph = Graph("conflict")
    graph.add_node("left", make_fn("left"), writes=["visited"])
    graph.add_node("right", make_fn("right"), writes=["visited"])

    with pytest.raises(SkeinGraphError, match="written by both 'left' and 'right'"):
        graph.build()


def test_the_same_field_across_different_waves_is_fine():
    """Sequenced writes have a defined winner, so they are not a conflict."""
    graph = Graph("sequenced")
    graph.add_node("first", make_fn("first"), writes=["visited"])
    graph.add_node("second", make_fn("second"), writes=["visited"])
    graph.add_edge("first", "second")

    assert graph.build().waves == [["first"], ["second"]]


def test_undeclared_nodes_take_part_in_no_conflict():
    """Without a declaration there is nothing to compare, so the graph still builds."""
    graph = Graph("opaque")
    graph.add_node("left", make_fn("left"))
    graph.add_node("right", make_fn("right"), writes=["visited"])

    assert graph.build().waves == [["left", "right"]]


def test_status_is_not_a_conflict():
    graph = Graph("both-can-fail")
    graph.add_node("left", make_fn("left"), writes=["status"])
    graph.add_node("right", make_fn("right"), writes=["status"])

    graph.build()


def test_build_is_idempotent_and_chainable():
    graph = Graph("repeat")
    graph.add_node("a", make_fn("a"))

    assert graph.build().build() is graph
    assert graph.waves == [["a"]]


def test_editing_the_graph_after_building_rebuilds_it():
    graph = Graph("mutating")
    graph.add_node("a", make_fn("a"))
    assert graph.waves == [["a"]]

    graph.add_node("b", make_fn("b"))
    graph.add_edge("a", "b")

    assert graph.waves == [["a"], ["b"]]


def test_run_builds_on_demand_and_surfaces_conflicts():
    graph = Graph("unbuilt", InMemoryExporter())
    graph.add_node("left", make_fn("left"), writes=["visited"])
    graph.add_node("right", make_fn("right"), writes=["visited"])

    with pytest.raises(SkeinGraphError):
        run_graph(graph, TrackedState(trace_id="t-unbuilt"))


# --- parallel waves ---------------------------------------------------------


def test_a_wave_really_runs_concurrently():
    """Each node waits for the other, so this only finishes if they overlap."""
    reached_a, reached_b = asyncio.Event(), asyncio.Event()

    async def a(state: TrackedState) -> StateDelta:
        reached_a.set()
        await asyncio.wait_for(reached_b.wait(), timeout=2)
        return {"visited": ["a"]}

    async def b(state: TrackedState) -> StateDelta:
        reached_b.set()
        await asyncio.wait_for(reached_a.wait(), timeout=2)
        return {"marker": "b"}

    graph = Graph("concurrent")
    graph.add_node("a", a, writes=["visited"])
    graph.add_node("b", b, writes=["marker"])

    result = run_graph(graph, TrackedState(trace_id="t-concurrent"))

    assert result.visited == ["a"]
    assert result.marker == "b"


def test_a_failing_node_discards_the_whole_wave():
    """All or nothing: a sibling's successful delta does not reach the state."""

    async def boom(state: TrackedState) -> StateDelta:
        raise RuntimeError("node exploded")

    exporter = InMemoryExporter()
    graph = Graph("atomic", exporter)
    graph.add_node("ok", make_fn("ok"), writes=["visited"])
    graph.add_node("boom", boom, writes=["marker"])
    graph.add_node("after", set_marker, writes=["marker"])
    graph.add_edge("ok", "after")
    graph.add_edge("boom", "after")

    result = run_graph(graph, TrackedState(trace_id="t-atomic"))

    assert result.status == GraphStatus.FAILED
    assert result.visited == []          # the sibling succeeded, its delta was dropped
    assert result.marker is None         # the next wave never ran

    events = {event.node_name: event for event in exporter.events}
    assert set(events) == {"ok", "boom"}                  # both siblings still traced
    assert events["ok"].status == TaskStatus.COMPLETED    # it did run, it just did not count
    assert events["boom"].status == TaskStatus.FAILED
    assert "node exploded" in events["boom"].error


def test_every_sibling_finishes_even_when_one_fails():
    """A failure does not cancel the rest of its wave."""
    finished: list[str] = []

    async def slow(state: TrackedState) -> StateDelta:
        await asyncio.sleep(0.05)
        finished.append("slow")
        return {"visited": ["slow"]}

    async def fails_fast(state: TrackedState) -> StateDelta:
        raise RuntimeError("immediate")

    graph = Graph("siblings", InMemoryExporter())
    graph.add_node("slow", slow, writes=["visited"])
    graph.add_node("fails", fails_fast, writes=["marker"])

    result = run_graph(graph, TrackedState(trace_id="t-siblings"))

    assert finished == ["slow"]
    assert result.status == GraphStatus.FAILED


def test_several_failures_in_one_wave_are_all_traced():
    async def boom(state: TrackedState) -> StateDelta:
        raise RuntimeError("boom")

    exporter = InMemoryExporter()
    graph = Graph("many-failures", exporter)
    graph.add_node("one", boom, writes=["visited"])
    graph.add_node("two", boom, writes=["marker"])

    result = run_graph(graph, TrackedState(trace_id="t-many"))

    assert result.status == GraphStatus.FAILED
    assert [event.status for event in exporter.events] == [TaskStatus.FAILED, TaskStatus.FAILED]


def test_a_wave_is_applied_in_one_validation_pass():
    """An invalid combined result fails the wave rather than half-applying it."""

    async def good(state: TrackedState) -> StateDelta:
        return {"visited": ["good"]}

    async def bad(state: TrackedState) -> StateDelta:
        return {"marker": ["not-a-string"]}

    exporter = InMemoryExporter()
    graph = Graph("invalid-merge", exporter)
    graph.add_node("good", good, writes=["visited"])
    graph.add_node("bad", bad, writes=["marker"])

    result = run_graph(graph, TrackedState(trace_id="t-merge"))

    assert result.status == GraphStatus.FAILED
    assert result.visited == []
    assert all("produced an invalid state" in event.error for event in exporter.events)


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
