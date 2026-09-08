"""Graph execution engine for agentic workflows."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic

from ..exporters.base import Exporter, NoOpExporter
from ..logging_config import get_logger
from .node import Node, NodeFunction, StateDelta, normalize_writes
from .state import BaseState, GraphStatus, S, SkeinStateError
from .trace import TaskStatus, TraceEvent, langsmith_tracing_enabled

logger = get_logger(__name__)

_STATE_SYSTEM_FIELDS = {"trace_id", "created_at", "current_node", "status"}

# Writable by any node without being declared: ending a run early is control flow
# available to every node, not something it should have to list as a write.
_ALWAYS_WRITABLE = frozenset({"status"})

class SkeinGraphError(Exception):
    """Raised when a graph definition is invalid, before any node runs."""


@dataclass(frozen=True)
class ScheduledGroup:
    """A set of nodes that may run together, and where it sits in the plan."""

    wave: int
    group: int
    names: list[str]


@dataclass
class _NodeOutcome:
    """What one node produced: its delta, or the error that stands in for it."""

    name: str
    delta: StateDelta
    error: BaseException | None
    event: TraceEvent


def _serialize_trace_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _serialize_trace_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_trace_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_trace_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _summarize_state(state: BaseState) -> dict[str, Any]:
    state_payload = _serialize_trace_payload(state)
    summary: dict[str, Any] = {
        "trace_id": state_payload.get("trace_id"),
        "current_node": state_payload.get("current_node"),
        "status": state_payload.get("status"),
    }
    domain_state = {
        key: value
        for key, value in state_payload.items()
        if key not in _STATE_SYSTEM_FIELDS and value is not None
    }
    if domain_state:
        summary["state"] = domain_state
    return summary


async def _invoke_node(
    name: str,
    node: Node[S] | NodeFunction[S],
    state: S,
    declared: frozenset[str] | None = None,
) -> StateDelta:
    """Run either a Node instance or a bare async NodeFunction."""
    fields = type(state).model_fields.keys()
    if declared is not None:
        missing = declared - fields
        if missing:
            raise SkeinStateError(
                f"node '{name}' declares writes to field(s) {sorted(missing)}, "
                f"which {type(state).__name__} does not have"
            )

    delta = await (node.run(state) if isinstance(node, Node) else node(state))
    if delta is None:
        return {}
    if not isinstance(delta, dict):
        raise SkeinStateError(
            f"node '{name}' must return a dict of changed fields or None, "
            f"got {type(delta).__name__}"
        )
    unknown = delta.keys() - fields
    if unknown:
        raise SkeinStateError(
            f"node '{name}' wrote unknown field(s): {sorted(unknown)}"
        )
    if declared is not None:
        undeclared = delta.keys() - declared - _ALWAYS_WRITABLE
        if undeclared:
            raise SkeinStateError(
                f"node '{name}' wrote undeclared field(s): {sorted(undeclared)}; "
                f"it declares writes={sorted(declared)}"
            )
    return delta


def _apply(state: S, delta: StateDelta) -> S:
    """Return a new state with the delta merged in, re-validating the result."""
    if not delta:
        return state
    return type(state).model_validate({**dict(state), **delta})


def _apply_system(state: S, delta: StateDelta) -> S:
    """Apply the graph's own bookkeeping writes (``current_node``, ``status``).

    These values are produced by the runtime itself rather than by a node, so they
    skip validation — but so do any model validators, which is why this must never
    be used for a node's delta.
    """
    return state.model_copy(update=delta)


def _load_langsmith_trace() -> Any:
    try:
        langsmith_module = importlib.import_module("langsmith")
    except ImportError:
        return None
    return getattr(langsmith_module, "trace", None)


def _load_langsmith_tracing_context() -> Any:
    try:
        run_helpers_module = importlib.import_module("langsmith.run_helpers")
    except ImportError:
        return None
    return getattr(run_helpers_module, "tracing_context", None)


def _load_langsmith_client() -> Any:
    try:
        langsmith_module = importlib.import_module("langsmith")
    except ImportError:
        return None
    client_cls = getattr(langsmith_module, "Client", None)
    if client_cls is None:
        return None
    try:
        return client_cls()
    except Exception:
        logger.warning("Failed to initialize LangSmith client", exc_info=True)
        return None


_LANGSMITH_TRACING_CONTEXT = _load_langsmith_tracing_context()
_UNSET = object()
_LANGSMITH_CLIENT: Any = _UNSET


def _get_langsmith_client() -> Any:
    """Return the process-wide LangSmith client, creating it at most once."""
    global _LANGSMITH_CLIENT
    if _LANGSMITH_CLIENT is _UNSET:
        _LANGSMITH_CLIENT = _load_langsmith_client()
    return _LANGSMITH_CLIENT


@contextmanager
def _maybe_langsmith_trace(
    name: str,
    run_type: str,
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    if not langsmith_tracing_enabled():
        yield None
        return

    trace_fn = _load_langsmith_trace()
    tracing_context_fn = _LANGSMITH_TRACING_CONTEXT
    if trace_fn is None or tracing_context_fn is None:
        yield None
        return

    client = _get_langsmith_client()
    project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    with tracing_context_fn(
        enabled=True,
        client=client,
        project_name=project_name,
    ):
        with trace_fn(
            name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata,
            client=client,
            project_name=project_name,
        ) as run:
            yield run


async def _flush_langsmith() -> None:
    if not langsmith_tracing_enabled():
        return

    client = _get_langsmith_client()
    if client is None:
        return

    flush = getattr(client, "flush", None)
    if flush is None:
        return

    try:
        result = flush()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("Failed to flush LangSmith traces", exc_info=True)


class Graph(Generic[S]):
    """Directed acyclic graph for composing nodes into workflows."""

    def __init__(self, name: str, exporter: Exporter | None = None):
        self.name = name
        self.nodes: dict[str, Node[S] | NodeFunction[S]] = {}
        self.node_writes: dict[str, frozenset[str] | None] = {}
        self.edges: dict[str, list[str]] = {}
        self.entry_point: str | None = None
        self.exporter: Exporter = exporter or NoOpExporter()
        self._waves: list[list[str]] | None = None
        self._plan: list[ScheduledGroup] | None = None

    def add_node(
        self,
        name: str,
        node: Node[S] | NodeFunction[S],
        *,
        writes: Iterable[str] | None = None,
    ) -> Graph[S]:
        """Add a node to the graph — either a Node instance or an async NodeFunction.

        ``writes`` declares the fields this node may change, for the cases a `Node`
        cannot cover: a bare async function has nowhere to hang the declaration, and
        a node you did not write may need one supplied from outside. It takes
        precedence over the node's own ``writes``.
        """
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists in the graph.")
        if not isinstance(node, Node) and not callable(node):
            raise TypeError(f"Node {name} must be a Node instance or an async callable.")
        declared = normalize_writes(writes)
        self.nodes[name] = node
        self.node_writes[name] = declared if declared is not None else getattr(node, "writes", None)
        self.edges[name] = []
        self._invalidate()
        return self

    def add_edge(self, from_node: str, to_node: str) -> Graph[S]:
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must be added to the graph before connecting.")
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)
        self._invalidate()
        return self

    def set_entry_point(self, name: str) -> Graph[S]:
        if name not in self.nodes:
            raise ValueError(f"Node {name} not in graph.")
        self.entry_point = name
        self._invalidate()
        return self

    def _invalidate(self) -> None:
        """Editing the graph drops the plan; the next run rebuilds it."""
        self._waves = None
        self._plan = None

    def build(self) -> Graph[S]:
        """Resolve the execution plan and reject a graph that cannot run correctly.

        Nodes are first grouped into waves: everything in a wave has had all its
        predecessors run, so nothing within one depends on another. Each wave is
        then split into groups that are safe to run together — see `_split_wave`.
        Two nodes in a group writing the same field is refused here rather than
        discovered later as a flaky result.

        Calling this is optional — `run` builds on demand — and repeating it is
        harmless.
        """
        waves = self._topological_wave_sort()
        plan: list[ScheduledGroup] = []
        for wave_index, wave in enumerate(waves):
            groups = self._split_wave(wave)
            if len(groups) > 1:
                logger.warning(
                    "graph %r: wave %d is split into %d groups because node(s) %s "
                    "declare no writes, so conflicts with them cannot be ruled out",
                    self.name,
                    wave_index,
                    len(groups),
                    sorted(name for name in wave if self.node_writes[name] is None),
                )
            for group_index, group in enumerate(groups):
                plan.append(ScheduledGroup(wave=wave_index, group=group_index, names=group))
        self._waves = waves
        self._plan = plan
        return self

    def _split_wave(self, wave: list[str]) -> list[list[str]]:
        """Split one topological wave into groups that are safe to run together.

        A node that declares its writes joins the group being accumulated, unless
        it collides with something already claimed there. A node that declares
        nothing cannot be cleared of a collision, so it runs alone — which also
        makes it a barrier: nodes on either side of it are ordered, and so no
        longer conflict with each other.
        """
        groups: list[list[str]] = []
        current: list[str] = []
        owner: dict[str, str] = {}  # field -> node that claimed it

        for name in wave:
            writes = self.node_writes[name]
            if writes is None:
                if current:
                    groups.append(current)
                groups.append([name])  # runs alone
                current, owner = [], {}
                continue

            claimed = writes - _ALWAYS_WRITABLE
            clash = owner.keys() & claimed
            if clash:
                field = sorted(clash)[0]
                raise SkeinGraphError(
                    f"'{field}' is written by both '{owner[field]}' and '{name}', "
                    f"which run in the same wave. Add an edge to order them, or "
                    f"split the field."
                )
            current.append(name)
            owner.update(dict.fromkeys(claimed, name))

        if current:
            groups.append(current)
        return groups

    @property
    def waves(self) -> list[list[str]]:
        """The topological waves — dependency depth, before any splitting."""
        if self._waves is None:
            self.build()
        assert self._waves is not None
        return self._waves

    @property
    def plan(self) -> list[ScheduledGroup]:
        """The execution plan: the groups that actually run, in order."""
        if self._plan is None:
            self.build()
        assert self._plan is not None
        return self._plan

    async def run(self, initial_state: S, max_steps: int | None = None) -> S:
        trace_id = initial_state.trace_id
        logger.info("Starting graph execution: %s", trace_id)
        state = initial_state
        plan = self.plan
        step_limit: float = float("inf") if max_steps is None else max_steps
        with _maybe_langsmith_trace(
            self.name,
            run_type="chain",
            inputs={
                "initial_state": _summarize_state(initial_state),
                "max_steps": step_limit,
            },
            metadata={"graph": self.name, "trace_id": trace_id},
        ) as graph_run:
            steps = 0
            halted = False
            for scheduled in plan:
                if steps + len(scheduled.names) > step_limit:
                    halted = True
                    break

                state = _apply_system(state, {"current_node": ",".join(scheduled.names)})
                outcomes = await self._run_group(scheduled, state, trace_id, steps)
                steps += len(scheduled.names)

                failure = next((o.error for o in outcomes if o.error is not None), None)
                if failure is None:
                    merged: StateDelta = {}
                    for outcome in outcomes:
                        merged.update(outcome.delta)
                    try:
                        state = _apply(state, merged)
                    except Exception as exc:
                        failure = exc
                        for outcome in outcomes:
                            outcome.event.status = TaskStatus.FAILED
                            outcome.event.error = (
                                f"wave {scheduled.wave} group {scheduled.group} "
                                f"produced an invalid state: {exc}"
                            )
                    else:
                        if state.status == GraphStatus.FAILED:
                            for outcome in outcomes:
                                if outcome.delta.get("status") == GraphStatus.FAILED:
                                    outcome.event.status = TaskStatus.FAILED

                self._emit(outcomes)

                if failure is not None:
                    # All or nothing: a group that lost a node contributes no delta.
                    state = _apply_system(state, {"status": GraphStatus.FAILED})
                    halted = True
                    break
                if state.status == GraphStatus.FAILED:
                    halted = True
                    break

            # A node that already failed the run keeps that status; otherwise the
            # run ended either short (halted) or clean.
            if state.status != GraphStatus.FAILED:
                state = _apply_system(
                    state,
                    {"status": GraphStatus.FAILED if halted else GraphStatus.COMPLETED},
                )

            if graph_run is not None:
                graph_run.end(
                    outputs={
                        "final_state": _summarize_state(state),
                        "status": state.status,
                        "completed_steps": steps,
                    }
                )
            await _flush_langsmith()
        return state

    async def _run_group(
        self, scheduled: ScheduledGroup, state: S, trace_id: str, step: int
    ) -> list[_NodeOutcome]:
        """Run one group concurrently. Every node reads the same state.

        Nodes do not mutate what they are given, so one shared reference is enough
        — no copy per node. Ordering between groups is the plan's job, not this one's.
        """
        results = await asyncio.gather(
            *(self._run_one(name, state, trace_id, step, scheduled) for name in scheduled.names),
            return_exceptions=True,
        )
        return [
            self._as_outcome(name, result, trace_id, scheduled)
            for name, result in zip(scheduled.names, results, strict=True)
        ]

    async def _run_one(
        self,
        name: str,
        state: S,
        trace_id: str,
        step: int,
        scheduled: ScheduledGroup,
    ) -> _NodeOutcome:
        """Run one node, capturing a failure instead of raising it at its siblings."""
        event = TraceEvent(
            trace_id=trace_id,
            node_name=name,
            wave=scheduled.wave,
            group=scheduled.group,
            status=TaskStatus.IN_PROGRESS,
        )
        event.started_at = datetime.now(timezone.utc)
        try:
            with _maybe_langsmith_trace(
                name,
                run_type="tool",
                inputs={"state": _summarize_state(state)},
                metadata={
                    "graph": self.name,
                    "node": name,
                    "step": step,
                    "wave": scheduled.wave,
                    "group": scheduled.group,
                    "trace_id": trace_id,
                },
            ) as node_run:
                delta = await _invoke_node(name, self.nodes[name], state, self.node_writes[name])
                if node_run is not None:
                    node_run.end(outputs={"delta": _serialize_trace_payload(delta)})
        except Exception as exc:
            event.status = TaskStatus.FAILED
            event.error = str(exc)
            event.finished_at = datetime.now(timezone.utc)
            return _NodeOutcome(name=name, delta={}, error=exc, event=event)

        event.status = TaskStatus.COMPLETED
        event.finished_at = datetime.now(timezone.utc)
        return _NodeOutcome(name=name, delta=delta, error=None, event=event)

    def _as_outcome(
        self, name: str, result: Any, trace_id: str, scheduled: ScheduledGroup
    ) -> _NodeOutcome:
        """Normalise a gather result; a bare exception means `_run_one` itself broke."""
        if isinstance(result, _NodeOutcome):
            return result
        now = datetime.now(timezone.utc)
        event = TraceEvent(
            trace_id=trace_id,
            node_name=name,
            wave=scheduled.wave,
            group=scheduled.group,
            status=TaskStatus.FAILED,
            started_at=now,
            finished_at=now,
            error=str(result),
        )
        return _NodeOutcome(name=name, delta={}, error=result, event=event)

    def _emit(self, outcomes: list[_NodeOutcome]) -> None:
        for outcome in outcomes:
            try:
                self.exporter.emit(outcome.event)
            except Exception:
                logger.warning("exporter failed for node %s", outcome.name, exc_info=True)

    def _topological_wave_sort(self) -> list[list[str]]:
        in_degrees: dict[str, int] = dict.fromkeys(self.nodes, 0)
        for to_nodes in self.edges.values():
            for to_node in to_nodes:
                in_degrees[to_node] += 1

        wave = [name for name in self.nodes if in_degrees[name] == 0]
        if not wave:
            raise ValueError("Graph has a cycle: every node has an incoming edge.")
        if self.entry_point is not None and self.entry_point not in wave:
            raise ValueError(
                f"Entry point {self.entry_point!r} has incoming edges; it cannot start the graph."
            )

        waves: list[list[str]] = []
        emitted = 0
        while wave:
            waves.append(wave)
            emitted += len(wave)
            next_wave = []
            for node in wave:
                for to_node in self.edges.get(node, []):
                    in_degrees[to_node] -= 1
                    if in_degrees[to_node] == 0:
                        next_wave.append(to_node)
            wave = next_wave

        if emitted != len(self.nodes):
            raise ValueError("Graph has a cycle.")
        return waves
