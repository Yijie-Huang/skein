"""Graph execution engine for agentic workflows."""

from __future__ import annotations

import importlib
import inspect
import os
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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
        return self

    def add_edge(self, from_node: str, to_node: str) -> Graph[S]:
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must be added to the graph before connecting.")
        self.edges[from_node].append(to_node)
        return self
    
    def set_entry_point(self, name: str) -> Graph[S]:
        if name not in self.nodes:
            raise ValueError(f"Node {name} not in graph.")
        self.entry_point = name
        return self
    
    async def run(self, initial_state: S, max_steps: int | None = None) -> S:
        trace_id = initial_state.trace_id
        logger.info("Starting graph execution: %s", trace_id)
        state = initial_state
        execution_order = self._topological_sort()
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
            step, execution_idx = 0, 0
            while step < step_limit and execution_idx < len(execution_order):
                running_node_name = execution_order[execution_idx]
                state = _apply_system(state, {"current_node": running_node_name})
                trace_event = TraceEvent(
                    trace_id=trace_id,
                    node_name=running_node_name,
                    status=TaskStatus.IN_PROGRESS,
                )
                trace_event.started_at = datetime.now(timezone.utc)
                running_node = self.nodes[running_node_name]
                node_inputs = {"state": _summarize_state(state)}
                try:
                    with _maybe_langsmith_trace(
                        running_node_name,
                        run_type="tool",
                        inputs=node_inputs,
                        metadata={
                            "graph": self.name,
                            "node": running_node_name,
                            "step": step,
                            "trace_id": trace_id,
                        },
                    ) as node_run:
                        delta = await _invoke_node(
                            running_node_name,
                            running_node,
                            state,
                            self.node_writes[running_node_name],
                        )
                        state = _apply(state, delta)
                        if node_run is not None:
                            node_run.end(outputs={"delta": _serialize_trace_payload(delta)})
                    if state.status == GraphStatus.FAILED:
                        trace_event.status = TaskStatus.FAILED
                        break
                    else:
                        trace_event.status = TaskStatus.COMPLETED
                except Exception as e:
                    trace_event.status = TaskStatus.FAILED
                    trace_event.error = str(e)
                    state = _apply_system(state, {"status": GraphStatus.FAILED})
                    break
                finally:
                    step += 1
                    execution_idx += 1
                    trace_event.finished_at = datetime.now(timezone.utc)
                    try:
                        self.exporter.emit(trace_event)
                    except Exception:
                        logger.warning(
                            "exporter failed for node %s", running_node_name, exc_info=True
                        )

            if state.status == GraphStatus.FAILED:
                pass
            elif execution_idx == len(execution_order):
                state = _apply_system(state, {"status": GraphStatus.COMPLETED})
            else:
                state = _apply_system(state, {"status": GraphStatus.FAILED})

            if graph_run is not None:
                graph_run.end(
                    outputs={
                        "final_state": _summarize_state(state),
                        "status": state.status,
                        "completed_steps": step,
                    }
                )
            await _flush_langsmith()
        return state
        
    
    def _topological_sort(self) -> list[str]:
        # calculate in-degrees for each node
        node_to_in_degrees: dict[str, int] = {}
        for to_nodes in self.edges.values():
            for to_node in to_nodes:
                node_to_in_degrees.setdefault(to_node, 0)
                node_to_in_degrees[to_node] += 1

        # queue of nodes with no incoming edges
        # queue = [node for node in self.nodes if node_to_in_degrees.get(node, 0) == 0]
        queue = deque(node for node in self.nodes if node_to_in_degrees.get(node, 0) == 0)
        if len(queue) == 0:
            raise ValueError("Graph has a cycle or no entry point.")
        if len(queue) > 1:
            if not self.entry_point:
                raise ValueError(
                    "Graph has multiple entry points. "
                    "Please set a single entry point using set_entry_point()."
                )
            else:
                if self.entry_point not in queue:
                    raise ValueError(
                        f"Specified entry point {self.entry_point} is not a valid starting node."
                    )
                # reorder queue to start with the specified entry point
                queue.remove(self.entry_point)
                queue.appendleft(self.entry_point)
        output_order = []
        while queue:
            # node = queue.pop(0)  # O(n) operation
            node = queue.popleft() # O(1)
            output_order.append(node)
            if node in self.edges:
                for to_node in self.edges[node]:
                    node_to_in_degrees[to_node] -= 1
                    if node_to_in_degrees[to_node] == 0:
                        queue.append(to_node)

        if len(output_order) != len(self.nodes):
            raise ValueError("Graph has a cycle.")
        return output_order
    

