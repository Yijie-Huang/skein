# Skein

A minimal, composable library for building multi-agent LLM systems.

Skein is a **framework layer**, not an application layer. It does not ship prompts, agents, or
domain logic — it gives you a small set of primitives for wiring agents together and, crucially,
for *seeing what they did*.

## Why

Multi-agent systems fail in ways single LLM calls do not: a step silently returns garbage, a
handoff loses context, a retry loop burns tokens, and by the time you notice, the only artifact
left is a wrong final answer. Skein's goal is to make multi-agent systems **visible and
controllable** — every node execution is a typed, exportable event, and the control flow lives in
ordinary Python you can read, diff, and test.

## Design principles

**Minimum kernel.** Five abstractions, and a strong bias against adding a sixth. Everything else
belongs in your code or in an optional extra.

**Observability first.** Tracing is not a plugin bolted on later — `TraceEvent` and `Exporter` are
part of the kernel. A node that runs without emitting an event is a bug.

**Explicit over magic.** No hidden global state, no implicit agent discovery, no string-templated
control flow. You register nodes and edges by hand, and the execution order is something you can
compute yourself.

**Code as guardrails.** Use deterministic Python to intercept and validate at the critical
junctures rather than delegating the whole workflow to an LLM. Typed Pydantic state means a node
that returns malformed output fails at the boundary, not three steps downstream.

## Install

Requires Python 3.10+.

```bash
pip install -e .                # core
pip install -e ".[langsmith]"   # + LangSmith tracing
pip install -e ".[mcp]"         # + MCP tool servers
pip install -e ".[dev]"         # + pytest, ruff, mypy
```

## Quick start

```python
import asyncio
from pydantic import Field
from skein import BaseState, Graph, JSONLExporter, Node, StateDelta


class ReviewState(BaseState):
    diff: str
    findings: list[str] = Field(default_factory=list)
    verdict: str | None = None


# A node reads the whole state and returns only the fields it changed.
async def lint(state: ReviewState) -> StateDelta | None:
    if "TODO" in state.diff:
        return {"findings": [*state.findings, "leftover TODO"]}
    return None


# A node can also be a class, when it needs configuration or dependencies.
class DecideNode(Node[ReviewState]):
    def __init__(self) -> None:
        super().__init__("decide")

    async def run(self, state: ReviewState) -> StateDelta:
        return {"verdict": "request-changes" if state.findings else "approve"}


graph = (
    Graph[ReviewState]("review", JSONLExporter("traces/review.jsonl"))
    .add_node("lint", lint)
    .add_node("decide", DecideNode())
    .add_edge("lint", "decide")
    .set_entry_point("lint")
)

state = asyncio.run(graph.run(ReviewState(trace_id="run-1", diff="+ # TODO: fix")))
print(state.verdict)   # request-changes
```

Every run leaves a trace behind:

```jsonl
{"trace_id":"run-1","node_name":"lint","status":"completed","started_at":"...","finished_at":"...","token_usage":{},"error":null}
{"trace_id":"run-1","node_name":"decide","status":"completed","started_at":"...","finished_at":"...","token_usage":{},"error":null}
```

## Core abstractions

| | What it is |
|---|---|
| `BaseState` | The frozen Pydantic model threaded through the graph. Subclass it to declare your domain fields; `trace_id`, `created_at`, `current_node`, and `status` come for free. |
| `Node` | One unit of work: `async run(state) -> StateDelta \| None`. It reads the whole state and returns only the fields it changed (or `None` for "nothing changed"). Either subclass `Node[YourState]` or pass a bare async function (`NodeFunction`) — the graph accepts both. |
| `Graph` | Registers nodes and edges, resolves execution order, runs the workflow, and emits a trace event per node. |
| `TraceEvent` | The typed record of a single node execution: status, start/finish time, token usage, error. |
| `Exporter` | Where trace events go. `NoOpExporter` (default), `InMemoryExporter`, and `JSONLExporter` ship in the box; implement `emit()` for anything else. |

### Execution semantics

- **Nodes return deltas, not state.** A node reads the whole state and returns a `dict` of the
  fields it changed — or `None` to change nothing. The graph merges the delta and re-validates the
  result, so every step produces a new state. `BaseState` is **frozen**: assigning to a field
  raises, which keeps "what changed" and "what the trace recorded" from drifting apart. (Freezing
  blocks rebinding a field, not mutating a list in place — build a new value instead.)
  Writing a field the state does not declare raises `SkeinStateError` instead of silently vanishing.
- Because a delta *replaces* the fields it names, appending means building the new value from the
  old one: `return {"findings": [*state.findings, "..."]}` rather than `state.findings.append(...)`.
- `current_node` and `status` are owned by the graph, but a node can still end a run early by
  returning `{"status": GraphStatus.FAILED}`.
- Nodes run in **topological order**, derived from the edges. If several nodes have no incoming
  edge, you must disambiguate with `set_entry_point()`; cycles are rejected up front.
- Execution is **sequential** — v0 runs a linear pass over the sorted order. Branching, parallel
  fan-out, and loops are not in the kernel yet (see [Roadmap](#roadmap)); an agent loop today lives
  *inside* a single node, as the example below shows.
- A node that raises **fails fast**: the run stops, `state.status` becomes `failed`, and the error
  is captured on that node's `TraceEvent`.
- `max_steps` caps how many nodes may execute, so a runaway workflow stops on your terms.
- `Node` and `Graph` are **generic in the state type**. `Graph[ReviewState].run()` returns a
  `ReviewState`, not a `BaseState`, and `class DecideNode(Node[ReviewState])` may narrow `run`'s
  argument without violating the base signature. The type parameter is optional — omit it and you
  get the same runtime behaviour with looser types.
- Trace events are emitted in a `finally` block — **success or failure, every executed node is
  recorded**. An exporter that itself throws is logged and swallowed, never masking the real error.

## Observability

### Local exporters

`JSONLExporter` appends one JSON object per node execution; `InMemoryExporter` keeps them in a list
for assertions in tests. Both are trivial to replace — `Exporter` has a single method.

### LangSmith

Install the extra (`pip install -e ".[langsmith]"`), set `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY`, plus optionally `LANGSMITH_PROJECT`. Each `Graph.run()` becomes a root `chain`
run and each node a child `tool` run, carrying a summary of the state and the fields that changed —
not the full state blob.

Export is **opt-in**: installing the dependency is not enough, the flag has to be set too. With it
off — or with `langsmith` absent altogether — no client is built and nothing leaves the process;
local exporters keep working either way.

Skein reads the nearest `.env` on import (without overriding variables you already exported in the
shell), so keeping LangSmith settings in a project `.env` is enough.

To pull runs back down for offline inspection:

```bash
python tests/export_langsmith_runs.py --trace-id <trace-id> --output runs.jsonl
```

## Example: alarm investigation

[`src/skein/examples/alarm_investigation/`](src/skein/examples/alarm_investigation/) is the v0
end-to-end flow — an AIOps triage workflow as a three-node linear DAG:

```
TriageNode  ──▶  InvestigationNode  ──▶  SummaryNode
 (rules)         (ReAct + MCP tool)        (rules)
```

It deliberately mixes deterministic and model-driven steps, which is the point: triage and
summarization are ordinary Python with typed results, while the investigation node runs a bounded
ReAct loop against Claude with a `get_service_cpu` tool served over MCP. The model's answer is
parsed straight into a Pydantic `InvestigationResult`, so a malformed response is caught at the
node boundary.

```bash
python -m skein.examples.alarm_investigation.graph                                   # run the workflow
pip install -e ".[mcp]" && python -m skein.examples.alarm_investigation.mcp_server   # serve the tool
```

The investigation node needs `ANTHROPIC_API_KEY`. Without it the run stops at that node and the
trace says why, rather than failing somewhere downstream:

```
Run failed: ANTHROPIC_API_KEY is not set. The investigation node calls Claude directly — ...
Node: triage,        Status: TaskStatus.COMPLETED
Node: investigation, Status: TaskStatus.FAILED
```

The `mcp` extra is optional: without it the tool call falls back to the same synthetic metric
lookup in-process, so that half of the demo still runs.

See also [`node_function_example.py`](src/skein/examples/node_function_example.py) for the smallest
possible graph.

## Roadmap

| | Focus |
|---|---|
| **v0** | Skeleton plus one working end-to-end flow: five abstractions, tracing, AIOps example. |
| **v1** | Context and memory — what a node sees, and what persists across runs. |
| **v2** | Eval harness — turn traces into regression tests and scored runs. |
| **v3** | Reliability — retries, timeouts, richer control flow, failure isolation. |
| **v∞** | Self-evolution — distill traces back into runnable workflows. Prototyped end-to-end in a production system; the mechanics are known, not speculative. |

## Scope

Skein intentionally does **not**:

- compete with LangChain on breadth of integrations;
- ship an application or content layer (prompts, agent personas, RAG pipelines);
- bind itself to a single domain — AIOps is the first example, not the product.

### Why not LangGraph?

LangGraph optimizes for breadth of control flow. Skein optimizes for a kernel small enough to read in one sitting, with tracing in the kernel rather than as an integration.

## Development

```bash
pip install -e ".[dev]"
pytest                    # kernel, exporters, and the example's no-API-call paths
ruff check src tests
mypy src/skein
```

## License

MIT — see [LICENSE](LICENSE).
