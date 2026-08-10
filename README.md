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
from skein import BaseState, Graph, JSONLExporter, Node


class ReviewState(BaseState):
    diff: str
    findings: list[str] = []
    verdict: str | None = None


# A node can be a plain async function...
async def lint(state: ReviewState) -> ReviewState:
    if "TODO" in state.diff:
        state.findings.append("leftover TODO")
    return state


# ...or a class, when it needs configuration or dependencies.
class DecideNode(Node):
    def __init__(self) -> None:
        super().__init__("decide")

    async def run(self, state: ReviewState) -> ReviewState:
        state.verdict = "request-changes" if state.findings else "approve"
        return state


graph = (
    Graph("review", JSONLExporter("traces/review.jsonl"))
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
| `BaseState` | The Pydantic model threaded through the graph. Subclass it to declare your domain fields; `trace_id`, `created_at`, `current_node`, and `status` come for free. |
| `Node` | One unit of work: `async run(state) -> state`. Either subclass `Node` or pass a bare async function (`NodeFunction`) — the graph accepts both. |
| `Graph` | Registers nodes and edges, resolves execution order, runs the workflow, and emits a trace event per node. |
| `TraceEvent` | The typed record of a single node execution: status, start/finish time, token usage, error. |
| `Exporter` | Where trace events go. `NoOpExporter` (default), `InMemoryExporter`, and `JSONLExporter` ship in the box; implement `emit()` for anything else. |

### Execution semantics

- Nodes run in **topological order**, derived from the edges. If several nodes have no incoming
  edge, you must disambiguate with `set_entry_point()`; cycles are rejected up front.
- Execution is **sequential** — v0 runs a linear pass over the sorted order. Branching, parallel
  fan-out, and loops are not in the kernel yet (see [Roadmap](#roadmap)); an agent loop today lives
  *inside* a single node, as the example below shows.
- A node that raises **fails fast**: the run stops, `state.status` becomes `failed`, and the error
  is captured on that node's `TraceEvent`.
- `max_steps` caps how many nodes may execute, so a runaway workflow stops on your terms.
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

Requires `ANTHROPIC_API_KEY` for the investigation node. Without the `mcp` extra, the tool call
falls back to the same synthetic metric lookup in-process, so the demo still runs.

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
pytest                    # graph execution + exporter tests
ruff check src tests
```

## License

MIT — see [LICENSE](LICENSE).
