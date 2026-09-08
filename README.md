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


# A node can also be a class, and can declare up front which fields it writes.
class DecideNode(Node[ReviewState]):
    def __init__(self) -> None:
        super().__init__("decide", writes=["verdict"])

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
{"trace_id":"run-1","node_name":"lint","wave":0,"group":0,"status":"completed","started_at":"...","finished_at":"...","token_usage":{},"error":null}
{"trace_id":"run-1","node_name":"decide","wave":1,"group":0,"status":"completed","started_at":"...","finished_at":"...","token_usage":{},"error":null}
```

## Core abstractions

| | What it is |
|---|---|
| `BaseState` | The frozen Pydantic model threaded through the graph. Subclass it to declare your domain fields; `trace_id`, `created_at`, `current_node`, and `status` come for free. |
| `Node` | One unit of work: `async run(state) -> StateDelta \| None`. It reads the whole state and returns only the fields it changed (or `None` for "nothing changed"), and can declare that write set as `writes=[...]`. Either subclass `Node[YourState]` or pass a bare async function (`NodeFunction`) — the graph accepts both. |
| `Graph` | Registers nodes and edges, resolves execution order, runs the workflow, and emits a trace event per node. |
| `TraceEvent` | The typed record of a single node execution: status, start/finish time, token usage, error, and the wave and group it ran in. |
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
- A `Node` may declare its write set — `super().__init__("decide", writes=["verdict"])` — and the
  graph holds it to that: returning anything outside the set raises, as does declaring a field the
  state does not have (so a typo surfaces even if the node never writes it). `status` is exempt, so
  any node can still end a run early. `writes` is optional; omit it and the node is unchecked. It
  doubles as documentation: you can read which fields a workflow touches, and where, without
  opening a single `run()`.
- `add_node(name, node, writes=[...])` declares the same thing from the outside, for the cases the
  constructor cannot reach: a bare async function has nowhere to hang the declaration, and a node
  you did not write may need one supplied for it. It wins over the node's own `writes`.
- `current_node` and `status` are owned by the graph, but a node can still end a run early by
  returning `{"status": GraphStatus.FAILED}`. Its group still finishes and is applied; the next one
  does not start.
- Nodes run in dependency order, in **waves** that execute concurrently — see below. Cycles are
  rejected up front, and independent roots simply share the first wave.
- `max_steps` caps how many nodes may execute, so a runaway workflow stops on your terms. A group
  that would exceed the cap is not started at all, rather than half-run.
- `Node` and `Graph` are **generic in the state type**. `Graph[ReviewState].run()` returns a
  `ReviewState`, not a `BaseState`, and `class DecideNode(Node[ReviewState])` may narrow `run`'s
  argument without violating the base signature. The type parameter is optional — omit it and you
  get the same runtime behaviour with looser types.
- **Every executed node is recorded**, success or failure. An exporter that itself throws is logged
  and swallowed, never masking the real error.

### Waves, groups, and what happens when one node fails

`build()` groups nodes into **waves** by dependency depth: everything in a wave has had all of its
predecessors run, so nothing in it depends on anything else in it. Each wave is then split into
**groups** that are safe to run together, and the nodes of a group run **concurrently**, on
`asyncio.gather`, against one shared snapshot of the state — no copy per node, because nodes do not
mutate what they are given.

```
build() → [["fetch"], ["lint", "score"], ["report"]]
           wave 0      wave 1 (parallel)   wave 2
```

Because there is no ordering within a group, two of its nodes writing the same field would make the
winner an accident of scheduling. `build()` refuses such a graph up front:

```
SkeinGraphError: 'findings' is written by both 'lint' and 'style', which run in the
                 same wave. Add an edge to order them, or split the field.
```

This check can only see **declared** writes. A node that declares none cannot be cleared of a
conflict, so instead of costing its whole wave its concurrency, `build()` splits the wave into
**groups**: the undeclared node runs alone, and everything else still runs together.

```
wave 0: [d1(log)] [opaque] [d2(log), d3(m)]
        group 0   group 1  group 2
```

Running alone also makes an undeclared node a **barrier**: `d1` and `d2` both write `log` here
without conflicting, because the split has already put them in a defined order.

`build()` logs a warning naming the nodes, and each event records the wave and group it ran in, so
the split is never silent — nodes sharing a `(wave, group)` really ran together:

```jsonl
{"node_name":"d1","wave":0,"group":0,"status":"completed", ...}
{"node_name":"opaque","wave":0,"group":1,"status":"completed", ...}
{"node_name":"d2","wave":0,"group":2,"status":"completed", ...}
{"node_name":"d3","wave":0,"group":2,"status":"completed", ...}
```

Declaring `writes` is what keeps a node in the shared group — and turns a possible conflict from a
split into a refusal.

A group is **all or nothing**:

- Siblings are never cancelled: when a node raises, the rest of its group still runs to completion.
- If any node in the group failed, **none** of that group's deltas are applied. The run stops and
  `status` becomes `failed`. "The whole group happened, or none of it did" is far easier to reason
  about — and to replay — than a state holding some fraction of it.
- Groups earlier in the plan keep their results: they are ordered relative to the failure, exactly
  as an earlier node in a linear graph has always been.
- Every node that ran still emits a `TraceEvent`; the failed one carries the error. A sibling that
  succeeded is recorded as `completed` even though its delta was dropped: it really did run, and
  the run-level `failed` status is what tells you the group was discarded.
- Deltas within a successful group are merged and validated in a single pass, so an invalid
  *combination* fails the group rather than half-applying it.

## Observability

### Local exporters

`JSONLExporter` appends one JSON object per node execution; `InMemoryExporter` keeps them in a list
for assertions in tests. Both are trivial to replace — `Exporter` has a single method.

### LangSmith

Install the extra (`pip install -e ".[langsmith]"`), set `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY`, plus optionally `LANGSMITH_PROJECT`. Each `Graph.run()` becomes a root `chain`
run and each node a child `tool` run, taking a summary of the state as its input and reporting the
node's delta as its output — what changed, not the whole state on every step.

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
