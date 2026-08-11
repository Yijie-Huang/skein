"""Example demonstrating NodeFunction usage."""

from __future__ import annotations

import asyncio

from pydantic import Field

from skein.core import BaseState, Graph, Node, StateDelta
from skein.exporters import InMemoryExporter


class PipelineState(BaseState):
    """Nodes read the whole state but only write the fields they own."""

    raw: str | None = None
    processed: str | None = None
    checks: list[str] = Field(default_factory=list)


# Style 1: a plain async function (a NodeFunction).
async def fetch_data(state: PipelineState) -> StateDelta:
    """Fetch data from source."""
    print(f"[fetch_data] Processing trace {state.trace_id}")
    await asyncio.sleep(0.1)  # stand-in for real I/O
    # A node returns only the fields it changed; the graph merges them into the state.
    return {"raw": "alpha,beta"}


async def process_data(state: PipelineState) -> StateDelta:
    """Process the fetched data."""
    print(f"[process_data] Trace {state.trace_id}, raw: {state.raw}")
    await asyncio.sleep(0.1)
    return {"processed": (state.raw or "").upper()}


# Style 2: a Node subclass, for logic that needs configuration or dependencies.
class ValidateNode(Node[PipelineState]):
    """Validate processed data."""

    def __init__(self):
        super().__init__("validate")

    async def run(self, state: PipelineState) -> StateDelta:
        print(f"[validate] Checking processed value: {state.processed}")
        await asyncio.sleep(0.1)
        # A delta replaces the field, so append by building a new list from the old one.
        return {"checks": [*state.checks, "non-empty"]}


async def main():
    graph = Graph[PipelineState]("node_function_example", InMemoryExporter())

    # Both styles can be mixed in one graph.
    graph.add_node("fetch", fetch_data)
    graph.add_node("process", process_data)
    graph.add_node("validate", ValidateNode())

    graph.add_edge("fetch", "process")
    graph.add_edge("process", "validate")
    graph.set_entry_point("fetch")

    state = PipelineState(trace_id="example-001")

    print("Starting workflow...")
    result = await graph.run(state)
    print(f"Completed! Status: {result.status}")
    print(f"Final state: raw={result.raw!r} processed={result.processed!r} checks={result.checks}")

    # The state passed in is untouched — each delta produces a new state.
    print(f"Initial state untouched: raw={state.raw!r}")

    print("Execution trace:")
    for event in graph.exporter.events:
        print(f"  {event.node_name}: {event.status}")


if __name__ == "__main__":
    asyncio.run(main())
