from __future__ import annotations

import uuid
from .state import AlarmInvestigationState, AlarmPayload
from .nodes import TriageNode, InvestigationNode, SummaryNode
from skein.core.graph import Graph
from skein.exporters.memory import InMemoryExporter

class AlarmInvestigationGraph(Graph):

    def __init__(self):
        super().__init__("Alarm Investigation Workflow", InMemoryExporter())
        self.add_node("triage", TriageNode())
        self.add_node("investigation", InvestigationNode())
        self.add_node("summary", SummaryNode())
        self.add_edge("triage", "investigation")
        self.add_edge("investigation", "summary")
        self.set_entry_point("triage")
    
    async def investage_alarm(self, alarm: AlarmPayload):
        initial_state = AlarmInvestigationState(trace_id=str(uuid.uuid4()), alarm=alarm)
        final_state = await self.run(initial_state)
        print(f"Final investigation summary: {final_state.summary}")
        exporter = self.exporter
        if isinstance(exporter, InMemoryExporter):
            print("Execution trace:")
            for event in exporter.events:
                print(f"Node: {event.node_name}, Status: {event.status}, Started at: {event.started_at}, Finished at: {event.finished_at}")
    
async def main():
    graph = AlarmInvestigationGraph()
    alarm = AlarmPayload(
        alarm_id="alarm-123",
        rule_name="high_latency",
        metric_name="p95_latency",
        value=0.92,
        threshold=0.9,
        started_at=__import__("datetime").datetime.now(),
        services=["service-a", "service-b"],
    )
    await graph.investage_alarm(alarm)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
