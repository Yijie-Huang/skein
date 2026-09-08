from __future__ import annotations

import uuid
from datetime import datetime, timezone

from skein.core.graph import Graph
from skein.core.state import GraphStatus
from skein.exporters.memory import InMemoryExporter

from .nodes import InvestigationNode, RecentChangesNode, SummaryNode, TriageNode
from .state import AlarmInvestigationState, AlarmPayload


class AlarmInvestigationGraph(Graph[AlarmInvestigationState]):

    def __init__(self):
        super().__init__("Alarm Investigation Workflow", InMemoryExporter())
        self.add_node("triage", TriageNode())
        self.add_node("investigation", InvestigationNode())
        self.add_node("recent_changes", RecentChangesNode())
        self.add_node("summary", SummaryNode())
        # investigation and recent_changes both depend only on triage, so they
        # share a wave and run concurrently.
        self.add_edge("triage", "investigation")
        self.add_edge("triage", "recent_changes")
        self.add_edge("investigation", "summary")
        self.add_edge("recent_changes", "summary")
        self.set_entry_point("triage")
    
    async def investigate_alarm(self, alarm: AlarmPayload):
        initial_state = AlarmInvestigationState(trace_id=str(uuid.uuid4()), alarm=alarm)
        final_state = await self.run(initial_state)
        if final_state.status == GraphStatus.FAILED:
            print(f"Run failed: {self._first_error() or 'no error recorded'}")
        else:
            print(f"Final investigation summary: {final_state.summary}")
        exporter = self.exporter
        if isinstance(exporter, InMemoryExporter):
            print("Execution trace:")
            for event in exporter.events:
                # Nodes sharing a (wave, group) ran concurrently — compare the
                # timestamps of investigation and recent_changes to see it.
                print(
                    f"Node: {event.node_name}, wave={event.wave} group={event.group}, "
                    f"Status: {event.status}, "
                    f"Started at: {event.started_at}, Finished at: {event.finished_at}"
                )
                if event.error:
                    print(f"  error: {event.error}")

    def _first_error(self) -> str | None:
        exporter = self.exporter
        if not isinstance(exporter, InMemoryExporter):
            return None
        return next((event.error for event in exporter.events if event.error), None)
    
async def main():
    graph = AlarmInvestigationGraph()
    alarm = AlarmPayload(
        alarm_id="alarm-123",
        rule_name="high_latency",
        metric_name="p95_latency",
        value=0.92,
        threshold=0.9,
        started_at=datetime.now(timezone.utc),
        services=["service-a", "service-b"],
    )
    await graph.investigate_alarm(alarm)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
