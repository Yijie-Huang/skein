"""Tests for the alarm investigation example — the paths that need no API call."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from skein.core.state import GraphStatus
from skein.examples.alarm_investigation.graph import AlarmInvestigationGraph
from skein.examples.alarm_investigation.nodes import (
    InvestigationNode,
    InvestigationUnavailable,
    SummaryNode,
    TriageNode,
)
from skein.examples.alarm_investigation.state import (
    AlarmInvestigationState,
    AlarmPayload,
    InvestigationResult,
    TriageResult,
)


def make_alarm(value: float = 0.92, threshold: float = 0.9) -> AlarmPayload:
    return AlarmPayload(
        alarm_id="alarm-123",
        rule_name="high_latency",
        metric_name="p95_latency",
        value=value,
        threshold=threshold,
        started_at=datetime.now(timezone.utc),
        services=["service-a"],
    )


def make_state(**overrides) -> AlarmInvestigationState:
    return AlarmInvestigationState(trace_id="t-alarm", alarm=make_alarm(), **overrides)


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return monkeypatch


def test_triage_classifies_a_latency_alarm():
    # threshold is 0.9 exactly, and the rule is `threshold > 0.9`, so this is the medium branch
    delta = asyncio.run(TriageNode().run(make_state()))

    assert delta["triage"] == TriageResult(severity="medium", alarm_type="latency")


@pytest.mark.parametrize(
    ("value", "threshold", "expected"),
    [
        (0.99, 0.95, "critical"),
        (0.93, 0.95, "high"),
        (0.92, 0.90, "medium"),
        (0.10, 0.90, "low"),
    ],
)
def test_triage_severity_ladder(value, threshold, expected):
    state = AlarmInvestigationState(
        trace_id="t-sev", alarm=make_alarm(value=value, threshold=threshold)
    )

    delta = asyncio.run(TriageNode().run(state))

    assert delta["triage"].severity == expected


def test_investigation_without_api_key_names_the_missing_variable(no_api_key):
    state = make_state(triage=TriageResult(severity="high", alarm_type="latency"))

    with pytest.raises(InvestigationUnavailable, match="ANTHROPIC_API_KEY is not set"):
        asyncio.run(InvestigationNode().run(state))


def test_investigation_without_triage_is_reported_not_crashed(no_api_key):
    with pytest.raises(InvestigationUnavailable, match="before triage"):
        asyncio.run(InvestigationNode().run(make_state()))


def test_summary_without_an_investigation_is_reported_not_crashed():
    with pytest.raises(InvestigationUnavailable, match="before investigation"):
        asyncio.run(SummaryNode().run(make_state()))


def test_summary_marks_a_confident_result_resolved():
    state = make_state(
        investigation=InvestigationResult(
            root_cause="cpu saturation",
            category="resource_exhaustion",
            confidence=0.9,
        )
    )

    summary = asyncio.run(SummaryNode().run(state))["summary"]

    assert summary.status == "resolved"
    assert "cpu saturation" in summary.reason


def test_summary_stays_pending_when_confidence_is_low():
    state = make_state(
        investigation=InvestigationResult(category="unknown", confidence=0.4)
    )

    summary = asyncio.run(SummaryNode().run(state))["summary"]

    assert summary.status == "pending"


def test_graph_run_without_api_key_fails_with_the_reason_in_the_trace(no_api_key):
    graph = AlarmInvestigationGraph()
    state = AlarmInvestigationState(trace_id="t-nokey", alarm=make_alarm())

    result = asyncio.run(graph.run(state))

    assert result.status == GraphStatus.FAILED
    assert result.triage is not None            # triage still ran
    assert result.investigation is None
    assert result.summary is None               # summary never ran, no AttributeError

    events = graph.exporter.events
    assert [event.node_name for event in events] == ["triage", "investigation"]
    assert "ANTHROPIC_API_KEY is not set" in events[-1].error
