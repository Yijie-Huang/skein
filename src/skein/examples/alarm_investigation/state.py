"""Alarm investigation workflow example."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from skein.core.state import BaseState


class AlarmPayload(BaseModel):
    """Input alarm data."""
    alarm_id: str
    rule_name: str
    metric_name: str
    value: float
    threshold: float
    started_at: datetime
    services: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


AlarmType = Literal["latency", "traffic", "error_rate", "saturation",
    "availability", "unknown"]

Severity = Literal["low", "medium", "high", "critical"]

class TriageResult(BaseModel):
    """Result of alarm triage."""
    severity: Severity
    alarm_type: AlarmType


class Evidence(BaseModel):
    observation: str   # objective fact only — what a tool actually returned
    source: str        # which mcp tool / node produced it

CauseCategory = Literal[
    "dependency_failure", "resource_exhaustion",
    "config_change", "traffic_spike", "unknown",
]

class InvestigationResult(BaseModel):
    """Result of alarm investigation."""
    root_cause: str | None = None                      # human-readable narrative
    category: CauseCategory
    root_cause_service_name: str | None = None            
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


SummaryStatus = Literal["resolved", "pending", "escalated"]

class SummaryResult(BaseModel):
    """Summary of alarm investigation."""
    summary: str
    status: SummaryStatus
    reason: str | None = None


class AlarmInvestigationState(BaseState):
    alarm: AlarmPayload                          # input
    triage: TriageResult | None = None           # written by triage
    investigation: InvestigationResult | None = None
    recent_changes: list[str] = Field(           # written alongside investigation
        default_factory=list
    )
    summary: SummaryResult | None = None
