"""Agent graph state management."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field


class SkeinStateError(Exception):
    """Raised when a node writes something the state cannot accept."""


class GraphStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class BaseState(BaseModel):
    """Base class for agent graph state — subclass to add domain-specific fields.

    Frozen: nodes describe changes by returning a delta, and the graph produces a
    new state from it. Assigning to a field raises instead of silently diverging
    from what the trace recorded.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_node: str | None = None
    status: GraphStatus = GraphStatus.RUNNING


S = TypeVar("S", bound=BaseState)
"""Type variable for a concrete state subclass, so nodes and graphs keep their type."""
