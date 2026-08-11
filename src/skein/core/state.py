"""Agent graph state management."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class GraphStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class BaseState(BaseModel):
    """Base class for agent graph state — subclass to add domain-specific fields."""
    trace_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_node: str | None = None
    status: GraphStatus = GraphStatus.RUNNING
