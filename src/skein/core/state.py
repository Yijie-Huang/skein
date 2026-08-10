"""Agent graph state management."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class GraphStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class BaseState(BaseModel):
    """Base class for agent graph state — subclass to add domain-specific fields."""
    trace_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    current_node: Optional[str] = None
    status: GraphStatus = GraphStatus.RUNNING
