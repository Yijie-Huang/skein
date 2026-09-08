"""Tracing and event recording for agent execution."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

_TRUTHY = {"1", "true", "yes", "on"}


def langsmith_tracing_enabled() -> bool:
    """Whether LangSmith export is switched on.

    Exporting to LangSmith is opt-in: installing the optional dependency is not
    enough, ``LANGSMITH_TRACING`` (or the legacy ``LANGCHAIN_TRACING_V2``) must
    also be set. Read at call time so the flag can be toggled at runtime.
    """
    raw = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return raw.strip().lower() in _TRUTHY

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TraceEvent(BaseModel):
    """Complete record of an agent run — used for eval and debugging.

    ``wave`` is dependency depth; ``group`` is the batch within it that actually
    ran together. Nodes sharing both ran concurrently; nodes in the same wave but
    different groups were split apart, so the scheduling is never left to be
    inferred from timestamps.
    """
    trace_id: str
    node_name: str
    wave: int | None = None
    group: int | None = None
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
