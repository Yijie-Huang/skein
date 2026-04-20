"""Core schemas for Skein. Domain-agnostic — examples define their own domain types."""

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from enum import Enum
from datetime import datetime


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """A unit of work for an agent to process."""
    id: str
    description: str
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A planned tool invocation."""
    tool_name: str
    tool_args: dict[str, Any]
    rationale: str = ""


class PlanStep(BaseModel):
    """A single step in an execution plan."""
    step_id: str
    description: str
    tool_call: ToolCall
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """Output of a Planner agent."""
    task_id: str
    hypothesis: str
    steps: list[PlanStep]
    reasoning: str = ""


class Finding(BaseModel):
    """Output of an Executor tool invocation."""
    step_id: str
    tool_name: str
    raw_output: str
    summary: str
    relevant: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conclusion(BaseModel):
    """Output of a Critic agent — the final synthesis."""
    task_id: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_findings: list[str]  # step_ids
    suggested_actions: list[str] = Field(default_factory=list)


class Trajectory(BaseModel):
    """Complete record of an agent run — used for eval and debugging."""
    task_id: str
    task: Task
    plan: Optional[Plan] = None
    findings: list[Finding] = Field(default_factory=list)
    conclusion: Optional[Conclusion] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None