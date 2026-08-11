"""Node interface for agentic workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Generic

from .state import S

StateDelta = dict[str, Any]
"""The fields a node changed. Anything the state does not declare is rejected."""

NodeFunction = Callable[[S], Awaitable[StateDelta | None]]


class Node(ABC, Generic[S]):
    """Base interface for graph nodes.

    Parameterise with your state type — ``class Triage(Node[AlarmState])`` — so
    ``run`` can narrow its argument without violating the base signature.
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    @abstractmethod
    async def run(self, state: S) -> StateDelta | None:
        """Execute the node logic and return the fields it changed, if any."""
