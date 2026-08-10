"""Node interface for agentic workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from .state import BaseState

NodeFunction = Callable[[BaseState], Awaitable[BaseState]]

class Node(ABC):
    """Base interface for graph nodes."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    @abstractmethod
    async def run(self, state: BaseState) -> BaseState:
        """Execute the node logic."""
        pass
