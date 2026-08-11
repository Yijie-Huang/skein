"""Node interface for agentic workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Generic

from .state import S

StateDelta = dict[str, Any]
"""The fields a node changed. Anything the state does not declare is rejected."""

NodeFunction = Callable[[S], Awaitable[StateDelta | None]]


def normalize_writes(writes: Iterable[str] | None) -> frozenset[str] | None:
    """Turn a declared write set into a frozenset, or ``None`` when undeclared."""
    if writes is None:
        return None
    if isinstance(writes, str):
        raise TypeError(
            f"writes must be a collection of field names, not the string {writes!r} — "
            f'did you mean ["{writes}"]?'
        )
    return frozenset(writes)


class Node(ABC, Generic[S]):
    """Base interface for graph nodes.

    Parameterise with your state type — ``class Triage(Node[AlarmState])`` — so
    ``run`` can narrow its argument without violating the base signature.

    Pass ``writes`` to declare which state fields this node is allowed to change::

        class DecideNode(Node[ReviewState]):
            def __init__(self) -> None:
                super().__init__("decide", writes=["verdict"])

    The graph enforces it: returning a field outside ``writes`` raises
    ``SkeinStateError``, and so does declaring a field the state does not have.
    ``status`` is exempt — any node may end a run early with it. Leave ``writes``
    unset to opt out of the check entirely.
    """

    def __init__(self, name: str, writes: Iterable[str] | None = None):
        self.name = name
        self.writes: frozenset[str] | None = normalize_writes(writes)

    def __repr__(self) -> str:
        if self.writes is None:
            return f"{self.__class__.__name__}(name={self.name!r})"
        return f"{self.__class__.__name__}(name={self.name!r}, writes={sorted(self.writes)})"

    @abstractmethod
    async def run(self, state: S) -> StateDelta | None:
        """Execute the node logic and return the fields it changed, if any."""
