"""LangSmith export is opt-in, even when the optional dependency is installed."""

from __future__ import annotations

import asyncio

import pytest

from skein import BaseState, Graph
from skein.core import graph as graph_module
from skein.core.trace import langsmith_tracing_enabled

TRACING_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")


@pytest.fixture
def tracing_env(monkeypatch):
    for var in TRACING_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize("value", ["true", "TRUE", " True ", "1", "yes", "on"])
def test_flag_enables_tracing(tracing_env, value):
    tracing_env.setenv("LANGSMITH_TRACING", value)
    assert langsmith_tracing_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
def test_flag_disables_tracing(tracing_env, value):
    tracing_env.setenv("LANGSMITH_TRACING", value)
    assert langsmith_tracing_enabled() is False


def test_unset_flag_disables_tracing(tracing_env):
    assert langsmith_tracing_enabled() is False


def test_legacy_flag_is_honoured(tracing_env):
    tracing_env.setenv("LANGCHAIN_TRACING_V2", "true")
    assert langsmith_tracing_enabled() is True


def test_disabled_flag_never_builds_a_langsmith_client(tracing_env):
    """The expensive path — client construction and export — must not be reached."""
    tracing_env.setattr(
        graph_module,
        "_load_langsmith_client",
        lambda: pytest.fail("LangSmith client built while tracing is disabled"),
    )
    tracing_env.setattr(graph_module, "_LANGSMITH_CLIENT", graph_module._UNSET)

    async def noop(state: BaseState) -> BaseState:
        return state

    graph = Graph("gated")
    graph.add_node("only", noop)
    graph.set_entry_point("only")

    result = asyncio.run(graph.run(BaseState(trace_id="t-gated")))
    assert result.status.value == "completed"
