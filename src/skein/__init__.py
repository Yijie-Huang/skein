"""Skein — agentic workflow framework."""

from __future__ import annotations

import importlib


def _load_workspace_dotenv() -> None:
	try:
		dotenv_module = importlib.import_module("dotenv")
	except ImportError:
		return

	find_dotenv = getattr(dotenv_module, "find_dotenv", None)
	load_dotenv = getattr(dotenv_module, "load_dotenv", None)
	if find_dotenv is None or load_dotenv is None:
		return

	load_dotenv(find_dotenv(usecwd=True), override=False)

# Load the nearest workspace .env without overriding already-exported shell vars.
# This must run before the imports below, so they are intentionally not at the
# top of the file (E402 is suppressed rather than reordered).
_load_workspace_dotenv()

from .core import BaseState, Graph, Node, StateDelta, TraceEvent  # noqa: E402
from .core.node import NodeFunction  # noqa: E402
from .core.state import SkeinStateError  # noqa: E402
from .exporters import Exporter, InMemoryExporter, JSONLExporter, NoOpExporter  # noqa: E402
from .logging_config import configure_logging, get_logger  # noqa: E402

# Initialize logging on import
configure_logging()

__all__ = [
	"BaseState",
	"Exporter",
	"Graph",
	"InMemoryExporter",
	"JSONLExporter",
	"NoOpExporter",
	"Node",
	"NodeFunction",
	"SkeinStateError",
	"StateDelta",
	"TraceEvent",
	"configure_logging",
	"get_logger",
]
