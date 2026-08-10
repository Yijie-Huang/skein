"""Minimal MCP server for the alarm investigation demo."""

from __future__ import annotations

from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional dependency
    FastMCP = None


def lookup_service_cpu(service_name: str) -> dict[str, Any]:
    """Return a deterministic synthetic CPU sample for a service."""
    cpu_samples = {
        "service-a": 0.95,
        "service-b": 0.72,
    }
    cpu_usage = cpu_samples.get(service_name, 0.61)
    status = "critical" if cpu_usage >= 0.9 else "ok"
    return {
        "service_name": service_name,
        "cpu_usage": cpu_usage,
        "status": status,
    }


if FastMCP is not None:
    mcp = FastMCP("alarm-investigation")

    @mcp.tool()
    def get_service_cpu(service_name: str) -> dict[str, Any]:
        """Return a synthetic CPU usage sample for an alarmed service."""
        return lookup_service_cpu(service_name)

else:  # pragma: no cover - optional dependency
    mcp = None


def main() -> None:
    if mcp is None:
        raise RuntimeError("Install skein[mcp] to run the alarm investigation MCP demo.")
    mcp.run()


if __name__ == "__main__":
    main()