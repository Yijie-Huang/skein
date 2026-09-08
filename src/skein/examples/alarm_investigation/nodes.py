"""Nodes for the alarm investigation workflow example."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from skein.core.node import Node, StateDelta
from skein.core.trace import langsmith_tracing_enabled

from .mcp_server import lookup_service_cpu
from .state import (
    AlarmInvestigationState,
    AlarmType,
    InvestigationResult,
    Severity,
    SummaryResult,
    SummaryStatus,
    TriageResult,
)

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


_MAX_REACT_ROUNDS = 4


class InvestigationUnavailable(RuntimeError):
    """The investigation step cannot run — misconfiguration, not a bad alarm."""


class InvestigationFailed(RuntimeError):
    """The investigation ran but produced nothing usable."""

_INVESTIGATION_SYSTEM_PROMPT = """You are an alarm investigation agent using a ReAct-style workflow.

You may call tools to gather evidence before deciding.
Reason step by step internally, but do not reveal chain-of-thought.
Your final answer must be valid JSON only and must match this schema exactly:
{
    "root_cause": "string or null",
    "category": "dependency_failure|resource_exhaustion|config_change|traffic_spike|unknown",
    "root_cause_service_name": "string or null",
    "evidence": [
        {
            "observation": "string",
            "source": "string"
        }
    ],
    "confidence": 0.0
}

Use these heuristics unless tool evidence clearly contradicts them:
- Start from the alarm's first impacted service if one exists.
- Query CPU for that service with the available tool.
- If CPU usage is >= 0.90, classify the case as resource_exhaustion with confidence close to 0.90.
- Otherwise keep resource_exhaustion as the default hypothesis with confidence close to 0.75.
- Evidence observations must be objective facts derived from tool outputs.
"""


def _load_mcp_client() -> tuple[Any, Any, Any]:
    try:
        mcp_module = importlib.import_module("mcp")
        stdio_module = importlib.import_module("mcp.client.stdio")
    except ImportError:  # pragma: no cover - optional dependency
        return None, None, None

    return (
        getattr(mcp_module, "ClientSession", None),
        getattr(mcp_module, "StdioServerParameters", None),
        getattr(stdio_module, "stdio_client", None),
    )


def _load_langsmith_trace() -> Any:
    try:
        langsmith_module = importlib.import_module("langsmith")
    except ImportError:
        return None
    return getattr(langsmith_module, "trace", None)


def _load_langsmith_tracing_context() -> Any:
    try:
        run_helpers_module = importlib.import_module("langsmith.run_helpers")
    except ImportError:
        return None
    return getattr(run_helpers_module, "tracing_context", None)


def _load_langsmith_wrap_anthropic() -> Any:
    try:
        wrappers_module = importlib.import_module("langsmith.wrappers")
    except ImportError:
        return None
    return getattr(wrappers_module, "wrap_anthropic", None)


@contextmanager
def _maybe_langsmith_trace(
    name: str,
    run_type: str,
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    if not langsmith_tracing_enabled():
        yield None
        return

    trace_fn = _load_langsmith_trace()
    tracing_context_fn = _load_langsmith_tracing_context()
    if trace_fn is None or tracing_context_fn is None:
        yield None
        return

    project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    with tracing_context_fn(enabled=True, project_name=project_name):
        with trace_fn(
            name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata,
            project_name=project_name,
        ) as run:
            yield run


def _load_anthropic_client() -> Any:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise InvestigationUnavailable(
            "ANTHROPIC_API_KEY is not set. The investigation node calls Claude directly — "
            "export the key, or put it in a project .env, and run the demo again."
        )

    try:
        anthropic_module = importlib.import_module("anthropic")
    except ImportError as exc:
        raise InvestigationUnavailable(
            "The 'anthropic' package is required by the investigation node. "
            "Install it with: pip install anthropic"
        ) from exc

    client_cls = getattr(anthropic_module, "AsyncAnthropic", None)
    if client_cls is None:
        raise InvestigationUnavailable(
            "The installed 'anthropic' package has no AsyncAnthropic client; upgrade it."
        )
    client = client_cls()
    if not langsmith_tracing_enabled():
        return client

    wrap_anthropic = _load_langsmith_wrap_anthropic()
    if wrap_anthropic is None:
        return client
    return wrap_anthropic(client)


def _extract_text_blocks(content: list[Any]) -> str:
    text_parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            text_parts.append(text)
    return "".join(text_parts).strip()


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
        stripped = stripped.strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


async def _run_react_investigation(state: AlarmInvestigationState) -> InvestigationResult:
    client = _load_anthropic_client()

    service_name = state.alarm.services[0] if state.alarm.services else "unknown-service"
    triage = state.triage
    model = os.getenv("SKEIN_ANTHROPIC_MODEL", _DEFAULT_ANTHROPIC_MODEL)
    user_prompt = {
        "alarm": state.alarm.model_dump(mode="json"),
        "triage": triage.model_dump(mode="json") if triage is not None else None,
        "instructions": {
            "focus_service": service_name,
            "return_json_only": True,
        },
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": json.dumps(user_prompt, ensure_ascii=True),
        }
    ]
    tools = [
        {
            "name": "get_service_cpu",
            "description": (
                "Fetch the current CPU usage observation for a service impacted by the alarm."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The service name to inspect.",
                    }
                },
                "required": ["service_name"],
            },
        }
    ]

    with _maybe_langsmith_trace(
        "react_investigation",
        run_type="chain",
        inputs={
            "alarm_id": state.alarm.alarm_id,
            "service_name": service_name,
            "triage": triage.model_dump(mode="json") if triage is not None else None,
        },
        metadata={"node": "investigation", "model": model},
    ) as react_run:
        for iteration in range(_MAX_REACT_ROUNDS):
            response = await client.messages.create(
                model=model,
                max_tokens=700,
                system=_INVESTIGATION_SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [
                block for block in assistant_content if getattr(block, "type", None) == "tool_use"
            ]
            if tool_uses:
                tool_results: list[dict[str, Any]] = []
                for block in tool_uses:
                    if getattr(block, "name", None) != "get_service_cpu":
                        continue
                    tool_input = getattr(block, "input", {}) or {}
                    requested_service = tool_input.get("service_name", service_name)
                    cpu_sample, evidence_source = await _collect_cpu_evidence(requested_service)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(
                                {
                                    "service_name": requested_service,
                                    "cpu_usage": cpu_sample["cpu_usage"],
                                    "status": cpu_sample["status"],
                                    "source": evidence_source,
                                },
                                ensure_ascii=True,
                            ),
                        }
                    )
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                    continue

            final_text = _extract_text_blocks(assistant_content)
            if not final_text:
                if react_run is not None:
                    react_run.end(outputs={"result": None, "iteration": iteration + 1})
                raise InvestigationFailed(
                    f"the model returned no text on round {iteration + 1}, "
                    "so there is nothing to parse into an InvestigationResult"
                )
            try:
                with _maybe_langsmith_trace(
                    "parse_investigation_result",
                    run_type="parser",
                    inputs={"response_text": final_text},
                    metadata={"node": "investigation", "iteration": iteration + 1},
                ) as parse_run:
                    result = InvestigationResult.model_validate_json(
                        _extract_json_payload(final_text)
                    )
                    if parse_run is not None:
                        parse_run.end(outputs=result.model_dump(mode="json"))
                if react_run is not None:
                    react_run.end(
                        outputs={
                            "result": result.model_dump(mode="json"),
                            "iteration": iteration + 1,
                        }
                    )
                return result
            except Exception as exc:
                if react_run is not None:
                    react_run.end(outputs={"result": None, "iteration": iteration + 1})
                raise InvestigationFailed(
                    "the model's final answer did not match the InvestigationResult schema: "
                    f"{final_text[:200]}"
                ) from exc

    raise InvestigationFailed(
        f"the investigation did not reach a conclusion within {_MAX_REACT_ROUNDS} tool-use rounds"
    )


async def _collect_cpu_evidence(service_name: str) -> tuple[dict[str, object], str]:
    with _maybe_langsmith_trace(
        "get_service_cpu",
        run_type="tool",
        inputs={"service_name": service_name},
        metadata={"node": "investigation"},
    ) as tool_run:
        client_session_cls, server_parameters_cls, stdio_client_fn = _load_mcp_client()
        if (
            client_session_cls is None
            or server_parameters_cls is None
            or stdio_client_fn is None
        ):
            payload = lookup_service_cpu(service_name)
            if tool_run is not None:
                tool_run.end(outputs={"payload": payload, "source": "local_fallback_probe"})
            return payload, "local_fallback_probe"

        server_script = Path(__file__).with_name("mcp_server.py")
        server_params = server_parameters_cls(command=sys.executable, args=[str(server_script)])

        try:
            async with stdio_client_fn(server_params) as (read_stream, write_stream):
                async with client_session_cls(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_result = await session.call_tool(
                        "get_service_cpu",
                        {"service_name": service_name},
                    )
        except Exception:
            payload = lookup_service_cpu(service_name)
            if tool_run is not None:
                tool_run.end(outputs={"payload": payload, "source": "local_fallback_probe"})
            return payload, "local_fallback_probe"

        text_parts: list[str] = []
        for item in getattr(tool_result, "content", []):
            text = getattr(item, "text", None)
            if text:
                text_parts.append(text)

        if not text_parts:
            payload = lookup_service_cpu(service_name)
            if tool_run is not None:
                tool_run.end(outputs={"payload": payload, "source": "local_fallback_probe"})
            return payload, "local_fallback_probe"

        payload = json.loads("".join(text_parts))
        if tool_run is not None:
            tool_run.end(
                outputs={"payload": payload, "source": "alarm_investigation_mcp.get_service_cpu"}
            )
        return payload, "alarm_investigation_mcp.get_service_cpu"


class TriageNode(Node[AlarmInvestigationState]):
    """Triage node for alarm investigation workflow."""

    def __init__(self):
        super().__init__("triage", writes=["triage"])
    
    async def run(self, state: AlarmInvestigationState) -> StateDelta:
        alarm = state.alarm
        print(f"[TriageNode] Triage alarm {alarm.alarm_id}")
        severity: Severity
        if alarm.threshold > 0.9:
            severity = "critical" if alarm.value > 0.95 else "high"
        else:
            severity = "medium" if alarm.value > 0.6 else "low"
        alarm_type: AlarmType = "latency" if "latency" in alarm.rule_name else "unknown"
        return {"triage": TriageResult(severity=severity, alarm_type=alarm_type)}


_RECENT_DEPLOYS = {
    "service-a": ["deploy 4f2a1c at 09:12", "config change: pool_size 20 -> 8 at 09:20"],
    "service-b": ["deploy 91be07 at 08:40"],
}


class RecentChangesNode(Node[AlarmInvestigationState]):
    """Pull the change log for the alarmed service.

    Deliberately trivial and LLM-free: it depends only on triage, exactly as the
    investigation does, so the two share a wave and run concurrently.
    """

    def __init__(self):
        super().__init__("recent_changes", writes=["recent_changes"])

    async def run(self, state: AlarmInvestigationState) -> StateDelta:
        service = state.alarm.services[0] if state.alarm.services else "unknown-service"
        print(f"[RecentChangesNode] Looking up recent changes for {service}")
        await asyncio.sleep(0.2)  # stand-in for a deploy-history API
        return {"recent_changes": _RECENT_DEPLOYS.get(service, [])}


class InvestigationNode(Node[AlarmInvestigationState]):
    """Investigation node for alarm investigation workflow."""

    def __init__(self):
        super().__init__("investigation", writes=["investigation"])
    
    async def run(self, state: AlarmInvestigationState) -> StateDelta:
        triage = state.triage
        if triage is None:
            raise InvestigationUnavailable(
                "investigation ran before triage produced a result — check the graph edges"
            )
        print(
            f"[InvestigationNode] Investigating alarm {state.alarm.alarm_id} "
            f"with severity {triage.severity}"
        )
        return {"investigation": await _run_react_investigation(state)}


class SummaryNode(Node[AlarmInvestigationState]):
    """Summary node for alarm investigation workflow."""

    def __init__(self):
        super().__init__("summary", writes=["summary"])
    
    async def run(self, state: AlarmInvestigationState) -> StateDelta:
        investigation = state.investigation
        if investigation is None:
            raise InvestigationUnavailable(
                "summary ran before investigation produced a result — check the graph edges"
            )
        print(f"[SummaryNode] Summarizing investigation for alarm {state.alarm.alarm_id}")
        status: SummaryStatus
        if investigation.confidence > 0.8:
            status = "resolved"
            reason = f"Root cause identified as {investigation.root_cause}"
        else:
            status = "pending"
            reason = "Insufficient evidence to determine root cause"
        # Both branches of the wave land here, so the summary can use either.
        if state.recent_changes:
            reason += f" ({len(state.recent_changes)} recent change(s) near the alarm)"
        return {
            "summary": SummaryResult(
                summary="Alarm investigation completed", status=status, reason=reason
            )
        }

