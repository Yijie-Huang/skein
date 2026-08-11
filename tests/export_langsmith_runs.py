"""Export LangSmith runs to JSONL for offline inspection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from langsmith import Client

import skein  # noqa: F401  # Ensures project .env is loaded before creating the client.


def _build_filter(trace_id: str | None, run_type: str | None, raw_filter: str | None) -> str | None:
    if raw_filter:
        return raw_filter

    clauses: list[str] = []
    if trace_id:
        clauses.append(f'eq(trace_id, "{trace_id}")')
    if run_type:
        clauses.append(f'eq(run_type, "{run_type}")')
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return f'and({", ".join(clauses)})'


def _serialize_run(run: Any) -> dict[str, Any]:
    if hasattr(run, "model_dump"):
        payload = run.model_dump(mode="json")
    else:
        payload = dict(run)

    return {
        "id": payload.get("id"),
        "name": payload.get("name"),
        "trace_id": payload.get("trace_id"),
        "parent_run_id": payload.get("parent_run_id"),
        "run_type": payload.get("run_type"),
        "status": payload.get("status"),
        "start_time": payload.get("start_time"),
        "end_time": payload.get("end_time"),
        "error": payload.get("error"),
        "inputs": payload.get("inputs"),
        "outputs": payload.get("outputs"),
        "metadata": payload.get("metadata"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LangSmith runs to JSONL.")
    parser.add_argument(
        "--project-name",
        default=os.getenv("LANGSMITH_PROJECT"),
        help="LangSmith project name. Defaults to LANGSMITH_PROJECT.",
    )
    parser.add_argument(
        "--trace-id",
        default=None,
        help="Trace ID to export. Builds filter eq(trace_id, \"...\").",
    )
    parser.add_argument(
        "--run-type",
        default=None,
        help="Optional run_type filter such as chain, tool, llm, or parser.",
    )
    parser.add_argument(
        "--filter",
        dest="raw_filter",
        default=None,
        help="Raw LangSmith filter expression. Overrides --trace-id and --run-type.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of runs to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("langsmith_runs.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.project_name:
        raise SystemExit("Missing --project-name and LANGSMITH_PROJECT is not set.")

    filter_expr = _build_filter(args.trace_id, args.run_type, args.raw_filter)
    client = Client()
    runs = client.list_runs(project_name=args.project_name, filter=filter_expr)

    exported = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(_serialize_run(run), ensure_ascii=True) + "\n")
            exported += 1
            if exported >= args.limit:
                break

    print(f"Exported {exported} runs to {args.output}")
    if filter_expr:
        print(f"Filter: {filter_expr}")


if __name__ == "__main__":
    main()