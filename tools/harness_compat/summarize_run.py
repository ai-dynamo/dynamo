#!/usr/bin/env python3
"""Summarize a compatibility-lab artifact without exposing prompt or response text."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _agent_depths(trace_rows: list[dict[str, Any]]) -> dict[str, int]:
    parents: dict[str, str | None] = {}
    for row in trace_rows:
        event = row.get("event")
        if not isinstance(event, dict) or event.get("event_type") != "request_end":
            continue
        context = event.get("agent_context")
        if not isinstance(context, dict) or not isinstance(context.get("session_id"), str):
            continue
        parents[context["session_id"]] = context.get("parent_session_id")

    def depth(session_id: str, trail: set[str]) -> int:
        parent = parents.get(session_id)
        if parent is None or parent not in parents or parent in trail:
            return 0
        return 1 + depth(parent, trail | {session_id})

    return {session_id: depth(session_id, set()) for session_id in parents}


def _codex_tool_summary(root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for journal in (root / "codex_home" / "sessions").glob("**/*.jsonl"):
        for row in _jsonl(journal):
            payload = row.get("payload")
            if (
                row.get("type") == "response_item"
                and isinstance(payload, dict)
                and payload.get("type") == "function_call"
                and isinstance(payload.get("name"), str)
            ):
                counts[payload["name"]] += 1
    return dict(sorted(counts.items()))


def _protocol_discriminators(requests: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Stable shape sets for detecting client protocol drift across model runs."""
    values: dict[str, set[str]] = {
        "request_paths": set(),
        "header_names": set(),
        "top_level_keys": set(),
        "shape_keys": set(),
        "input_item_types": set(),
        "input_content_types": set(),
        "message_roles": set(),
        "message_content_types": set(),
        "tool_types": set(),
        "tool_names": set(),
        "output_config_keys": set(),
        "text_keys": set(),
    }
    for request in requests:
        if isinstance(request.get("path"), str):
            values["request_paths"].add(request["path"])
        values["header_names"].update(str(name) for name in request.get("headers", {}))
        shape = request.get("shape")
        if not isinstance(shape, dict):
            continue
        values["shape_keys"].update(str(key) for key in shape)
        values["top_level_keys"].update(str(key) for key in shape.get("top_level_keys", []))
        for item in shape.get("input_items", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") is not None:
                values["input_item_types"].add(str(item["type"]))
            values["input_content_types"].update(str(kind) for kind in item.get("content_types", []))
        for message in shape.get("messages", []):
            if not isinstance(message, dict):
                continue
            if message.get("role") is not None:
                values["message_roles"].add(str(message["role"]))
            values["message_content_types"].update(str(kind) for kind in message.get("content_types", []))
        values["tool_types"].update(str(kind) for kind in shape.get("tool_types", []))
        values["tool_names"].update(str(name) for name in shape.get("tool_names", []))
        values["output_config_keys"].update(str(key) for key in shape.get("output_config_keys", []))
        values["text_keys"].update(str(key) for key in shape.get("text_keys", []))
    return {key: sorted(value) for key, value in values.items()}


def _trace_window(scenario: dict[str, Any], wire: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Scope a cumulative server trace to this artifact's wall-clock window."""
    start = scenario.get("run_started_unix_ms")
    end = scenario.get("run_finished_unix_ms")
    if not isinstance(start, int) or not isinstance(end, int):
        request_times = [row.get("timestamp_unix_ms") for row in wire if row.get("kind") == "request"]
        response_times = [row.get("timestamp_unix_ms") for row in wire if row.get("kind") == "response"]
        times = [value for value in request_times + response_times if isinstance(value, int)]
        if not times:
            return None
        start, end = min(times), max(times)
    # The proxy timestamp is local to the client side and the frontend emits a
    # completion event shortly after it. This cushion also covers clock rounding.
    return start - 5_000, end + 5_000


def _trace_for_run(raw_trace: list[dict[str, Any]], scenario: dict[str, Any], wire: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer the redacted client-session match over a clock-based fallback."""
    expected_digest = scenario.get("client_session_sha256_12")
    if isinstance(expected_digest, str):
        contexts: list[dict[str, Any]] = []
        for row in raw_trace:
            event = row.get("event")
            if isinstance(event, dict) and isinstance(event.get("agent_context"), dict):
                contexts.append(event["agent_context"])
        session_ids = {
            context["session_id"]
            for context in contexts
            if isinstance(context.get("session_id"), str)
            and hashlib.sha256(context["session_id"].encode()).hexdigest()[:12] == expected_digest
        }
        # Codex descendants receive distinct session IDs. Include their complete
        # parent closure without storing any plain session identifier in output.
        while True:
            expanded = {
                context["session_id"]
                for context in contexts
                if isinstance(context.get("session_id"), str) and context.get("parent_session_id") in session_ids
            }
            if expanded <= session_ids:
                break
            session_ids |= expanded
        if session_ids:
            return [
                row
                for row in raw_trace
                if isinstance(row.get("event"), dict)
                and isinstance(row["event"].get("agent_context"), dict)
                and row["event"]["agent_context"].get("session_id") in session_ids
            ]

    window = _trace_window(scenario, wire)
    return [
        row
        for row in raw_trace
        if window is None
        or (
            isinstance(row.get("event"), dict)
            and isinstance(row["event"].get("event_time_unix_ms"), int)
            and window[0] <= row["event"]["event_time_unix_ms"] <= window[1]
        )
    ]


def summarize(root: Path) -> dict[str, Any]:
    wire = _jsonl(root / "wire.jsonl")
    scenario_path = root / "scenario.json"
    scenario = json.loads(scenario_path.read_text()) if scenario_path.exists() else {}
    raw_trace = _jsonl(root / "remote" / "request-trace.jsonl")
    trace = _trace_for_run(raw_trace, scenario, wire)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    transport_path = root / "transport.json"
    transport = json.loads(transport_path.read_text()) if transport_path.exists() else None
    requests = [row for row in wire if row.get("kind") == "request"]
    responses = [row for row in wire if row.get("kind") == "response"]
    request_shapes = Counter(json.dumps(row.get("shape", {}), sort_keys=True) for row in requests)
    depths = _agent_depths(trace)
    return {
        "run": root.name,
        "result": result,
        "transport": transport,
        "http": {
            "request_count": len(requests),
            "request_paths": dict(sorted(Counter(row.get("path") for row in requests).items())),
            "response_statuses": dict(sorted(Counter(row.get("status") for row in responses).items())),
            "request_header_names": sorted({header for row in requests for header in row.get("headers", {})}),
            # `wire.jsonl` retains the redacted structural details for diagnosing a
            # drift. Keep the main summary compact and safe to check into a nightly
            # baseline: it needs to say that the shape changed, not repeat a large
            # tools schema for every scenario.
            "request_shape_digests": [
                {
                    "count": count,
                    "sha256_12": hashlib.sha256(shape.encode()).hexdigest()[:12],
                }
                for shape, count in sorted(request_shapes.items())
            ],
            "protocol_discriminators": _protocol_discriminators(requests),
            "sse_events": dict(
                sorted(Counter(row.get("event") for row in wire if row.get("kind") == "sse_event").items())
            ),
            "stream_tool_names": dict(
                sorted(Counter(row.get("tool_name") for row in wire if isinstance(row.get("tool_name"), str)).items())
            ),
            "terminal_stop_reasons": dict(
                sorted(
                    Counter(row.get("stop_reason") for row in wire if isinstance(row.get("stop_reason"), str)).items()
                )
            ),
            "error_shapes": [row.get("shape") for row in wire if row.get("kind") == "response_error"],
            "injected_faults": [
                {
                    key: row[key]
                    for key in ("fault", "request_number", "status", "after_sse_events")
                    if key in row
                }
                for row in wire
                if row.get("kind") == "fault_injected"
            ],
        },
        "codex_function_calls": _codex_tool_summary(root),
        "agent_context": {
            "trace_event_count": len(trace),
            "session_count": len(depths),
            "max_depth": max(depths.values(), default=0),
            "parented_session_count": sum(depth > 0 for depth in depths.values()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.artifact.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
