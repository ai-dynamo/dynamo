#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate a long-running agent workload using Dynamo request-trace evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

REQUIRED_TRIGGERS = frozenset({"user_message", "tool_result"})


@dataclass
class SessionSummary:
    session_id: str
    parent_session_ids: set[str] = field(default_factory=set)
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_call_count: int = 0
    input_triggers: Counter[str] = field(default_factory=Counter)
    finish_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def is_root(self) -> bool:
        return not self.parent_session_ids

    def as_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["parent_session_ids"] = sorted(self.parent_session_ids)
        value["input_triggers"] = dict(sorted(self.input_triggers.items()))
        value["finish_reasons"] = dict(sorted(self.finish_reasons.items()))
        value["is_root"] = self.is_root
        return value


@dataclass(frozen=True)
class ValidationConfig:
    expected_model: str
    minimum_root_sessions: int
    minimum_requests_per_session: int
    required_triggers: frozenset[str] = REQUIRED_TRIGGERS


def _unwrap_record(value: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: trace record must be a JSON object")
    event = value.get("event")
    if event is not None:
        if not isinstance(event, dict):
            raise ValueError(
                f"line {line_number}: event wrapper must contain an object"
            )
        return event
    return value


def load_records(path: Path, start_line: int = 0) -> list[dict[str, Any]]:
    if start_line < 0:
        raise ValueError("start_line must be non-negative")
    if not path.is_file():
        raise ValueError(f"request trace does not exist: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(errors="strict").splitlines(), 1):
        if line_number <= start_line or not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"line {line_number}: invalid request-trace JSON: {error.msg}"
            ) from error
        records.append(_unwrap_record(value, line_number))
    return records


def _request_model(record: dict[str, Any]) -> str | None:
    request = record.get("request")
    if isinstance(request, dict) and isinstance(request.get("model"), str):
        return request["model"]
    payload = record.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("model"), str):
        return payload["model"]
    return None


def _finish_metadata(record: dict[str, Any]) -> dict[str, Any]:
    request = record.get("request")
    if not isinstance(request, dict):
        return {}
    metadata = request.get("finish_reason_metadata")
    return metadata if isinstance(metadata, dict) else {}


def summarize_sessions(
    records: Iterable[dict[str, Any]], expected_model: str
) -> tuple[dict[str, SessionSummary], Counter[str]]:
    sessions: dict[str, SessionSummary] = {}
    event_counts: Counter[str] = Counter()

    for record in records:
        event_type = record.get("event_type")
        if isinstance(event_type, str):
            event_counts[event_type] += 1
        if event_type != "request_end" or _request_model(record) != expected_model:
            continue

        agent_context = record.get("agent_context")
        if not isinstance(agent_context, dict):
            continue
        session_id = agent_context.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue

        summary = sessions.setdefault(session_id, SessionSummary(session_id=session_id))
        parent_session_id = agent_context.get("parent_session_id")
        if isinstance(parent_session_id, str) and parent_session_id:
            summary.parent_session_ids.add(parent_session_id)

        summary.request_count += 1
        trigger = agent_context.get("input_trigger")
        if isinstance(trigger, str):
            summary.input_triggers[trigger] += 1

        request = record.get("request")
        if isinstance(request, dict):
            for key, target in (
                ("input_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
            ):
                value = request.get(key)
                if isinstance(value, int) and value >= 0:
                    setattr(summary, target, getattr(summary, target) + value)

        finish_metadata = _finish_metadata(record)
        finish_reason = finish_metadata.get("finish_reason")
        if isinstance(finish_reason, str):
            summary.finish_reasons[finish_reason] += 1
        tool_calls = finish_metadata.get("tool_calls")
        if isinstance(tool_calls, list):
            summary.tool_call_count += len(tool_calls)

    return sessions, event_counts


def validate_records(
    records: Iterable[dict[str, Any]], config: ValidationConfig
) -> tuple[dict[str, Any], list[str]]:
    sessions, event_counts = summarize_sessions(records, config.expected_model)
    root_sessions = [session for session in sessions.values() if session.is_root]
    qualifying_sessions = [
        session
        for session in root_sessions
        if session.request_count >= config.minimum_requests_per_session
        and config.required_triggers.issubset(session.input_triggers)
    ]

    errors: list[str] = []
    if len(qualifying_sessions) < config.minimum_root_sessions:
        errors.append(
            "expected at least "
            f"{config.minimum_root_sessions} qualifying root sessions, found "
            f"{len(qualifying_sessions)}"
        )
        for session in root_sessions:
            if session.request_count < config.minimum_requests_per_session:
                continue
            missing = config.required_triggers.difference(session.input_triggers)
            if missing:
                errors.append(
                    f"root session {session.session_id!r} is missing input triggers: "
                    + ", ".join(sorted(missing))
                )

    summary = {
        "schema_version": 1,
        "expected_model": config.expected_model,
        "record_count": sum(event_counts.values()),
        "event_counts": dict(sorted(event_counts.items())),
        "session_count": len(sessions),
        "root_session_count": len(root_sessions),
        "qualifying_root_session_count": len(qualifying_sessions),
        "requirements": {
            "minimum_root_sessions": config.minimum_root_sessions,
            "minimum_requests_per_session": config.minimum_requests_per_session,
            "required_input_triggers": sorted(config.required_triggers),
        },
        "sessions": [
            session.as_json()
            for session in sorted(sessions.values(), key=lambda item: item.session_id)
        ],
        "ok": not errors,
        "errors": errors,
    }
    return summary, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--minimum-root-sessions", type=int, default=1)
    parser.add_argument("--minimum-requests-per-session", type=int, default=4)
    parser.add_argument(
        "--start-line",
        type=int,
        default=0,
        help="Ignore this many physical JSONL lines from the beginning of the trace.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_root_sessions < 1:
        raise SystemExit("--minimum-root-sessions must be positive")
    if args.minimum_requests_per_session < 2:
        raise SystemExit("--minimum-requests-per-session must be at least 2")

    try:
        records = load_records(args.trace, args.start_line)
        summary, errors = validate_records(
            records,
            ValidationConfig(
                expected_model=args.expected_model,
                minimum_root_sessions=args.minimum_root_sessions,
                minimum_requests_per_session=args.minimum_requests_per_session,
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"request-trace validation failed: {error}", file=sys.stderr)
        return 2

    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if errors:
        for error in errors:
            print(f"request-trace validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
