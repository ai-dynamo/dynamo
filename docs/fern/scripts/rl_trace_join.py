#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join an application-owned RL attempt ledger to Dynamo request-trace v1.

This is a documentation example, not a new Dynamo wire schema. The framework
ledger stays application-owned. Correlation uses the opaque application
headers documented by the RL operations guide and request_id between Dynamo's
request_payload and request_end records. The output deliberately excludes
captured request and response bodies.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

TRACE_SCHEMA = "dynamo.request.trace.v1"
ROLLOUT_HEADER = "x-rl-rollout-id"
ATTEMPT_HEADER = "x-rl-attempt-id"
POLICY_HEADER = "x-rl-policy-version"


class JoinError(ValueError):
    """Raised when an input cannot produce an unambiguous correlation join."""


def _open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_jsonl(paths: Iterable[Path], label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise JoinError(f"{label}: {path} does not exist")
        with _open_text(path) as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise JoinError(
                        f"{label}: invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise JoinError(
                        f"{label}: row at {path}:{line_number} must be an object"
                    )
                records.append(record)
    return records


def _required_string(record: dict[str, Any], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise JoinError(f"{label}: {field} must be a non-empty string")
    return value


def _attempt_key(record: dict[str, Any], label: str) -> tuple[str, str]:
    return (
        _required_string(record, "rollout_id", label),
        _required_string(record, "attempt_id", label),
    )


def _normalise_headers(value: Any, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise JoinError(f"{label}: http_request_headers must be an object")
    headers: dict[str, str] = {}
    for key, header_value in value.items():
        if not isinstance(key, str) or not isinstance(header_value, str):
            raise JoinError(
                f"{label}: captured header names and values must be strings"
            )
        headers[key.lower()] = header_value
    return headers


def _index_framework_attempts(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    attempts: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        label = f"framework attempt {index}"
        key = _attempt_key(record, label)
        if key in attempts:
            raise JoinError(
                f"{label}: duplicate rollout_id/attempt_id {key[0]!r}/{key[1]!r}"
            )
        terminal_expected = record.get("terminal_expected", True)
        if not isinstance(terminal_expected, bool):
            raise JoinError(f"{label}: terminal_expected must be a boolean")
        target_policy = record.get("target_policy_version")
        if target_policy is not None and (
            not isinstance(target_policy, str) or not target_policy
        ):
            raise JoinError(
                f"{label}: target_policy_version must be a non-empty string or null"
            )
        attempts[key] = record
    return attempts


def _trace_rows(
    records: Iterable[dict[str, Any]],
) -> tuple[
    dict[str, tuple[dict[str, Any], dict[str, str]]],
    dict[str, dict[str, Any]],
]:
    payloads: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        label = f"request trace row {index}"
        if record.get("schema") != TRACE_SCHEMA:
            raise JoinError(f"{label}: expected schema {TRACE_SCHEMA!r}")
        event_type = record.get("event_type")
        if event_type == "request_payload":
            payload = record.get("payload")
            if not isinstance(payload, dict):
                raise JoinError(f"{label}: request_payload is missing payload")
            request_id = _required_string(payload, "request_id", label)
            if request_id in payloads:
                raise JoinError(f"{label}: duplicate payload request_id {request_id!r}")
            headers = _normalise_headers(payload.get("http_request_headers"), label)
            payloads[request_id] = (record, headers)
        elif event_type == "request_end":
            request = record.get("request")
            if not isinstance(request, dict):
                raise JoinError(f"{label}: request_end is missing request")
            request_id = _required_string(request, "request_id", label)
            if request_id in terminals:
                raise JoinError(
                    f"{label}: duplicate terminal request_id {request_id!r}"
                )
            terminals[request_id] = record
    return payloads, terminals


def _payload_attempt_key(headers: dict[str, str]) -> tuple[str, str] | None:
    rollout_id = headers.get(ROLLOUT_HEADER)
    attempt_id = headers.get(ATTEMPT_HEADER)
    if not rollout_id or not attempt_id:
        return None
    return rollout_id, attempt_id


def join_records(
    framework_records: Iterable[dict[str, Any]],
    trace_records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    attempts = _index_framework_attempts(framework_records)
    payloads, terminals = _trace_rows(trace_records)

    payload_by_attempt: dict[
        tuple[str, str], tuple[str, dict[str, Any], dict[str, str]]
    ] = {}
    uncorrelated_payloads = 0
    for request_id, (payload_record, headers) in payloads.items():
        key = _payload_attempt_key(headers)
        if key is None or key not in attempts:
            uncorrelated_payloads += 1
            continue
        if key in payload_by_attempt:
            prior = payload_by_attempt[key][0]
            raise JoinError(
                f"multiple Dynamo request IDs map to framework attempt {key[0]!r}/{key[1]!r}: {prior!r}, {request_id!r}"
            )
        payload_by_attempt[key] = (request_id, payload_record, headers)

    joined: list[dict[str, Any]] = []
    counters = {
        "framework_attempts": len(attempts),
        "payload_records": len(payloads),
        "terminal_records": len(terminals),
        "joined_payloads": 0,
        "joined_terminals": 0,
        "complete": 0,
        "expected_incomplete": 0,
        "missing_payload": 0,
        "missing_terminal": 0,
        "unexpected_terminal": 0,
        "policy_mismatches": 0,
        "uncorrelated_payloads": uncorrelated_payloads,
        "orphan_terminals": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "strict_violations": 0,
    }
    joined_request_ids: set[str] = set()

    for key, framework in attempts.items():
        rollout_id, attempt_id = key
        payload_match = payload_by_attempt.get(key)
        request_id = payload_match[0] if payload_match else None
        headers = payload_match[2] if payload_match else {}
        terminal = terminals.get(request_id) if request_id else None
        terminal_expected = framework.get("terminal_expected", True)
        target_policy = framework.get("target_policy_version")
        observed_policy = headers.get(POLICY_HEADER)
        policy_matches = (
            None if target_policy is None else observed_policy == target_policy
        )

        if payload_match is None:
            join_status = "missing_payload"
            counters["missing_payload"] += 1
            counters["strict_violations"] += 1
        else:
            counters["joined_payloads"] += 1
            assert request_id is not None
            joined_request_ids.add(request_id)
            if terminal is None and terminal_expected:
                join_status = "missing_terminal"
                counters["missing_terminal"] += 1
                counters["strict_violations"] += 1
            elif terminal is None:
                join_status = "expected_incomplete"
                counters["expected_incomplete"] += 1
            elif not terminal_expected:
                join_status = "unexpected_terminal"
                counters["unexpected_terminal"] += 1
                counters["joined_terminals"] += 1
                counters["strict_violations"] += 1
            else:
                join_status = "complete"
                counters["complete"] += 1
                counters["joined_terminals"] += 1

        if policy_matches is False:
            counters["policy_mismatches"] += 1
            counters["strict_violations"] += 1

        compact_terminal = None
        if terminal is not None:
            request = terminal["request"]
            input_tokens = request.get("input_tokens")
            output_tokens = request.get("output_tokens")
            if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
                counters["input_tokens"] += input_tokens
            if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
                counters["output_tokens"] += output_tokens
            compact_terminal = {
                "event_time_unix_ms": terminal.get("event_time_unix_ms"),
                "agent_context": terminal.get("agent_context"),
                "request": request,
            }

        joined.append(
            {
                "framework": framework,
                "dynamo_request_id": request_id,
                "join_status": join_status,
                "observed_policy_version": observed_policy,
                "policy_header_matches": policy_matches,
                "request_end": compact_terminal,
            }
        )

    counters["orphan_terminals"] = len(set(terminals) - joined_request_ids)
    return joined, counters


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-attempts", type=Path, required=True)
    parser.add_argument("--request-trace", type=Path, nargs="+", required=True)
    parser.add_argument("--joined-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail for missing joins/required terminals, unexpected terminals, or policy mismatch",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        framework = load_jsonl([args.framework_attempts], "framework attempts")
        trace = load_jsonl(args.request_trace, "request trace")
        joined, summary = join_records(framework, trace)
    except JoinError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.joined_jsonl:
        _write_jsonl(args.joined_jsonl, joined)
    if args.summary_json:
        _write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict and summary["strict_violations"]:
        print(
            f"ERROR: strict correlation checks found {summary['strict_violations']} violation(s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
