#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact SWE-Gym agent/subagent trace fidelity checks shared by both gates."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from enforce_root_final import POST_AGENT_CONTEXT
from prepare_swegym_task import (
    BASH_COMMAND,
    CHILD_SUCCESS_HANDOFF,
    ROOT_SUCCESS_FINAL,
)

EXPECTED_TOOL_SEQUENCE = ["Agent", "Bash"]
SUCCESS_SENTINEL = "__M6_BASH_VALIDATION_PASS__"
MODEL_NAME = "Qwen/Qwen3-30B-A3B"


def sample_field(sample: Any, name: str) -> Any:
    return sample[name] if isinstance(sample, dict) else getattr(sample, name)


def polar_metadata(sample: Any) -> dict[str, Any]:
    metadata = sample_field(sample, "metadata")
    polar = metadata["polar"]
    assert isinstance(polar, dict), polar
    return polar


def decoded_tool_calls(sample: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    debug = polar_metadata(sample)["trace_debug"]
    for message in debug.get("response_messages", []):
        for call in message.get("tool_calls", []):
            function = call.get("function") or {}
            name = function.get("name")
            raw_arguments = function.get("arguments", {})
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            assert isinstance(arguments, dict), (name, arguments)
            calls.append(
                {"id": str(call.get("id", "")), "name": str(name), "arguments": arguments}
            )
    return calls


def tool_names(sample: Any) -> list[str]:
    return [call["name"] for call in decoded_tool_calls(sample)]


def trace_key(sample: Any) -> tuple[str, int]:
    polar = polar_metadata(sample)
    return str(polar["session_id"]), int(polar["trace_index"])


def _call_identity(calls: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(call.get("id", "")), str(call.get("name", ""))) for call in calls)


def sample_request_signature(sample: Any) -> tuple[int, int, str, tuple[tuple[str, str], ...]]:
    tokens = sample_field(sample, "tokens")
    response_length = int(sample_field(sample, "response_length"))
    finish_reason = str(polar_metadata(sample)["trace_debug"]["finish_reason"])
    return (
        len(tokens) - response_length,
        response_length,
        finish_reason,
        _call_identity(decoded_tool_calls(sample)),
    )


def request_record_signature(
    record: dict[str, Any],
) -> tuple[int, int, str, tuple[tuple[str, str], ...]] | None:
    event = record.get("event")
    if not isinstance(event, dict) or event.get("event_type") != "request_end":
        return None
    request = event.get("request")
    context = event.get("agent_context")
    if not isinstance(request, dict) or not isinstance(context, dict):
        return None
    if not context.get("session_id"):
        return None
    finish = request.get("finish_reason_metadata")
    if not isinstance(finish, dict):
        return None
    calls = finish.get("tool_calls") or []
    if not isinstance(calls, list):
        return None
    call_identity = tuple(
        (str(call.get("id", "")), str(call.get("name", "")))
        for call in calls
        if isinstance(call, dict)
    )
    return (
        int(request["input_tokens"]),
        int(request["output_tokens"]),
        str(finish["finish_reason"]),
        call_identity,
    )


def snapshot_trace_line_counts(trace_dir: Path) -> dict[str, int]:
    return {
        str(path): len(path.read_text(errors="replace").splitlines())
        for path in sorted(trace_dir.rglob("*.jsonl"))
    }


def load_trace_snapshot(path: Path) -> dict[str, int]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), value
    return {str(key): int(count) for key, count in value.items()}


def _fresh_trace_records(
    trace_dir: Path, start_line_counts: dict[str, int]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(trace_dir.rglob("*.jsonl")):
        lines = path.read_text(errors="replace").splitlines()
        for line in lines[start_line_counts.get(str(path), 0) :]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def match_samples_to_agent_contexts(
    samples: list[Any], records: list[dict[str, Any]]
) -> tuple[dict[tuple[str, int], dict[str, str]], list[dict[str, Any]]]:
    by_signature: dict[
        tuple[int, int, str, tuple[tuple[str, str], ...]], list[dict[str, Any]]
    ] = defaultdict(list)
    for record in records:
        signature = request_record_signature(record)
        if signature is not None:
            by_signature[signature].append(record)

    contexts: dict[tuple[str, int], dict[str, str]] = {}
    matched_records: list[dict[str, Any]] = []
    for sample in samples:
        key = trace_key(sample)
        assert key not in contexts, key
        signature = sample_request_signature(sample)
        candidates = by_signature.get(signature, [])
        assert len(candidates) == 1, (key, signature, len(candidates))
        record = candidates.pop()
        event = record["event"]
        request = event["request"]
        assert event["schema"] == "dynamo.request.trace.v1", event
        assert event["event_source"] == "dynamo", event
        assert request["model"] == MODEL_NAME, request["model"]
        context = event["agent_context"]
        contexts[key] = {
            str(name): str(value)
            for name, value in context.items()
            if value is not None
        }
        matched_records.append(
            {
                "polar_session_id": key[0],
                "polar_trace_index": key[1],
                "request_id": str(request["request_id"]),
                "input_tokens": int(request["input_tokens"]),
                "output_tokens": int(request["output_tokens"]),
                "finish_reason": signature[2],
                "tool_calls": [list(call) for call in signature[3]],
                "agent_context": contexts[key],
            }
        )
    assert len({record["request_id"] for record in matched_records}) == len(samples)
    return contexts, matched_records


def load_and_match_trace_records(
    trace_dir: Path,
    start_line_counts: dict[str, int],
    samples: list[Any],
    timeout: float = 60.0,
) -> tuple[
    dict[tuple[str, int], dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    deadline = time.monotonic() + timeout
    last_records: list[dict[str, Any]] = []
    last_error: AssertionError | None = None
    while time.monotonic() < deadline:
        last_records = _fresh_trace_records(trace_dir, start_line_counts)
        try:
            contexts, matched = match_samples_to_agent_contexts(samples, last_records)
        except AssertionError as exc:
            last_error = exc
            time.sleep(1.0)
            continue
        return contexts, matched, last_records
    raise AssertionError(
        f"could not join {len(samples)} samples to fresh Dynamo requests; "
        f"records={len(last_records)} last_error={last_error}"
    )


def _prompt_messages(sample: Any) -> list[dict[str, Any]]:
    prompt = sample_field(sample, "prompt")
    assert isinstance(prompt, list), type(prompt)
    return prompt


def validate_session_fidelity(
    session_id: str,
    traces: list[Any],
    contexts: dict[tuple[str, int], dict[str, str]],
) -> dict[str, Any]:
    ordered = sorted(traces, key=lambda trace: trace_key(trace)[1])
    calls = [call for trace in ordered for call in decoded_tool_calls(trace)]
    names = [call["name"] for call in calls]
    assert names == EXPECTED_TOOL_SEQUENCE, (session_id, names)
    call_ids = [call["id"] for call in calls]
    assert all(call_ids), (session_id, call_ids)
    assert len(set(call_ids)) == len(call_ids), (session_id, call_ids)
    agent_args, bash_args = [call["arguments"] for call in calls]

    assert agent_args.get("run_in_background") in (None, False), agent_args
    assert agent_args.get("isolation") in (None, ""), agent_args
    delegation = str(agent_args.get("prompt", ""))
    assert agent_args.get("subagent_type") == "general-purpose", agent_args
    assert delegation.strip(), agent_args
    delegation_lower = delegation.lower()
    assert all(
        anchor in delegation_lower
        for anchor in (
            "dynamodb",
            "dynamotype.__add__",
            "__sub__",
            "cast_value",
            "decimal",
            "integer branch",
            "item add",
        )
    ), delegation
    assert bash_args.get("command") == BASH_COMMAND, bash_args

    response_blob = json.dumps(
        [
            message
            for trace in ordered
            for message in polar_metadata(trace)["trace_debug"].get("response_messages", [])
        ],
        ensure_ascii=False,
    )
    assert "<tool_call>" not in response_blob, session_id

    root_ids: set[str] = set()
    child_links: set[tuple[str, str]] = set()
    root_calls: list[dict[str, Any]] = []
    child_calls: list[dict[str, Any]] = []
    root_finishes: list[str] = []
    child_finishes: list[str] = []
    root_trace_entries: list[dict[str, Any]] = []
    child_trace_entries: list[dict[str, Any]] = []
    for trace in ordered:
        key = trace_key(trace)
        context = contexts[key]
        context_id = context["session_id"]
        parent_id = context.get("parent_session_id")
        trace_calls = decoded_tool_calls(trace)
        finish_reason = str(polar_metadata(trace)["trace_debug"]["finish_reason"])
        if parent_id:
            child_links.add((context_id, parent_id))
            child_calls.extend(trace_calls)
            child_finishes.append(finish_reason)
            child_trace_entries.append(
                {
                    "trace_index": key[1],
                    "finish_reason": finish_reason,
                    "calls": trace_calls,
                    "prompt": _prompt_messages(trace),
                    "response": polar_metadata(trace)["trace_debug"].get(
                        "response_messages", []
                    ),
                }
            )
        else:
            root_ids.add(context_id)
            root_calls.extend(trace_calls)
            root_finishes.append(finish_reason)
            root_trace_entries.append(
                {
                    "trace_index": key[1],
                    "finish_reason": finish_reason,
                    "calls": trace_calls,
                    "prompt": _prompt_messages(trace),
                    "response": polar_metadata(trace)["trace_debug"].get(
                        "response_messages", []
                    ),
                }
            )

    assert len(root_ids) == 1, (session_id, root_ids)
    assert len(child_links) == 1, (session_id, child_links)
    root_id = next(iter(root_ids))
    child_id, parent_id = next(iter(child_links))
    assert parent_id == root_id, (session_id, root_id, child_links)
    assert child_id != root_id, (session_id, root_id, child_id)
    assert [call["name"] for call in root_calls] == ["Agent"], root_calls
    assert [call["name"] for call in child_calls] == ["Bash"], child_calls
    assert root_finishes == ["tool_calls", "stop"], (
        session_id,
        root_finishes,
    )
    assert child_finishes == ["tool_calls", "stop"], (
        session_id,
        child_finishes,
    )
    assert [
        [call["name"] for call in entry["calls"]] for entry in root_trace_entries
    ] == [["Agent"], []], root_trace_entries
    assert [
        [call["name"] for call in entry["calls"]] for entry in child_trace_entries
    ] == [["Bash"], []], child_trace_entries
    child_system_prompt = json.dumps(
        [
            message
            for message in child_trace_entries[0]["prompt"]
            if message.get("role") == "system"
        ],
        ensure_ascii=False,
    )
    for required in (
        BASH_COMMAND,
        SUCCESS_SENTINEL,
        "Do not invoke any other command or tool",
    ):
        assert required in child_system_prompt, (session_id, required, child_system_prompt)

    bash_call_id = calls[-1]["id"]
    bash_trace_index = child_trace_entries[0]["trace_index"]
    assert bash_trace_index < child_trace_entries[-1]["trace_index"], (
        session_id,
        bash_trace_index,
        child_trace_entries[-1]["trace_index"],
    )
    assert child_trace_entries[-1]["trace_index"] < root_trace_entries[-1][
        "trace_index"
    ], (
        session_id,
        child_trace_entries[-1]["trace_index"],
        root_trace_entries[-1]["trace_index"],
    )
    premature_bash_results = [
        message
        for entry in child_trace_entries
        if entry["trace_index"] <= bash_trace_index
        for message in entry["prompt"]
        if message.get("role") == "tool"
        and str(message.get("tool_call_id", "")) == bash_call_id
    ]
    assert not premature_bash_results, (session_id, premature_bash_results)
    bash_results = [
        message
        for entry in child_trace_entries
        if entry["trace_index"] > bash_trace_index
        for message in entry["prompt"]
        if message.get("role") == "tool"
        and str(message.get("tool_call_id", "")) == bash_call_id
    ]
    assert len(bash_results) == 1, (session_id, bash_call_id, bash_results)
    assert SUCCESS_SENTINEL in str(bash_results[0].get("content", "")), bash_results[0]

    child_stop_messages = [
        message
        for message in child_trace_entries[-1]["response"]
        if message.get("role") == "assistant"
    ]
    assert len(child_stop_messages) == 1, (session_id, child_stop_messages)
    assert str(child_stop_messages[0].get("content", "")).strip() == (
        CHILD_SUCCESS_HANDOFF
    ), child_stop_messages[0]

    agent_call_id = calls[0]["id"]
    agent_results = [
        message
        for message in root_trace_entries[-1]["prompt"]
        if message.get("role") == "tool"
        and str(message.get("tool_call_id", "")) == agent_call_id
    ]
    assert len(agent_results) == 1, (session_id, agent_call_id, agent_results)
    assert CHILD_SUCCESS_HANDOFF in str(agent_results[0].get("content", "")), (
        session_id,
        agent_results[0],
    )
    root_prompt_blob = json.dumps(
        root_trace_entries[-1]["prompt"], ensure_ascii=False
    )
    assert POST_AGENT_CONTEXT in root_prompt_blob, (session_id, root_prompt_blob)

    root_stop_messages = [
        message
        for message in root_trace_entries[-1]["response"]
        if message.get("role") == "assistant"
    ]
    assert len(root_stop_messages) == 1, (session_id, root_stop_messages)
    assert str(root_stop_messages[0].get("content", "")).strip() == (
        ROOT_SUCCESS_FINAL
    ), root_stop_messages[0]

    return {
        "tool_names": names,
        "root_agent_session_id": root_id,
        "child_agent_session_id": child_id,
        "child_parent_session_id": parent_id,
        "root_finish_reasons": root_finishes,
        "child_finish_reasons": child_finishes,
        "root_final_stop_observed": True,
        "root_bash_guarded": True,
        "subagent_type": "general-purpose",
        "child_custom_policy_loaded": True,
        "bash_tool_call_id": bash_call_id,
        "bash_validation_sentinel": True,
        "child_success_handoff_exact": True,
        "post_agent_context_observed": True,
        "root_success_final_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--trace-dir", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        value = snapshot_trace_line_counts(args.trace_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
