# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.agent_harness.nightly.validate_request_trace import (
    ValidationConfig,
    load_records,
    validate_records,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
]


def _request(
    session_id: str,
    trigger: str,
    *,
    parent_session_id: str | None = None,
    model: str = "agent-nightly",
    tool_calls: int = 0,
) -> dict:
    agent_context = {"session_id": session_id, "input_trigger": trigger}
    if parent_session_id:
        agent_context["parent_session_id"] = parent_session_id
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "agent_context": agent_context,
        "request": {
            "model": model,
            "input_tokens": 100,
            "output_tokens": 20,
            "finish_reason_metadata": {
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "tool_calls": [{} for _ in range(tool_calls)],
            },
        },
    }


def test_validates_multiple_long_root_sessions() -> None:
    records = []
    for session_id in ("claude-1", "claude-2"):
        records.extend(
            [
                _request(session_id, "user_message", tool_calls=1),
                _request(session_id, "tool_result", tool_calls=1),
                _request(session_id, "tool_result"),
                _request(session_id, "other"),
            ]
        )

    summary, errors = validate_records(
        records,
        ValidationConfig(
            expected_model="agent-nightly",
            minimum_root_sessions=2,
            minimum_requests_per_session=4,
        ),
    )

    assert errors == []
    assert summary["ok"] is True
    assert summary["qualifying_root_session_count"] == 2
    assert summary["sessions"][0]["input_tokens"] == 400
    assert summary["sessions"][0]["tool_call_count"] == 2


def test_rejects_long_session_without_tool_result() -> None:
    records = [
        _request("codex-1", "user_message"),
        _request("codex-1", "other"),
        _request("codex-1", "other"),
        _request("codex-1", "other"),
    ]

    summary, errors = validate_records(
        records,
        ValidationConfig(
            expected_model="agent-nightly",
            minimum_root_sessions=1,
            minimum_requests_per_session=4,
        ),
    )

    assert summary["ok"] is False
    assert summary["qualifying_root_session_count"] == 0
    assert any("tool_result" in error for error in errors)


def test_child_session_does_not_satisfy_root_session_budget() -> None:
    records = [
        _request("child", "user_message", parent_session_id="root"),
        _request("child", "tool_result", parent_session_id="root"),
    ]

    summary, errors = validate_records(
        records,
        ValidationConfig(
            expected_model="agent-nightly",
            minimum_root_sessions=1,
            minimum_requests_per_session=2,
        ),
    )

    assert summary["root_session_count"] == 0
    assert errors


def test_load_records_honors_cursor_and_event_wrapper(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(_request("old", "user_message"))
        + "\n"
        + json.dumps({"event": _request("new", "tool_result")})
        + "\n"
    )

    records = load_records(path, start_line=1)

    assert len(records) == 1
    assert records[0]["agent_context"]["session_id"] == "new"


def test_model_filter_prevents_cross_model_false_positive() -> None:
    records = [
        _request("wrong-model", "user_message", model="other"),
        _request("wrong-model", "tool_result", model="other"),
    ]

    summary, errors = validate_records(
        records,
        ValidationConfig(
            expected_model="agent-nightly",
            minimum_root_sessions=1,
            minimum_requests_per_session=2,
        ),
    )

    assert summary["session_count"] == 0
    assert errors


def test_load_records_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"event_type":"request_end"\n')

    with pytest.raises(ValueError, match="invalid request-trace JSON"):
        load_records(path)
