# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tool_calling_probe as probe  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("mode", ("nonstream", "stream"))
@pytest.mark.parametrize("alias", ("reasoning", "reasoning_content"))
@pytest.mark.parametrize("provider", (False, True))
def test_reasoning_aliases_preserve_raw_response_and_finish_reason(
    monkeypatch, mode, alias, provider
):
    fields = {alias: "Need to call the calculator."}
    if provider:
        fields = {"provider_specific_fields": fields}
    fields.update({"content": "", "tool_calls": []})
    key = "delta" if mode == "stream" else "message"
    raw = {"choices": [{key: fields, "finish_reason": "stop"}]}
    body = json.dumps(raw)
    if mode == "stream":
        body = f"data: {body}\n\ndata: [DONE]\n\n"
    monkeypatch.setattr(probe, "post_json", lambda *a, **k: io.BytesIO(body.encode()))
    reader = probe.request_stream if mode == "stream" else probe.request_nonstream
    result = reader("unused", {}, api_key=None, timeout=1, extra_headers={})

    assert result.reasoning_content == "Need to call the calculator."
    assert result.finish_reason == "stop"
    assert result.parse_errors == []
    assert result.raw_response == ([raw] if mode == "stream" else raw)


def test_streaming_aliases_are_not_duplicated_and_can_alternate(monkeypatch):
    deltas = [
        {"reasoning": "First ", "reasoning_content": "First "},
        {"reasoning": "then "},
        {
            "reasoning_content": "last.",
            "provider_specific_fields": {"reasoning": "last."},
        },
    ]
    raw = [{"choices": [{"delta": delta}]} for delta in deltas]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in raw)
    monkeypatch.setattr(probe, "post_json", lambda *a, **k: io.BytesIO(body.encode()))
    result = probe.request_stream(
        "unused", {}, api_key=None, timeout=1, extra_headers={}
    )
    assert result.reasoning_content == "First then last."
    assert result.parse_errors == []


def test_conflicting_reasoning_fields_still_fail():
    errors = []
    assert (
        probe.response_reasoning(
            {"reasoning_content": "one", "reasoning": "different"}, errors
        )
        == "one"
    )
    assert [finding["kind"] for finding in errors] == ["reasoning_mismatch"]


def test_empty_primary_reasoning_falls_back_to_alias():
    errors = []
    assert (
        probe.response_reasoning(
            {"reasoning_content": None, "reasoning": "native reasoning"}, errors
        )
        == "native reasoning"
    )
    assert errors == []


def test_alias_support_does_not_relax_finish_reason_or_truncated_argument_checks():
    case = next(
        case
        for case in probe.build_cases("deepseek_v4")
        if case.case_id == "customer_named_calculator_choice"
    )
    result = probe.ChatResult(
        finish_reason="stop",
        reasoning_content=probe.response_reasoning(
            {"reasoning": "Use calculator."}, []
        ),
        tool_calls=[
            {
                "id": "call",
                "type": "function",
                "function": {"name": "calculate_sum", "arguments": '{"numbers":['},
            }
        ],
    )
    errors, _ = probe.validate_result(case, result)
    kinds = {finding["kind"] for finding in errors}
    assert "missing_reasoning_content" not in kinds
    assert "unexpected_finish_reason" in kinds
    assert "invalid_arguments_json" in kinds
