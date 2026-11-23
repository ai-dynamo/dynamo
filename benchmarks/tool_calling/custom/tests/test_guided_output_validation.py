# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay #18527's completed responses through the Custom verifier."""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tool_calling_probe as probe  # noqa: E402

from benchmarks.tool_calling.custom_runner import apply_request_contract  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

CASE_IDS = (
    "regression_guided_json_answer_in_content",
    "regression_guided_choice_answer_in_content",
    "regression_guided_regex_answer_in_content",
    "regression_guided_grammar_answer_in_content",
    "regression_response_format_answer_in_content",
)


def configured_case(case_id):
    target = next(c for c in probe.build_cases("glm5") if c.case_id == case_id)
    return apply_request_contract(
        (target,),
        {
            "enabled": {"thinking": True, "enable_thinking": True},
            "capabilities": ["guided_reasoning_three_parsers"],
            "parser_configuration": target.required_parser_configuration,
        },
    )[0]


def answer_for(target):
    return (
        json.dumps(target.expected_json)
        if target.expected_json is not None
        else "Alice"
    )


def usage(completion=13, reasoning=4):
    return {
        "prompt_tokens": 27,
        "completion_tokens": completion,
        "total_tokens": 27 + completion,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }


def response_wire(content, reasoning, counts, mode):
    message = {"content": content, "reasoning_content": reasoning}
    if mode == "nonstream":
        raw = {
            "id": "chatcmpl_cpu_replay",
            "model": "cpu",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": counts,
        }
        return json.dumps(raw).encode()
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": reasoning},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": counts},
    ]
    return (
        "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    ).encode()


def run_replay(monkeypatch, target, content, reasoning, counts, mode):
    requests = []

    def post(url, payload, **kwargs):
        requests.append(payload)
        return io.BytesIO(response_wire(content, reasoning, counts, mode))

    monkeypatch.setattr(probe, "post_json", post)
    record = probe.run_case(
        target,
        mode,
        iteration=1,
        url="unused",
        api_key=None,
        model="cpu",
        temperature=0,
        max_tokens=4096,
        timeout=1,
        extra_headers={},
        raw_chars=20000,
        record_success_raw=True,
    )
    return record, requests


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("mode", ("nonstream", "stream"))
def test_completed_answer_only_in_reasoning_fails_compatibility(
    monkeypatch, case_id, mode
):
    target = configured_case(case_id)
    answer = answer_for(target)
    # GLM's captured scalar failure was 1 reasoning / 2 completion tokens.
    counts = usage(13, 13) if target.expected_json is not None else usage(2, 1)
    record, requests = run_replay(monkeypatch, target, None, answer, counts, mode)
    assert record["pass"] is False
    assert record["failure_category"] == "dynamo_api"
    assert "reasoning_only_response" in record["failure_kinds"]
    assert record["response"]["finish_reason"] == "stop"
    assert requests[0]["max_tokens"] == 256
    assert (
        not {"temperature", "chat_template_kwargs", "tools", "tool_choice"}
        & requests[0].keys()
    )


@pytest.mark.parametrize("mode", ("nonstream", "stream"))
@pytest.mark.parametrize(
    "thought_kind", ("none", "quoted_answer", "mixed_chunk_estimate")
)
def test_visible_final_answer_accepts_supported_reasoning_shapes(
    monkeypatch, mode, thought_kind
):
    target = configured_case("regression_response_format_answer_in_content")
    answer = answer_for(target)
    reasoning = "" if thought_kind == "none" else f"The requested object is {answer}."
    counts = usage(13, 0) if not reasoning else usage(13, 4)
    if thought_kind == "mixed_chunk_estimate":
        counts = usage(4, 4)
    record, _ = run_replay(monkeypatch, target, answer, reasoning, counts, mode)
    assert record["pass"], record["errors"]
    warnings = {entry["kind"] for entry in record["warnings"]}
    assert ("reasoning_usage_estimate" in warnings) is (
        thought_kind == "mixed_chunk_estimate"
    )


@pytest.mark.parametrize("reasoning_tokens", (None, True, -1, 0.5, 14))
def test_reasoning_usage_requires_valid_bounded_integers(reasoning_tokens):
    target = configured_case("regression_guided_json_answer_in_content")
    result = probe.ChatResult(
        content=answer_for(target),
        finish_reason="stop",
        usage=usage(13, reasoning_tokens),
    )
    errors, _ = probe.validate_result(target, result)
    assert "usage_inconsistent" in {entry["kind"] for entry in errors}
    assert probe.classify_failure(target, result, errors) == "dynamo_api"


def test_malformed_usage_is_reported_instead_of_crashing():
    target = configured_case("regression_guided_json_answer_in_content")
    result = probe.ChatResult(
        content=answer_for(target), finish_reason="stop", usage=["invalid"]
    )
    errors, _ = probe.validate_result(target, result)
    assert "usage_missing" in {entry["kind"] for entry in errors}
    assert probe.classify_failure(target, result, errors) == "dynamo_api"


def test_truncated_thought_is_not_mislabeled_as_completed_routing_failure():
    target = configured_case("regression_guided_json_answer_in_content")
    result = probe.ChatResult(
        reasoning_content="I should prepare an object",
        finish_reason="length",
        usage=usage(13, 13),
    )
    errors, _ = probe.validate_result(target, result)
    assert "reasoning_only_response" not in {entry["kind"] for entry in errors}
    assert probe.classify_failure(target, result, errors) == "model_quality"


@pytest.mark.parametrize(
    "configuration",
    (
        {},
        {"chat_processor": "sglang"},
        {
            "chat_processor": "dynamo",
            "dynamo_reasoning_parser": "glm45",
            "native_reasoning_parser": "glm45",
        },
        {
            "chat_processor": "dynamo",
            "dynamo_reasoning_parser": "glm45",
            "dynamo_tool_call_parser": "glm47",
            "native_reasoning_parser": "wrong",
        },
    ),
)
def test_unverified_parser_configuration_is_inconclusive_without_dispatch(
    monkeypatch, configuration
):
    target = configured_case("regression_guided_json_answer_in_content")
    target = dataclasses.replace(
        target, runtime_provenance={"parser_configuration": configuration}
    )

    def no_dispatch(*args, **kwargs):
        pytest.fail("the configured three-parser path was not verified")

    monkeypatch.setattr(probe, "post_json", no_dispatch)
    record = probe.run_case(
        target,
        "nonstream",
        iteration=1,
        url="unused",
        api_key=None,
        model="cpu",
        temperature=0,
        max_tokens=4096,
        timeout=1,
        extra_headers={},
        raw_chars=1000,
        record_success_raw=True,
    )
    assert record["pass"] is False
    assert record["failure_category"] == "unclassified"
    assert record["failure_kinds"] == ["capability_unavailable"]
    assert record["errors"][0]["details"]["parser_configuration_mismatches"]
