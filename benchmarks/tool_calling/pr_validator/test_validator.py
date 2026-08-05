# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from .validator import (
    Case,
    NormalizedResponse,
    build_cases,
    normalize_stream,
    validate_response,
)


def test_matrix_has_eleven_standard_and_one_structural_tag_case() -> None:
    standard = build_cases("either", structural_tag_deployment=False)
    structural = build_cases("either", structural_tag_deployment=True)

    assert [case.case_id for case in standard] == [f"{i:02d}" for i in range(1, 12)]
    assert [case.case_id for case in structural] == [f"{i:02d}" for i in range(1, 13)]
    assert structural[-1].structural_tag_only
    assert structural[-1].parallel_tool_calls is False


def test_stream_normalization_reconstructs_reasoning_and_tool_arguments() -> None:
    events = [
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "Need weather. "},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Par',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": 'is","unit":"celsius"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    raw = "\n\n".join(f"data: {json.dumps(event)}" for event in events)
    raw += "\n\ndata: [DONE]\n"

    response = normalize_stream(raw)

    assert response.reasoning_content == "Need weather. "
    assert response.finish_reason == "tool_calls"
    assert response.done
    assert json.loads(response.tool_calls[0]["function"]["arguments"]) == {
        "city": "Paris",
        "unit": "celsius",
    }


def test_tool_validation_accepts_separated_reasoning_and_expected_call() -> None:
    case = Case(
        "04",
        "required",
        [],
        expected_reasoning="present",
        expected_tool=True,
    )
    response = NormalizedResponse(
        reasoning_content="I should use the weather tool.",
        tool_calls=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Paris","unit":"celsius"}',
                },
            }
        ],
        finish_reason="tool_calls",
        done=True,
    )

    assert validate_response(case, response) == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            NormalizedResponse(
                reasoning_content='[{"name":"get_weather"}]',
                finish_reason="stop",
            ),
            "raw guided JSON",
        ),
        (
            NormalizedResponse(
                content="<think>hidden</think>",
                finish_reason="stop",
            ),
            "parser marker",
        ),
    ],
)
def test_validation_rejects_guided_json_and_marker_leaks(
    response: NormalizedResponse, message: str
) -> None:
    case = Case("01", "direct", [], expected_content=True)

    assert any(message in error for error in validate_response(case, response))


def test_thinking_disabled_rejects_reasoning() -> None:
    case = Case(
        "09",
        "disabled",
        [],
        expected_reasoning="absent",
        expected_tool=True,
    )
    response = NormalizedResponse(
        reasoning_content="This should not be present.",
        tool_calls=[
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Paris", "unit": "celsius"},
                }
            }
        ],
        finish_reason="tool_calls",
    )

    errors = validate_response(case, response)

    assert "thinking-disabled request returned reasoning_content" in errors
