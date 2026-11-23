# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone guided-answer regressions for the customer three-parser setup."""

from __future__ import annotations

import copy
from typing import Any, Callable

_PROFILES = ("glm5", "glm47", "minimax_m2")
_CASE_IDS = (
    "regression_guided_json_answer_in_content",
    "regression_guided_choice_answer_in_content",
    "regression_guided_regex_answer_in_content",
    "regression_guided_grammar_answer_in_content",
    "regression_response_format_answer_in_content",
)
_REFS = (
    "https://github.com/ai-dynamo/dynamo/issues/18527",
    "https://github.com/ai-dynamo/dynamo/pull/18718",
    "https://linear.app/nvidia/issue/DGH-2516",
)


def guided_output_case_ids(profile: str) -> tuple[str, ...]:
    """Select the bounded reproduction only for the affected parser families."""

    return _CASE_IDS if profile in (*_PROFILES, "all") else ()


def build_guided_output_cases(
    case_factory: Callable[..., Any], profile: str
) -> tuple[Any, ...]:
    """Keep the customer's default thinking and sampling settings untouched."""

    if not guided_output_case_ids(profile):
        return ()
    parser_configuration = (
        {
            "chat_processor": "dynamo",
            "dynamo_reasoning_parser": "minimax_m2",
            "dynamo_tool_call_parser": "minimax_m2",
            "native_reasoning_parser": "minimax-append-think",
        }
        if profile == "minimax_m2"
        else {
            "chat_processor": "dynamo",
            "dynamo_reasoning_parser": "glm45",
            "dynamo_tool_call_parser": "glm47",
            "native_reasoning_parser": "glm45",
        }
    )
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    json_prompt = "Give me a JSON object with name and age for Alice, 30."
    literal_prompt = "Return exactly Alice."
    definitions = (
        (
            _CASE_IDS[0],
            json_prompt,
            {"guided_json": schema},
            {"name": "Alice", "age": 30},
            None,
        ),
        (
            _CASE_IDS[1],
            literal_prompt,
            {"guided_choice": ["Alice", "Bob"]},
            None,
            "Alice",
        ),
        (
            _CASE_IDS[2],
            literal_prompt,
            {"guided_regex": "Alice"},
            None,
            "Alice",
        ),
        (
            _CASE_IDS[3],
            literal_prompt,
            {"guided_grammar": 'root ::= "Alice"'},
            None,
            "Alice",
        ),
        (
            _CASE_IDS[4],
            json_prompt,
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "p", "schema": schema},
                }
            },
            {"name": "Alice", "age": 30},
            None,
        ),
    )
    return tuple(
        case_factory(
            case_id=case_id,
            description="standalone guided answers reach content with correct reasoning usage",
            messages=({"role": "user", "content": prompt},),
            tools=(),
            tool_choice=None,
            expected_finish_reasons=("stop",),
            expect_no_tool_calls=True,
            min_tool_calls=0,
            exact_tool_calls=0,
            expected_json=copy.deepcopy(expected_json),
            expected_content=expected_content,
            expect_content=True,
            expect_usage=True,
            expect_reasoning_usage=True,
            max_tokens=256,
            preserve_request_defaults=True,
            request_overrides={
                "temperature": None,
                "stream_options": {"include_usage": True},
                **copy.deepcopy(constraints),
            },
            required_capabilities=("guided_reasoning_three_parsers",),
            required_parser_configuration=dict(parser_configuration),
            regression_refs=_REFS,
            profiles=_PROFILES,
        )
        for case_id, prompt, constraints, expected_json, expected_content in definitions
    )
