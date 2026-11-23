# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the customer request shape rather than an easier tool-call path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import tool_calling_probe as probe  # noqa: E402
from custom_runner import apply_request_contract  # noqa: E402
from guided_output_cases import (  # noqa: E402
    build_guided_output_cases,
    guided_output_case_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

_FIELDS = (
    "guided_json",
    "guided_choice",
    "guided_regex",
    "guided_grammar",
    "response_format",
)


@pytest.mark.parametrize("profile", ("glm5", "glm47", "minimax_m2", "all"))
def test_guided_reproduction_is_bounded_to_four_formats_and_one_control(profile):
    cases = build_guided_output_cases(SimpleNamespace, profile)
    assert len(cases) == 5
    assert tuple(case.case_id for case in cases) == guided_output_case_ids(profile)
    assert len({case.case_id for case in cases}) == 5
    for case in cases:
        assert sum(field in case.request_overrides for field in _FIELDS) == 1
        assert case.regression_refs == (
            "https://github.com/ai-dynamo/dynamo/issues/18527",
            "https://github.com/ai-dynamo/dynamo/pull/18718",
            "https://linear.app/nvidia/issue/DGH-2516",
        )


@pytest.mark.parametrize("profile", ("generic", "kimi_k3", "qwen3_coder_xml"))
def test_unrelated_model_profiles_do_not_gain_customer_specific_cases(profile):
    assert build_guided_output_cases(SimpleNamespace, profile) == ()
    assert guided_output_case_ids(profile) == ()


@pytest.mark.parametrize("profile", ("glm5", "glm47", "minimax_m2"))
@pytest.mark.parametrize("stream", (False, True))
def test_resolved_cases_preserve_customer_defaults_through_the_runner(profile, stream):
    selected = {
        case.case_id: case
        for case in probe.build_cases(profile)
        if case.case_id in guided_output_case_ids(profile)
    }
    assert set(selected) == set(guided_output_case_ids(profile))
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
    contracted = apply_request_contract(
        selected.values(),
        {
            "enabled": {"thinking": True, "enable_thinking": True},
            "disabled": {"thinking": False},
            "capabilities": ["guided_reasoning_three_parsers"],
            "parser_configuration": parser_configuration,
        },
    )
    for case in contracted:
        payload = probe.build_payload(
            case,
            model="customer-model",
            stream=stream,
            temperature=0.75,
            default_max_tokens=4096,
        )
        assert payload["max_tokens"] == 256
        assert payload["stream"] is stream
        assert "temperature" not in payload
        assert "chat_template_kwargs" not in payload
        assert "tools" not in payload
        assert "tool_choice" not in payload
        assert "reasoning_effort" not in payload
        assert case.expect_content is True
        assert case.expect_usage is True
        assert case.expect_reasoning_usage is True
        assert case.required_parser_configuration == parser_configuration
        assert case.required_capabilities == ("guided_reasoning_three_parsers",)
        if stream:
            assert payload["stream_options"] == {"include_usage": True}
        else:
            assert "stream_options" not in payload


def test_original_json_reproducer_and_control_use_the_same_schema_and_prompt():
    cases = build_guided_output_cases(SimpleNamespace, "glm5")
    json_case, choice, regex, grammar, control = cases
    assert (
        json_case.messages
        == control.messages
        == (
            {
                "role": "user",
                "content": "Give me a JSON object with name and age for Alice, 30.",
            },
        )
    )
    assert (
        json_case.request_overrides["guided_json"]
        == control.request_overrides["response_format"]["json_schema"]["schema"]
    )
    assert (
        json_case.expected_json == control.expected_json == {"name": "Alice", "age": 30}
    )
    assert choice.request_overrides["guided_choice"] == ["Alice", "Bob"]
    assert regex.request_overrides["guided_regex"] == "Alice"
    assert grammar.request_overrides["guided_grammar"] == 'root ::= "Alice"'
    assert all(case.expected_content == "Alice" for case in (choice, regex, grammar))
