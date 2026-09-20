# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest

from dynamo.llm.exceptions import InvalidArgument
from dynamo.sglang.thinking_budget import (
    apply_thinking_budget,
    extract_thinking_budget,
    thinking_budget_requested,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _server_args(**overrides):
    values = {
        "enable_strict_thinking": True,
        "reasoning_parser": "qwen3",
        "skip_tokenizer_init": False,
        "grammar_backend": "xgrammar",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("request_data", "expected"),
    [
        ({"stop_conditions": {"max_thinking_tokens": 32}}, 32),
        (
            {
                "stop_conditions": {"max_thinking_tokens": None},
                "thinking_token_budget": 32,
            },
            32,
        ),
        ({"thinking_token_budget": 32}, 32),
        ({"thinking_token_budget": 0}, 0),
        ({"nvext": {"max_thinking_tokens": 16}}, 16),
        ({}, None),
    ],
)
def test_extract_thinking_budget(request_data, expected):
    assert extract_thinking_budget(request_data) == expected
    assert thinking_budget_requested(request_data) is (expected is not None)


def test_root_thinking_budget_overrides_legacy_nvext():
    request = {
        "thinking_token_budget": 32,
        "nvext": {"max_thinking_tokens": 16},
    }

    assert extract_thinking_budget(request) == 32


@pytest.mark.parametrize("value", [True, -1, 2**32, 1.5, "32"])
def test_extract_thinking_budget_rejects_invalid_values(value):
    with pytest.raises(InvalidArgument, match="thinking_token_budget"):
        extract_thinking_budget({"thinking_token_budget": value})


def test_apply_thinking_budget_preserves_params_without_mutating_input():
    sampling_params = {
        "temperature": 0.2,
        "custom_params": {"future_engine_control": True, "thinking_budget": 8},
    }
    original = deepcopy(sampling_params)

    actual = apply_thinking_budget(
        {
            "stop_conditions": {"max_thinking_tokens": 32},
            "require_reasoning": True,
        },
        sampling_params,
        _server_args(),
    )

    assert sampling_params == original
    assert actual == {
        "temperature": 0.2,
        "custom_params": {
            "future_engine_control": True,
            "thinking_budget": 32,
        },
    }


def test_apply_thinking_budget_leaves_omitted_budget_unset():
    sampling_params = {"temperature": 0.2}

    actual = apply_thinking_budget({}, sampling_params, _server_args())

    assert actual == sampling_params
    assert actual is not sampling_params


def test_apply_thinking_budget_rejects_forwarded_budget_without_canonical_value():
    with pytest.raises(InvalidArgument, match="requires a canonical"):
        apply_thinking_budget(
            {},
            {"custom_params": {"thinking_budget": 32}},
            _server_args(),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_strict_thinking": False}, "--enable-strict-thinking"),
        ({"reasoning_parser": None}, "--reasoning-parser"),
        ({"skip_tokenizer_init": True}, "--skip-tokenizer-init"),
        ({"grammar_backend": "none"}, "grammar backend"),
    ],
)
def test_apply_thinking_budget_rejects_unsupported_server_config(overrides, message):
    with pytest.raises(InvalidArgument, match=message):
        apply_thinking_budget(
            {
                "stop_conditions": {"max_thinking_tokens": 32},
                "require_reasoning": True,
            },
            {},
            _server_args(**overrides),
        )


def test_apply_thinking_budget_rejects_disabled_request_reasoning():
    with pytest.raises(InvalidArgument, match="requires reasoning to be enabled"):
        apply_thinking_budget(
            {
                "stop_conditions": {"max_thinking_tokens": 32},
                "require_reasoning": False,
            },
            {},
            _server_args(),
        )


def test_apply_thinking_budget_rejects_noncanonical_request():
    with pytest.raises(InvalidArgument, match="requires Dynamo frontend preprocessing"):
        apply_thinking_budget(
            {"thinking_token_budget": 32, "require_reasoning": True},
            {},
            _server_args(),
        )


def test_apply_thinking_budget_uses_runtime_parser_for_auto_config():
    tokenizer_manager = SimpleNamespace(
        config_value=lambda name: "qwen3" if name == "reasoning_parser" else None
    )
    engine = SimpleNamespace(tokenizer_manager=tokenizer_manager)

    actual = apply_thinking_budget(
        {
            "stop_conditions": {"max_thinking_tokens": 32},
            "require_reasoning": True,
        },
        {},
        _server_args(reasoning_parser="auto"),
        engine=engine,
    )

    assert actual == {"custom_params": {"thinking_budget": 32}}


def test_apply_thinking_budget_rejects_custom_logit_processor():
    request = {
        "stop_conditions": {"max_thinking_tokens": 32},
        "custom_logit_processor": "serialized-processor",
        "require_reasoning": True,
    }

    with pytest.raises(InvalidArgument, match="custom_logit_processor"):
        apply_thinking_budget(request, {}, _server_args())


def test_apply_thinking_budget_rejects_sampling_param_custom_logit_processor():
    request = {
        "stop_conditions": {"max_thinking_tokens": 32},
        "require_reasoning": True,
    }

    with pytest.raises(InvalidArgument, match="custom_logit_processor"):
        apply_thinking_budget(
            request,
            {"custom_logit_processor": "serialized-processor"},
            _server_args(),
        )


def test_apply_thinking_budget_rejects_non_object_custom_params():
    with pytest.raises(InvalidArgument, match="custom_params"):
        apply_thinking_budget(
            {
                "stop_conditions": {"max_thinking_tokens": 32},
                "require_reasoning": True,
            },
            {"custom_params": ["not", "an", "object"]},
            _server_args(),
        )


def test_apply_thinking_budget_rejects_parser_without_active_token_filter(
    monkeypatch,
):
    monkeypatch.delenv("SGLANG_MAX_THINK_TOKENS", raising=False)

    with pytest.raises(InvalidArgument, match="cannot enforce per-request"):
        apply_thinking_budget(
            {
                "stop_conditions": {"max_thinking_tokens": 32},
                "require_reasoning": True,
            },
            {},
            _server_args(reasoning_parser="deepseek-r1"),
        )
