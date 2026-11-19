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


@pytest.mark.parametrize("budget", [0, 16])
@pytest.mark.parametrize(
    "reasoning_fields", [{}, {"require_reasoning": False}, {"require_reasoning": True}]
)
@pytest.mark.parametrize("parser", ["gpt-oss", "auto"])
def test_gpt_oss_structured_output_rejects_budget(
    monkeypatch, budget, reasoning_fields, parser
):
    monkeypatch.setenv("SGLANG_MAX_THINK_TOKENS", "128")
    engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(config_value=lambda key: "gpt-oss")
    )
    with pytest.raises(InvalidArgument, match="GPT-OSS.*json_schema"):
        apply_thinking_budget(
            {"stop_conditions": {"max_thinking_tokens": budget}, **reasoning_fields},
            {"json_schema": '{"type":"object"}'},
            _server_args(reasoning_parser=parser),
            engine=engine,
        )


def test_gpt_oss_structured_output_without_budget_preserves_sampling():
    sampling = {"json_schema": '{"type":"object"}'}
    assert (
        apply_thinking_budget({}, sampling, _server_args(reasoning_parser="gpt-oss"))
        == sampling
    )


def test_gpt_oss_plain_output_accepts_budget(monkeypatch):
    monkeypatch.setenv("SGLANG_MAX_THINK_TOKENS", "128")
    assert apply_thinking_budget(
        {"stop_conditions": {"max_thinking_tokens": 16}, "require_reasoning": True},
        {"json_schema": None},
        _server_args(reasoning_parser="gpt-oss"),
    ) == {"json_schema": None, "custom_params": {"thinking_budget": 16}}


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


@pytest.mark.parametrize(
    "reasoning_fields", [{"require_reasoning": False}, {}], ids=["false", "omitted"]
)
def test_apply_thinking_budget_ignores_budget_when_request_reasoning_disabled(
    reasoning_fields,
):
    sampling = {"custom_params": {"thinking_budget": 8, "other": True}}
    original = deepcopy(sampling)
    actual = apply_thinking_budget(
        {
            "stop_conditions": {"max_thinking_tokens": 32},
            **reasoning_fields,
        },
        sampling,
        _server_args(),
    )
    assert actual == {"custom_params": {"other": True}}
    assert sampling == original


@pytest.mark.parametrize("require_reasoning", [None, 0, 1, "false", [], {}])
def test_apply_thinking_budget_rejects_malformed_request_reasoning(require_reasoning):
    with pytest.raises(InvalidArgument, match="requires reasoning to be enabled"):
        apply_thinking_budget(
            {
                "stop_conditions": {"max_thinking_tokens": 32},
                "require_reasoning": require_reasoning,
            },
            {},
            _server_args(),
        )


@pytest.mark.parametrize(
    "reasoning_fields", [{"require_reasoning": False}, {}], ids=["false", "omitted"]
)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_strict_thinking": False}, "--enable-strict-thinking"),
        ({"reasoning_parser": None}, "--reasoning-parser"),
        ({"skip_tokenizer_init": True}, "--skip-tokenizer-init"),
    ],
)
def test_disabled_reasoning_does_not_bypass_unsupported_server_validation(
    reasoning_fields, overrides, message
):
    with pytest.raises(InvalidArgument, match=message):
        apply_thinking_budget(
            {
                "stop_conditions": {"max_thinking_tokens": 32},
                **reasoning_fields,
            },
            {},
            _server_args(**overrides),
        )


@pytest.mark.parametrize(
    "reasoning_fields", [{"require_reasoning": False}, {}], ids=["false", "omitted"]
)
@pytest.mark.parametrize("value", [True, -1, 2**32, 1.5, "32"])
def test_disabled_reasoning_does_not_bypass_budget_validation(reasoning_fields, value):
    with pytest.raises(InvalidArgument, match="must be an integer"):
        apply_thinking_budget(
            {
                "stop_conditions": {"max_thinking_tokens": value},
                **reasoning_fields,
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


def test_apply_thinking_budget_ignores_unforwarded_custom_logit_processor():
    request = {
        "stop_conditions": {"max_thinking_tokens": 32},
        "custom_logit_processor": "serialized-processor",
        "require_reasoning": True,
    }

    assert apply_thinking_budget(request, {}, _server_args()) == {
        "custom_params": {"thinking_budget": 32}
    }


def test_apply_thinking_budget_accepts_null_forwarded_custom_params():
    request = {
        "stop_conditions": {"max_thinking_tokens": 32},
        "require_reasoning": True,
    }
    assert apply_thinking_budget(request, {"custom_params": None}, _server_args()) == {
        "custom_params": {"thinking_budget": 32}
    }


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


@pytest.mark.parametrize("global_budget", [0, 32, -1])
def test_apply_thinking_budget_matches_global_token_filter_activation(
    monkeypatch, global_budget
):
    monkeypatch.setenv("SGLANG_MAX_THINK_TOKENS", str(global_budget))

    def apply():
        return apply_thinking_budget(
            {
                "stop_conditions": {"max_thinking_tokens": 16},
                "require_reasoning": True,
            },
            {},
            _server_args(reasoning_parser="deepseek-r1"),
        )

    if global_budget >= 0:
        assert apply() == {"custom_params": {"thinking_budget": 16}}
    else:
        with pytest.raises(InvalidArgument, match="cannot enforce per-request"):
            apply()


def test_global_token_filter_activation_is_not_cached(monkeypatch):
    request = {
        "stop_conditions": {"max_thinking_tokens": 16},
        "require_reasoning": True,
    }
    args = _server_args(reasoning_parser="deepseek-r1")
    monkeypatch.setenv("SGLANG_MAX_THINK_TOKENS", "-1")
    with pytest.raises(InvalidArgument, match="cannot enforce per-request"):
        apply_thinking_budget(request, {}, args)
    monkeypatch.setenv("SGLANG_MAX_THINK_TOKENS", "32")
    assert apply_thinking_budget(request, {}, args) == {
        "custom_params": {"thinking_budget": 16}
    }
