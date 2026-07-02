# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.vllm.capacity import (
    REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY,
    get_metrics_model_name,
    get_spec_decode_runtime_data,
    publish_reasoning_aware_guided_decoding,
    reasoning_aware_guided_decoding_runtime_data,
    supports_reasoning_aware_guided_decoding,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_spec_decode_runtime_data_uses_vllm_speculative_config():
    config = SimpleNamespace(
        engine_args=SimpleNamespace(
            speculative_config={"num_speculative_tokens": 99, "method": "ignored"}
        )
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3, method="eagle")
    )

    assert get_spec_decode_runtime_data(config, vllm_config) == {
        "nextn": 3,
        "method": "eagle",
        "source": "backend_config",
    }


def test_metrics_model_name_prefers_served_model_name():
    config = SimpleNamespace(model="meta-llama/Llama-3.1-8B", served_model_name="llama")

    assert get_metrics_model_name(config) == "llama"


def test_metrics_model_name_falls_back_to_model():
    config = SimpleNamespace(model="meta-llama/Llama-3.1-8B", served_model_name=None)

    assert get_metrics_model_name(config) == "meta-llama/Llama-3.1-8B"


@pytest.mark.parametrize(
    ("structured_outputs_config", "expected"),
    [
        (
            SimpleNamespace(reasoning_parser="nemotron_v3", enable_in_reasoning=False),
            True,
        ),
        (
            SimpleNamespace(reasoning_parser="nemotron_v3", enable_in_reasoning=True),
            False,
        ),
        (SimpleNamespace(reasoning_parser="", enable_in_reasoning=False), False),
        (SimpleNamespace(reasoning_parser="nemotron_v3"), False),
        (None, False),
    ],
)
def test_supports_reasoning_aware_guided_decoding(structured_outputs_config, expected):
    vllm_config = SimpleNamespace(structured_outputs_config=structured_outputs_config)

    assert (
        supports_reasoning_aware_guided_decoding(vllm_config, "nemotron_v3") is expected
    )


def test_granite_without_native_reasoning_boundary_fails_capability_closed():
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser="granite", enable_in_reasoning=False
        )
    )

    assert not supports_reasoning_aware_guided_decoding(vllm_config, "granite")


def test_reasoning_aware_guided_decoding_missing_config_fails_closed():
    assert (
        supports_reasoning_aware_guided_decoding(SimpleNamespace(), "nemotron_v3")
        is False
    )


def test_reasoning_aware_guided_decoding_requires_dynamo_parser():
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser="nemotron_v3", enable_in_reasoning=False
        )
    )

    assert not supports_reasoning_aware_guided_decoding(vllm_config)


def test_reasoning_aware_guided_decoding_rejects_parser_mismatch():
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser="qwen3", enable_in_reasoning=False
        )
    )

    assert not supports_reasoning_aware_guided_decoding(vllm_config, "gpt_oss")


@pytest.mark.parametrize(
    ("dynamo_name", "vllm_name"),
    [
        ("gpt_oss", "openai_gptoss"),
        ("kimi_k25", "kimi_k2"),
        ("minimax_append_think", "minimax_m2_append_think"),
        ("deepseek_v3_1", "deepseek_v3"),
        ("deepseek_v3_2", "deepseek_v3"),
        ("gemma-4", "gemma4"),
        ("nemotron_nano", "nemotron_v3"),
        ("nemotron3", "nemotron_v3"),
    ],
)
def test_normalizes_dynamo_reasoning_parser_aliases(dynamo_name, vllm_name):
    from dynamo.vllm.capacity import normalize_vllm_reasoning_parser

    assert normalize_vllm_reasoning_parser(dynamo_name) == vllm_name


@pytest.mark.parametrize(
    ("dynamo_name", "vllm_name"),
    [
        ("qwen25", "hermes"),
        ("harmony", "openai"),
        ("deepseek_v3_1", "deepseek_v31"),
        ("deepseek_v3_2", "deepseek_v32"),
        ("deepseek-v4", "deepseek_v4"),
        ("deepseekv4", "deepseek_v4"),
        ("gemma-4", "gemma4"),
        ("minimax-m3-nom", "minimax_m3"),
        ("nemotron_nano", "qwen3_coder"),
    ],
)
def test_normalizes_dynamo_tool_parser_aliases(dynamo_name, vllm_name):
    from dynamo.vllm.capacity import normalize_vllm_tool_parser

    assert normalize_vllm_tool_parser(dynamo_name) == vllm_name


def test_reasoning_aware_guided_decoding_runtime_metadata_publication():
    calls = []
    runtime_config = SimpleNamespace(
        set_engine_specific=lambda key, value: calls.append((key, value))
    )
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser="nemotron_v3", enable_in_reasoning=False
        )
    )

    assert reasoning_aware_guided_decoding_runtime_data(vllm_config, "nemotron_v3") == {
        REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY: True
    }
    assert (
        publish_reasoning_aware_guided_decoding(
            runtime_config, vllm_config, "nemotron_v3"
        )
        is True
    )
    assert calls == [(REASONING_AWARE_GUIDED_DECODING_RUNTIME_KEY, "true")]


def test_reasoning_aware_guided_decoding_runtime_metadata_not_published_when_unsupported():
    calls = []
    runtime_config = SimpleNamespace(
        set_engine_specific=lambda key, value: calls.append((key, value))
    )
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            reasoning_parser="nemotron_v3", enable_in_reasoning=True
        )
    )

    assert reasoning_aware_guided_decoding_runtime_data(vllm_config) is None
    assert publish_reasoning_aware_guided_decoding(runtime_config, vllm_config) is False
    assert calls == []


def test_spec_decode_runtime_data_falls_back_to_engine_args_json():
    config = SimpleNamespace(
        engine_args=SimpleNamespace(
            speculative_config='{"num_speculative_tokens": "4", "method": "ngram"}'
        )
    )
    vllm_config = SimpleNamespace(speculative_config=None)

    assert get_spec_decode_runtime_data(config, vllm_config) == {
        "nextn": 4,
        "method": "ngram",
        "source": "backend_config",
    }


@pytest.mark.parametrize(
    "speculative_config",
    [None, {}, {"num_speculative_tokens": 0}, {"num_speculative_tokens": "bad"}],
)
def test_spec_decode_runtime_data_ignores_invalid_nextn(speculative_config):
    config = SimpleNamespace(
        engine_args=SimpleNamespace(speculative_config=speculative_config)
    )
    vllm_config = SimpleNamespace(speculative_config=None)

    assert get_spec_decode_runtime_data(config, vllm_config) is None
