# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.token_budget import TOKEN_BUDGET_RUNTIME_KEY
from dynamo.llm import ModelType
from dynamo.vllm.capacity import get_metrics_model_name, get_spec_decode_runtime_data

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
    ("model_type", "should_publish"),
    [(ModelType.Chat, True), (ModelType.Embedding, False)],
)
def test_vllm_token_budget_is_generation_only(model_type, should_publish):
    from dynamo.vllm.main import _set_vllm_token_budget

    runtime_config = SimpleNamespace(set_engine_specific=Mock())

    _set_vllm_token_budget(runtime_config, model_type, 4096)

    if not should_publish:
        runtime_config.set_engine_specific.assert_not_called()
    else:
        runtime_config.set_engine_specific.assert_called_once()
        key, value = runtime_config.set_engine_specific.call_args.args
        assert key == TOKEN_BUDGET_RUNTIME_KEY
        assert json.loads(value) == {
            "combined_limit": 4096,
            "output_overflow": "reject",
            "prompt_overflow": "reject",
        }


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
