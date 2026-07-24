# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.llm import ModelType
from dynamo.vllm.capacity import (
    STRICT_REQUEST_TOKEN_LIMIT_RUNTIME_KEY,
    get_metrics_model_name,
    get_spec_decode_runtime_data,
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
    ("model_type", "expected_limit"),
    [(ModelType.Chat, "4096"), (ModelType.Embedding, None)],
)
def test_vllm_request_token_limit_is_generation_only(model_type, expected_limit):
    from dynamo.vllm.main import _set_vllm_request_token_limit

    runtime_config = SimpleNamespace(set_engine_specific=Mock())

    _set_vllm_request_token_limit(runtime_config, model_type, 4096)

    if expected_limit is None:
        runtime_config.set_engine_specific.assert_not_called()
    else:
        runtime_config.set_engine_specific.assert_called_once_with(
            STRICT_REQUEST_TOKEN_LIMIT_RUNTIME_KEY, expected_limit
        )


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
