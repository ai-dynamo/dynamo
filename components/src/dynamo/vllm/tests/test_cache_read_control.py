# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only adapters honoring the typed decision advertise cache-read control."""

import json

import pytest

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, WorkerType
from dynamo.vllm.cache_read_control import (
    SUPPORTS_PREFIX_CACHE_READ_CONTROL_KEY,
    publish_prefix_cache_read_control_capability,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
]


@pytest.mark.parametrize(
    "model_input,model_type,worker_type,expected",
    [
        (ModelInput.Tokens, ModelType.Chat, WorkerType.Aggregated, True),
        (ModelInput.Tokens, ModelType.Completions, WorkerType.Aggregated, True),
        (ModelInput.Text, ModelType.Chat, WorkerType.Aggregated, False),
        (ModelInput.Tokens, ModelType.Chat, WorkerType.Prefill, False),
        (ModelInput.Tokens, ModelType.Chat, WorkerType.Decode, False),
        (ModelInput.Tokens, ModelType.Chat, WorkerType.Encode, False),
        (ModelInput.Tokens, ModelType.Embedding, WorkerType.Aggregated, False),
    ],
)
def test_cache_read_control_only_advertised_on_supported_adapters(
    model_input, model_type, worker_type, expected
):
    config = ModelRuntimeConfig()
    assert (
        publish_prefix_cache_read_control_capability(
            config, model_input, model_type, worker_type
        )
        is expected
    )
    runtime_data = config.runtime_data
    if expected:
        assert json.loads(runtime_data[SUPPORTS_PREFIX_CACHE_READ_CONTROL_KEY]) is True
    else:
        assert SUPPORTS_PREFIX_CACHE_READ_CONTROL_KEY not in runtime_data
    assert "prefix_cache_read_policy" not in runtime_data
