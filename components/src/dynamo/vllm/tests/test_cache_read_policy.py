# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache-read declarations must not label different worker execution paths."""

import pytest

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, WorkerType
from dynamo.vllm.cache_read_policy import publish_prefix_cache_read_policy

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
        (ModelInput.Tokens, ModelType.Embedding, WorkerType.Aggregated, False),
    ],
)
def test_cache_read_policy_only_labels_the_known_adapter(
    model_input, model_type, worker_type, expected
):
    config = ModelRuntimeConfig()
    assert (
        publish_prefix_cache_read_policy(config, model_input, model_type, worker_type)
        is expected
    )
    value = config.get_engine_specific("prefix_cache_read_policy")
    assert value == ("vllm_python_v1" if expected else None)
