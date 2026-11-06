# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from dynamo.llm import ModelRuntimeConfig
from dynamo.vllm.capacity import publish_vllm_scheduling_budget

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize("scheduled", [None, 64, 256, 512])
def test_publish_scheduling_budget_preserves_model_capacity(scheduled):
    runtime = ModelRuntimeConfig()
    runtime.max_num_batched_tokens = 256

    publish_vllm_scheduling_budget(
        runtime,
        SimpleNamespace(
            max_num_batched_tokens=256,
            max_num_scheduled_tokens=scheduled,
        ),
    )

    assert runtime.max_num_batched_tokens == 256
    payload = {key: json.loads(value) for key, value in runtime.runtime_data.items()}
    assert payload == ({"max_num_scheduled_tokens": 64} if scheduled == 64 else {})


def test_legacy_vllm_scheduler_has_no_extra_budget_metadata():
    runtime = ModelRuntimeConfig()
    publish_vllm_scheduling_budget(runtime, SimpleNamespace(max_num_batched_tokens=256))
    assert runtime.runtime_data == {}


@pytest.mark.parametrize("scheduled", [0, -1, "64", True])
def test_invalid_scheduling_budget_is_not_published(scheduled):
    runtime = ModelRuntimeConfig()
    with pytest.raises(ValueError, match="max_num_scheduled_tokens"):
        publish_vllm_scheduling_budget(
            runtime,
            SimpleNamespace(
                max_num_batched_tokens=256,
                max_num_scheduled_tokens=scheduled,
            ),
        )
    assert runtime.runtime_data == {}
