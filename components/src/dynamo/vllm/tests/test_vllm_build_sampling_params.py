# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.vllm.handlers import build_sampling_params

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_max_thinking_tokens_maps_to_thinking_token_budget():
    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_thinking_tokens": 1024},
        "output_options": {},
    }
    sp = build_sampling_params(request, default_sampling_params={})
    assert sp.thinking_token_budget == 1024
