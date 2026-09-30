# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.request_handlers.handler_base import _prompt_tokens_details

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
]


def _res(cached_tokens, reused_blocks=None, with_perf=True):
    km = (
        SimpleNamespace(num_reused_blocks=reused_blocks)
        if reused_blocks is not None
        else None
    )
    pm = SimpleNamespace(kv_cache_metrics=km) if with_perf else None
    return SimpleNamespace(
        cached_tokens=cached_tokens, outputs=[SimpleNamespace(request_perf_metrics=pm)]
    )


def test_without_perf_metrics_keeps_clamped_engine_value():
    assert _prompt_tokens_details(_res(5000, with_perf=False), 4000, 32) == {
        "cached_tokens": 4000
    }
    assert _prompt_tokens_details(_res(None, with_perf=False), 4000, 32) == {
        "cached_tokens": 0
    }


def test_kv_metrics_override_engine_value():
    d = _prompt_tokens_details(_res(4000, reused_blocks=10), 4000, 32)
    assert d == {"cached_tokens": 320, "_engine_reported": 4000}


def test_kv_metrics_never_report_full_prompt():
    d = _prompt_tokens_details(_res(4000, reused_blocks=125), 4000, 32)
    assert d["cached_tokens"] == 3999


def test_kv_metrics_zero_reuse():
    assert (
        _prompt_tokens_details(_res(4000, reused_blocks=0), 4000, 32)["cached_tokens"]
        == 0
    )
