# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA/GPU not available, but the tensorrt_llm import behind the handler "
        "module requires GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.request_handlers.handler_base import _prompt_tokens_details

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
]


def _res(cached_tokens, reused=None, missed=None, with_perf=True):
    km = (
        SimpleNamespace(num_reused_blocks=reused, num_missed_blocks=missed)
        if reused is not None or missed is not None
        else None
    )
    pm = SimpleNamespace(kv_cache_metrics=km) if with_perf else None
    return SimpleNamespace(
        cached_tokens=cached_tokens, outputs=[SimpleNamespace(request_perf_metrics=pm)]
    )


def test_context_path_keeps_clamped_engine_value():
    assert _prompt_tokens_details(_res(5000, with_perf=False), 4000, False) == {
        "cached_tokens": 4000
    }
    assert _prompt_tokens_details(_res(None, with_perf=False), 4000, False) == {
        "cached_tokens": 0
    }
    assert _prompt_tokens_details(_res(320, reused=250, missed=0), 4000, False) == {
        "cached_tokens": 320
    }


def test_generation_only_without_perf_metrics_is_unchanged():
    assert _prompt_tokens_details(_res(4000, with_perf=False), 4000, True) == {
        "cached_tokens": 4000
    }
    assert _prompt_tokens_details(_res(4000), 4000, True) == {"cached_tokens": 4000}


def test_generation_only_uses_hit_ratio():
    d = _prompt_tokens_details(_res(4000, reused=10, missed=115), 4000, True)
    assert d == {"cached_tokens": 320, "_engine_reported": 4000}


def test_generation_only_ratio_is_window_count_independent():
    one_window = _prompt_tokens_details(_res(4000, reused=10, missed=115), 4000, True)
    two_windows = _prompt_tokens_details(_res(4000, reused=20, missed=230), 4000, True)
    assert one_window["cached_tokens"] == two_windows["cached_tokens"] == 320


def test_generation_only_never_reports_full_prompt():
    d = _prompt_tokens_details(_res(4000, reused=125, missed=0), 4000, True)
    assert d["cached_tokens"] == 3999


def test_generation_only_zero_reuse():
    assert (
        _prompt_tokens_details(_res(4000, reused=0, missed=125), 4000, True)[
            "cached_tokens"
        ]
        == 0
    )
    assert (
        _prompt_tokens_details(_res(4000, reused=0, missed=0), 4000, True)[
            "cached_tokens"
        ]
        == 0
    )
