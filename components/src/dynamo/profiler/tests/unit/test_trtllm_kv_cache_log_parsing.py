# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRT-LLM KV-cache token parsing from runtime logs."""

import pytest

from dynamo.profiler.utils.config_modifiers.trtllm import TrtllmConfigModifier

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def test_get_kv_cache_size_uses_last_matching_log_entry(tmp_path) -> None:
    """Parser returns the last paged-KV token value when multiple entries exist."""
    log_path = tmp_path / "dynamo.log"
    log_path.write_text(
        "\n".join(
            [
                "random startup line",
                "[TensorRT-LLM][INFO] [MemUsageChange] Allocated 12.50 GiB for max tokens in paged KV cache (65536).",
                "intermediate line",
                "[TensorRT-LLM][INFO] [MemUsageChange] Allocated 43.87 GiB for max tokens in paged KV cache (229984).",
            ]
        )
    )

    parsed = TrtllmConfigModifier.get_kv_cache_size_from_dynamo_log(str(log_path))

    assert parsed == 229984


def test_get_kv_cache_size_falls_back_when_missing(tmp_path) -> None:
    """Parser returns default fallback when no paged-KV line is present."""
    log_path = tmp_path / "dynamo.log"
    log_path.write_text("startup\nhealthcheck ok\n")

    parsed = TrtllmConfigModifier.get_kv_cache_size_from_dynamo_log(str(log_path))

    assert parsed == 100000
