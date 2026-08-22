# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the prefix-cache discount in aic_dataframe.

``build_prefill_row`` previously hardcoded ``prefix = 0`` (cold cache). It now
accepts a ``kv_hit_rate`` (0-1) so disagg planning can model a warm prefix
cache: ``prefix = round(kv_hit_rate * isl)`` and the prefill is charged only
for the uncached ``effective_isl = isl - prefix`` tokens.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.profiler.utils.aic_dataframe import (
        build_decode_row,
        build_disagg_df_from_static,
        build_prefill_row,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


def _prefill(**overrides) -> dict:
    base = dict(
        model="test-model",
        isl=4000,
        osl=1000,
        ttft=50.0,
        tp=1,
        pp=1,
        dp=1,
        moe_tp=1,
        moe_ep=1,
        backend="trtllm",
        system="h200_sxm",
    )
    base.update(overrides)
    return build_prefill_row(**base)


def test_default_is_cold_cache() -> None:
    """Default kv_hit_rate keeps the previous prefix=0 behaviour."""
    assert _prefill()["prefix"] == 0


@pytest.mark.parametrize(
    "isl,rate,expected",
    [
        (4000, 0.0, 0),
        (4000, 0.5, 2000),
        (4000, 1.0, 4000),
        (5424, 0.51, round(0.51 * 5424)),  # silicon agg anchor
        (1001, 0.5, round(0.5 * 1001)),  # rounding (500 or 501)
    ],
)
def test_prefix_from_hit_rate(isl: int, rate: float, expected: int) -> None:
    assert _prefill(isl=isl, kv_hit_rate=rate)["prefix"] == expected


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
def test_hit_rate_out_of_range_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        _prefill(kv_hit_rate=bad)


def test_build_disagg_df_threads_hit_rate() -> None:
    """kv_hit_rate on build_disagg_df_from_static sets prefix on the disagg row."""
    prefill_df = pd.DataFrame([_prefill(isl=4000)])  # prefix defaults to 0
    decode_df = pd.DataFrame(
        [
            build_decode_row(
                tpot=10.0,
                thpt_per_gpu=100.0,
                num_request=8,
                num_gpus=1,
                osl=1000,
                tp=1,
                pp=1,
                dp=1,
                moe_tp=1,
                moe_ep=1,
            )
        ]
    )

    cold = build_disagg_df_from_static(prefill_df, decode_df)
    assert int(cold.iloc[0]["prefix"]) == 0

    warm = build_disagg_df_from_static(prefill_df, decode_df, kv_hit_rate=0.51)
    assert int(warm.iloc[0]["prefix"]) == round(0.51 * 4000)


def test_build_disagg_df_none_preserves_row_prefix() -> None:
    """kv_hit_rate=None leaves a prefix already baked into the prefill row."""
    prefill_df = pd.DataFrame([_prefill(isl=4000, kv_hit_rate=0.25)])
    decode_df = pd.DataFrame(
        [
            build_decode_row(
                tpot=10.0,
                thpt_per_gpu=100.0,
                num_request=8,
                num_gpus=1,
                osl=1000,
                tp=1,
                pp=1,
                dp=1,
                moe_tp=1,
                moe_ep=1,
            )
        ]
    )
    out = build_disagg_df_from_static(prefill_df, decode_df)  # kv_hit_rate=None
    assert int(out.iloc[0]["prefix"]) == round(0.25 * 4000)
