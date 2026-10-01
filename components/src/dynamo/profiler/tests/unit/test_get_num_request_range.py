# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for get_num_request_range() — the decode num_request sweep.

Regression guard for the attn_dp_size double-multiplication that pushed the
sweep's upper bound to max_concurrency * attn_dp_size instead of max_concurrency,
saturating the temporary decode worker's KV cache for any attn_dp_size > 1."""

import pytest

from dynamo.profiler.utils.defaults import DECODE_MAX_CONCURRENCY
from dynamo.profiler.utils.profile_decode import get_num_request_range

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.parametrize(
    "attn_dp_size, engine_max_concurrency, granularity",
    [
        (1, 1200, 6),
        (2, 1200, 6),
        (4, 1209, 6),
        (8, 4096, 6),  # engine_max above DECODE_MAX_CONCURRENCY -> clamped
        (4, 2000, 8),
    ],
)
def test_sweep_stays_within_kv_capacity(
    attn_dp_size, engine_max_concurrency, granularity
):
    """The sweep upper bound must not exceed the KV-cache concurrency limit.

    Before the fix the upper bound was ~max_concurrency * attn_dp_size, which
    saturated the temporary decode worker's KV cache for any attn_dp_size > 1
    and drove its liveness probe into a restart loop.
    """
    result = get_num_request_range(attn_dp_size, engine_max_concurrency, granularity)
    max_concurrency = min(engine_max_concurrency, DECODE_MAX_CONCURRENCY)

    assert result, "sweep must not be empty"
    # upper bound lands within the KV-cache capacity (the regression)
    assert result[-1] <= max_concurrency
    # every probe is a whole multiple of attn_dp_size (round-robin dp invariant)
    assert all(n % attn_dp_size == 0 for n in result)
    # monotonically non-decreasing
    assert result == sorted(result)


def test_dense_path_unchanged():
    """attn_dp_size == 1 is a no-op for the fix (step is identical either way)."""
    assert get_num_request_range(1, 1200, 6) == [1, 240, 480, 720, 960, 1200]


def test_moe_attn_dp_upper_bound_not_inflated():
    """Explicit regression for the attn_dp_size=4 overshoot.

    Pre-fix produced [4, 964, 1928, 2892, 3856, 4820] (last = 4820, ~4x the
    1209 limit). Post-fix the last probe is less than 1209.
    """
    result = get_num_request_range(4, 1209, 6)
    assert result[-1] <= 1209
    assert result == [4, 244, 484, 724, 964, 1208]


def test_small_capacity_branch_unaffected():
    """conc_per_dp < granularity uses the range() branch, untouched by the fix."""
    # max_concurrency=12, attn_dp_size=4 -> conc_per_dp=3 < granularity=6
    assert get_num_request_range(4, 12, 6) == [4, 8, 12]
