# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from tests.utils.payloads import CachedTokensChatPayload, RouterKvHitRateBelowThreshold

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _payload_with_hit_rate(
    avg: float, *, threshold: float = 0.5
) -> CachedTokensChatPayload:
    payload = CachedTokensChatPayload(
        body={},
        min_cached_tokens=0,
        min_avg_kv_hit_rate=threshold,
        require_positive_avg_kv_hit_rate=True,
    )
    payload._metrics_baseline = (1.0, 1.0)
    payload._scrape_router_kv_hit_rate = lambda: (1.0 + avg, 2.0)
    return payload


def test_router_hit_rate_zero_fails_as_broken_control_prefix():
    with pytest.raises(AssertionError, match="control prefix") as exc_info:
        _payload_with_hit_rate(0.0).final_validation()

    assert type(exc_info.value) is AssertionError


def test_router_hit_rate_partial_raises_expected_threshold_failure():
    with pytest.raises(RouterKvHitRateBelowThreshold, match=r"\(0\.250\)"):
        _payload_with_hit_rate(0.25).final_validation()


def test_router_hit_rate_exactly_half_fails_strictly_above_half_gate():
    with pytest.raises(RouterKvHitRateBelowThreshold, match=r"\(0\.500\)"):
        _payload_with_hit_rate(
            0.5, threshold=math.nextafter(0.5, 1.0)
        ).final_validation()


def test_router_hit_rate_above_threshold_passes():
    _payload_with_hit_rate(0.75, threshold=math.nextafter(0.5, 1.0)).final_validation()
