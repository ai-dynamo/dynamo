# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared GPU budget primitives in
``dynamo.planner.core.budget``."""

import pytest

from dynamo.planner.core.budget import (
    bounds_for_total,
    compute_tolerance,
    proportional_clamp_pair,
    proportional_clamp_single,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------- #
# compute_tolerance                                                            #
# ---------------------------------------------------------------------------- #


def test_compute_tolerance_takes_max():
    assert compute_tolerance([1, 2]) == 2
    assert compute_tolerance([2, 1]) == 2
    assert compute_tolerance([4]) == 4


def test_compute_tolerance_ignores_zero_and_negative():
    assert compute_tolerance([0, 0]) == 0
    assert compute_tolerance([1, 0, 2, -1]) == 2


def test_compute_tolerance_empty_returns_zero():
    assert compute_tolerance([]) == 0


# ---------------------------------------------------------------------------- #
# bounds_for_total                                                             #
# ---------------------------------------------------------------------------- #


def test_bounds_in_band():
    assert bounds_for_total(8, 4, 8, 0) == (True, "")
    assert bounds_for_total(4, 4, 8, 0) == (True, "")


def test_bounds_above_ceiling_strict():
    in_band, reason = bounds_for_total(9, 4, 8, 0)
    assert in_band is False
    assert "exceeds ceiling" in reason


def test_bounds_below_floor_strict():
    in_band, reason = bounds_for_total(3, 4, 8, 0)
    assert in_band is False
    assert "below floor" in reason


def test_bounds_with_tolerance_accepts_edges():
    # tol=2 widens the band to [2, 10].
    assert bounds_for_total(2, 4, 8, 2) == (True, "")
    assert bounds_for_total(10, 4, 8, 2) == (True, "")
    in_band, reason = bounds_for_total(11, 4, 8, 2)
    assert in_band is False
    assert "+ tol 2" in reason


def test_bounds_disabled_floor():
    # min=-1 disables the floor entirely.
    assert bounds_for_total(0, -1, 8, 0) == (True, "")
    assert bounds_for_total(9, -1, 8, 0)[0] is False  # ceiling still active


def test_bounds_disabled_ceiling():
    assert bounds_for_total(100, 4, -1, 0) == (True, "")
    assert bounds_for_total(3, 4, -1, 0)[0] is False  # floor still active


def test_bounds_both_disabled():
    assert bounds_for_total(999, -1, -1, 0) == (True, "")


# ---------------------------------------------------------------------------- #
# proportional_clamp_pair — happy paths                                        #
# ---------------------------------------------------------------------------- #


def test_clamp_pair_no_op_when_both_disabled():
    assert proportional_clamp_pair(3, 5, 1, 1, -1, -1, 1) == (3, 5)


def test_clamp_pair_in_band_returns_inputs():
    # min=max=4, p_gpu=d_gpu=1, desired (3,1) totals 4 — in band.
    assert proportional_clamp_pair(3, 1, 1, 1, 4, 4, 1) == (3, 1)


def test_clamp_pair_pushes_up_to_floor():
    # desired (1,1)=2 < min=4. Symmetric pools, tol=1, accepts [3, 5].
    new_p, new_d = proportional_clamp_pair(1, 1, 1, 1, 4, 4, 1)
    assert 3 <= new_p + new_d <= 5


def test_clamp_pair_shrinks_to_ceiling_when_only_max_set():
    # No floor → strict ceiling, no tolerance. Mirrors historical behavior.
    new_p, new_d = proportional_clamp_pair(5, 5, 1, 1, -1, 4, 1)
    assert new_p * 1 + new_d * 1 == 4


def test_clamp_pair_disabled_caps_returns_inputs():
    assert proportional_clamp_pair(3, 5, 1, 1, -1, -1, 1) == (3, 5)


# ---------------------------------------------------------------------------- #
# proportional_clamp_pair — asymmetric pool sizes                              #
# ---------------------------------------------------------------------------- #


def test_clamp_pair_asymmetric_within_tolerance_no_op():
    # prefill 1 GPU/worker, decode 2 GPU/worker. min=max=4. tol=2 → [2, 6].
    # desired (1, 2) = 1 + 4 = 5 → in band, returned unchanged.
    assert proportional_clamp_pair(1, 2, 1, 2, 4, 4, 1) == (1, 2)


def test_clamp_pair_asymmetric_floor_grows():
    # prefill=1, decode=2. min=max=5. desired (1,1)=3 < min=5. tol=2 → [3, 7].
    # 3 is the lower edge — floor logic still pushes up to land in band.
    new_p, new_d = proportional_clamp_pair(1, 1, 1, 2, 5, 5, 1)
    total = new_p * 1 + new_d * 2
    assert 3 <= total <= 7


def test_clamp_pair_asymmetric_unreachable_target_converges():
    # Both pools = 2 GPU/worker. min=max=5 unreachable (totals are even).
    # tol=2 → band [3, 7]. Should land at 4 or 6 in one pass.
    new_p, new_d = proportional_clamp_pair(1, 1, 2, 2, 5, 5, 1)
    total = new_p * 2 + new_d * 2
    assert total in (4, 6)


def test_clamp_pair_no_oscillation_after_one_pass():
    # Calling clamp again on its own output must yield the same result.
    p1, d1 = proportional_clamp_pair(1, 1, 2, 2, 5, 5, 1)
    p2, d2 = proportional_clamp_pair(p1, d1, 2, 2, 5, 5, 1)
    assert (p1, d1) == (p2, d2)


def test_clamp_pair_asymmetric_min_max_open_band():
    # min=4, max=10, prefill=1, decode=2. desired (1, 1) = 3 < min.
    # Tolerance applies (both bounds set), tol=2, band [2, 12].
    # 3 is in band → no-op.
    assert proportional_clamp_pair(1, 1, 1, 2, 4, 10, 1) == (1, 1)


# ---------------------------------------------------------------------------- #
# proportional_clamp_pair — edge cases                                         #
# ---------------------------------------------------------------------------- #


def test_clamp_pair_invalid_gpu_returns_inputs():
    # If capabilities aren't initialized (gpu count <= 0), no-op.
    assert proportional_clamp_pair(3, 5, 0, 1, 4, 8, 1) == (3, 5)
    assert proportional_clamp_pair(3, 5, 1, 0, 4, 8, 1) == (3, 5)


def test_clamp_pair_ceiling_below_min_endpoint_returns_zero():
    # Ceiling can't fit even min_endpoint per pool -> (0, 0).
    assert proportional_clamp_pair(5, 5, 1, 1, -1, 1, 1) == (0, 0)


def test_clamp_pair_ceiling_below_min_endpoint_within_tolerance_keeps_alive():
    # Reviewer's repro: p_gpu=2, d_gpu=2, min=max=3, min_endpoint=1.
    # Tolerance=2 so band is [1, 5]. Strict ceiling (3) is below
    # min_endpoint*p + min_endpoint*d = 4, but max + tolerance = 5 >= 4,
    # so (1,1)=4 GPUs is feasible inside the band — must NOT zero out.
    new_p, new_d = proportional_clamp_pair(1, 2, 2, 2, 3, 3, 1)
    assert (new_p, new_d) != (0, 0)
    total = new_p * 2 + new_d * 2
    assert 1 <= total <= 5


def test_clamp_pair_ceiling_below_min_endpoint_outside_tolerance_zeros():
    # Same shape but band is too narrow to hold (1,1): max=1, tol=2,
    # band [-1, 3]; min_req=4 > 3, so we still zero correctly.
    assert proportional_clamp_pair(1, 2, 2, 2, 1, 1, 1) == (0, 0)


def test_clamp_pair_zero_inputs_with_floor_distributes():
    new_p, new_d = proportional_clamp_pair(0, 0, 1, 1, 4, 4, 1)
    total = new_p * 1 + new_d * 1
    assert 3 <= total <= 5


def test_clamp_pair_infeasible_band_falls_back_to_inputs():
    # Floor 100, ceiling 4 — infeasible. Should not crash; returns inputs.
    new_p, new_d = proportional_clamp_pair(2, 2, 1, 1, 100, 4, 1)
    # Floor push would land at >> 4 + tol=1 = 5, so we keep inputs.
    assert (new_p, new_d) == (2, 2)


# ---------------------------------------------------------------------------- #
# proportional_clamp_single (agg)                                              #
# ---------------------------------------------------------------------------- #


def test_clamp_single_no_op_when_both_disabled():
    assert proportional_clamp_single(3, 1, -1, -1, 1) == 3


def test_clamp_single_in_band():
    # 3 replicas * 1 GPU = 3, in [2, 5]. With tol=1, [1, 6].
    assert proportional_clamp_single(3, 1, 2, 5, 1) == 3


def test_clamp_single_above_ceiling_strict():
    # ceiling-only (min=-1) → strict, no tolerance.
    assert proportional_clamp_single(10, 1, -1, 4, 1) == 4


def test_clamp_single_below_floor_grows():
    # 1 replica * 1 GPU = 1 < min=4. Both bounds active → tol=1, band [3, 5].
    # Ceiling allows 5 max replicas. Should grow to >=3.
    out = proportional_clamp_single(1, 1, 4, 5, 1)
    assert 3 <= out * 1 <= 5


def test_clamp_single_min_endpoint_clamp():
    assert proportional_clamp_single(0, 1, 4, 5, 2) >= 2


def test_clamp_single_ceiling_below_min_endpoint():
    # ceiling=2 GPUs with min_endpoint=3 replicas of 1 GPU each → can't fit.
    assert proportional_clamp_single(5, 1, -1, 2, 3) == 0


def test_clamp_single_ceiling_below_min_endpoint_within_tolerance_keeps_alive():
    # Reviewer's repro: engine_gpu=4, min_endpoint=1, min=max=3.
    # Tolerance=4, band [-1, 7]. Strict ceiling=3 is below min_endpoint*4=4
    # but max + tolerance = 7 >= 4, so 1 replica (=4 GPUs) is feasible.
    out = proportional_clamp_single(2, 4, 3, 3, 1)
    assert out != 0
    assert out * 4 <= 7  # within max + tolerance
