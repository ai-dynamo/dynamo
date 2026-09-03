# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared GPU budget primitives in
``dynamo.planner.core.budget``."""

from types import SimpleNamespace

import pytest

from dynamo.planner.core.budget import (
    bounds_for_total,
    compute_tolerance,
    guard_disagg_load_budget,
    proportional_clamp_pair,
    proportional_clamp_single,
)
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities

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


def test_bounds_tolerance_relaxes_only_lower_edge():
    # tol=2 widens the band to [2, 8] — max is a hard cap, never relaxed.
    assert bounds_for_total(2, 4, 8, 2) == (True, "")
    assert bounds_for_total(8, 4, 8, 2) == (True, "")
    in_band, reason = bounds_for_total(9, 4, 8, 2)
    assert in_band is False
    assert "exceeds ceiling" in reason
    # No "+ tol N" on the ceiling reason — tolerance is lower-only.
    assert "tol" not in reason
    in_band, reason = bounds_for_total(1, 4, 8, 2)
    assert in_band is False
    assert "- tol 2" in reason


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


def test_clamp_pair_preserves_asymmetric_component_minimums():
    assert proportional_clamp_pair(10, 10, 1, 1, -1, 5, 2, 1) == (2, 3)


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


def test_clamp_pair_asymmetric_overshoot_shrinks_to_strict_ceiling():
    # prefill 1 GPU/worker, decode 2 GPU/worker. min=max=4. tol=2 lowers
    # only the floor → band [2, 4]. Desired (1, 2) = 5 overshoots the hard
    # cap and must be shrunk back to <= 4.
    new_p, new_d = proportional_clamp_pair(1, 2, 1, 2, 4, 4, 1)
    assert new_p * 1 + new_d * 2 <= 4
    assert new_p >= 1 and new_d >= 1  # min_endpoint preserved when feasible


def test_clamp_pair_asymmetric_floor_grows():
    # prefill=1, decode=2. min=max=5. desired (1,1)=3 < min=5. tol=2 → [3, 5].
    # Floor logic pushes up; result must stay at or below the strict ceiling.
    new_p, new_d = proportional_clamp_pair(1, 1, 1, 2, 5, 5, 1)
    total = new_p * 1 + new_d * 2
    assert 3 <= total <= 5


def test_clamp_pair_asymmetric_unreachable_target_converges():
    # Both pools = 2 GPU/worker. min=max=5 unreachable (totals are even).
    # tol=2 only on lower → band [3, 5]. Should land at 4 (largest feasible
    # multiple of 2 that doesn't exceed the hard cap of 5).
    new_p, new_d = proportional_clamp_pair(1, 1, 2, 2, 5, 5, 1)
    total = new_p * 2 + new_d * 2
    assert total == 4


def test_clamp_pair_no_oscillation_after_one_pass():
    # Calling clamp again on its own output must yield the same result.
    p1, d1 = proportional_clamp_pair(1, 1, 2, 2, 5, 5, 1)
    p2, d2 = proportional_clamp_pair(p1, d1, 2, 2, 5, 5, 1)
    assert (p1, d1) == (p2, d2)


def test_clamp_pair_asymmetric_min_max_open_band():
    # min=4, max=10, prefill=1, decode=2. desired (1, 1) = 3 < min.
    # Tolerance applies only to lower edge → band [2, 10].
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


def test_clamp_pair_ceiling_below_min_endpoint_zeros():
    # p_gpu=2, d_gpu=2, min=max=3, min_endpoint=1. min_endpoint of each pool
    # would need 4 GPUs but the hard cap is 3. Configuration is infeasible
    # — must zero the deployment rather than overshoot the hard cap.
    assert proportional_clamp_pair(1, 2, 2, 2, 3, 3, 1) == (0, 0)


def test_clamp_pair_ceiling_below_min_endpoint_far_outside_zeros():
    # Same shape with an even tighter ceiling — also zeroes.
    assert proportional_clamp_pair(1, 2, 2, 2, 1, 1, 1) == (0, 0)


def test_clamp_pair_zero_inputs_with_floor_distributes():
    new_p, new_d = proportional_clamp_pair(0, 0, 1, 1, 4, 4, 1)
    total = new_p * 1 + new_d * 1
    assert 3 <= total <= 5


def test_clamp_pair_infeasible_band_falls_back_to_inputs():
    # Floor 100, ceiling 4 — infeasible. Should not crash; returns inputs.
    new_p, new_d = proportional_clamp_pair(2, 2, 1, 1, 100, 4, 1)
    # Floor push would land far above the strict ceiling (4), so we keep
    # inputs and let the caller surface the config error.
    assert (new_p, new_d) == (2, 2)


# ---------------------------------------------------------------------------- #
# disaggregated load-budget safety guard                                       #
# ---------------------------------------------------------------------------- #


def _guard(
    current: tuple[int, int],
    proposed: tuple[int, int],
    *,
    p_gpu: int = 1,
    d_gpu: int = 1,
    min_gpus: int = 64,
    max_gpus: int = 64,
    prefill_min_endpoint: int = 1,
    decode_min_endpoint: int = 1,
) -> tuple[int, int, str | None]:
    return guard_disagg_load_budget(
        current[0],
        current[1],
        proposed[0],
        proposed[1],
        p_gpu,
        d_gpu,
        min_gpus,
        max_gpus,
        prefill_min_endpoint,
        decode_min_endpoint,
    )


@pytest.mark.parametrize(
    ("current", "proposed"),
    [
        ((28, 36), (27, 36)),
        ((27, 36), (26, 35)),
    ],
)
def test_load_budget_holds_scale_down_without_scale_up_at_floor(current, proposed):
    assert _guard(current, proposed) == (
        current[0],
        current[1],
        "gpu_budget_guard_hold",
    )


def test_load_budget_allows_explicit_opposing_swap_at_fixed_budget():
    assert _guard((27, 37), (26, 38)) == (26, 38, None)


def test_load_budget_allows_reverse_explicit_opposing_swap_at_fixed_budget():
    assert _guard((27, 37), (28, 36)) == (28, 36, None)


@pytest.mark.parametrize("proposed", [(28, 37), (29, 37)])
def test_load_budget_does_not_invent_donor_at_full_budget(proposed):
    # hold/up and up/up do not establish that either pool is safe to donate.
    assert _guard((28, 36), proposed) == (
        28,
        36,
        "gpu_budget_guard_hold",
    )


def test_load_budget_rejects_opposing_swap_that_does_not_fit():
    # Removing one 1-GPU prefill cannot fund one 2-GPU decode.
    assert _guard(
        (2, 1),
        (1, 2),
        p_gpu=1,
        d_gpu=2,
        min_gpus=4,
        max_gpus=4,
    ) == (2, 1, "gpu_budget_guard_hold")


def test_load_budget_does_not_execute_only_donor_leg_of_weighted_swap():
    # The generic clamp turns the 7-GPU proposal into (1P, 1D) = 4 GPUs,
    # suppressing decode-up but retaining prefill-down because tolerance makes
    # 4 valid for a min=5 floor. At the floor, the swap must be atomic.
    assert _guard(
        (2, 1),
        (1, 2),
        p_gpu=1,
        d_gpu=3,
        min_gpus=5,
        max_gpus=6,
    ) == (2, 1, "gpu_budget_guard_hold")


def test_load_budget_allows_scale_down_above_floor():
    assert _guard((35, 35), (34, 35), max_gpus=80) == (34, 35, None)


def test_load_budget_disabled_minimum_preserves_legacy_clamp():
    expected = proportional_clamp_pair(3, 2, 1, 1, -1, 4, 1, 1)
    actual_p, actual_d, reason = _guard(
        (3, 1),
        (3, 2),
        min_gpus=-1,
        max_gpus=4,
    )
    assert (actual_p, actual_d) == expected
    assert reason is None


def test_load_budget_weighted_tolerance_does_not_enable_donor_only_churn():
    # min=max=5 is unreachable with 2-GPU replicas. The existing tolerance
    # accepts total=4, but the nominal floor still guards donor-only removal.
    assert _guard(
        (1, 1),
        (0, 1),
        p_gpu=2,
        d_gpu=2,
        min_gpus=5,
        max_gpus=5,
        prefill_min_endpoint=0,
    ) == (1, 1, "gpu_budget_guard_hold")


def test_load_budget_weighted_tolerance_allows_feasible_opposing_swap():
    assert _guard(
        (1, 1),
        (0, 2),
        p_gpu=2,
        d_gpu=2,
        min_gpus=5,
        max_gpus=5,
        prefill_min_endpoint=0,
    ) == (0, 2, None)


def test_load_budget_clamp_never_inverts_hold_into_scale_up():
    # The proportional floor clamp would turn (1P, 3D) into (2P, 4D),
    # inventing a decode scale-up from a hold signal. Fail closed instead.
    assert _guard(
        (4, 3),
        (1, 3),
        min_gpus=6,
        max_gpus=10,
    ) == (4, 3, "gpu_budget_guard_hold")


def test_load_budget_holds_when_integer_width_clamp_cannot_reach_band():
    # Current total 5 is valid in the tolerance-relaxed [5, 7] band. The
    # proportional primitive cannot lift the proposed total 4 without
    # overshooting the hard max, so it returns the proposal unchanged. The
    # load guard must not dispatch that out-of-band result.
    assert _guard(
        (1, 2),
        (2, 1),
        p_gpu=1,
        d_gpu=2,
        min_gpus=7,
        max_gpus=7,
    ) == (1, 2, "gpu_budget_guard_hold")


def test_load_budget_below_floor_uses_explicit_reconcile_from_current():
    assert _guard(
        (1, 1),
        (0, 1),
        min_gpus=4,
        max_gpus=6,
        prefill_min_endpoint=1,
    ) == (2, 2, "gpu_budget_reconcile")


# ---------------------------------------------------------------------------- #
# proportional_clamp_single (agg)                                              #
# ---------------------------------------------------------------------------- #


def test_clamp_single_no_op_when_both_disabled():
    assert proportional_clamp_single(3, 1, -1, -1, 1) == 3


def test_clamp_single_in_band():
    # 3 replicas * 1 GPU = 3, in [2, 5]. With tol=1 (lower-only), [1, 5].
    assert proportional_clamp_single(3, 1, 2, 5, 1) == 3


def test_clamp_single_above_ceiling_strict():
    # ceiling-only (min=-1) → strict, no tolerance.
    assert proportional_clamp_single(10, 1, -1, 4, 1) == 4


def test_clamp_single_below_floor_grows():
    # 1 replica * 1 GPU = 1 < min=4. Both bounds active → tol=1 (lower-only),
    # band [3, 5]. Should grow to >=3 and stay <= 5.
    out = proportional_clamp_single(1, 1, 4, 5, 1)
    assert 3 <= out * 1 <= 5


def test_clamp_single_min_endpoint_clamp():
    assert proportional_clamp_single(0, 1, 4, 5, 2) >= 2


def test_clamp_single_ceiling_below_min_endpoint():
    # ceiling=2 GPUs with min_endpoint=3 replicas of 1 GPU each → can't fit.
    assert proportional_clamp_single(5, 1, -1, 2, 3) == 0


def test_clamp_single_ceiling_below_min_endpoint_zeros():
    # engine_gpu=4, min_endpoint=1, min=max=3. Strict ceiling=3 is below
    # min_endpoint*4=4 — even one replica overshoots the hard cap, so the
    # deployment is infeasible and must be zeroed.
    assert proportional_clamp_single(2, 4, 3, 3, 1) == 0


@pytest.mark.parametrize(
    ("gpu_cost_per_replica", "expected"),
    [
        (None, 2),
        (4, 2),
        (5, 1),
    ],
)
def test_planner_budget_uses_replica_cost_not_engine_width(
    gpu_cost_per_replica, expected
):
    """A zero-GPU sidecar preserves the old limit; one GPU lowers it."""
    state = PlannerScalingState.__new__(PlannerScalingState)
    state._config = SimpleNamespace(
        min_gpu_budget=-1,
        max_gpu_budget=8,
        min_endpoint=1,
        prefill_min_endpoint=None,
        decode_min_endpoint=None,
    )
    state._capabilities = WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=4,
            gpu_cost_per_replica=gpu_cost_per_replica,
        )
    )

    assert state._apply_single_budget(3, "decode") == expected
