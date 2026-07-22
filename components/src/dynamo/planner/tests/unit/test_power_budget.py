# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power-budget clamp (Phase 4): pure ceiling math + engine_adapter wiring.

Covers the ceiling-only clamp on projected watts (fit → no-op, proportional
shrink, decode-no-upscale, partial proposals, scale-up-blocked hold) and the
final-boundary ordering guarantee in ``OrchestratorEngineAdapter``: GPU budget
first, then power budget (non-commutative; power wins over the GPU floor).
"""

from types import SimpleNamespace

import pytest

from dynamo.planner.core.budget import (
    apply_power_budget,
    minimum_power_footprint_fits,
    project_watts,
)
from dynamo.planner.core.types import (
    EngineCapabilities,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# project_watts / minimum_power_footprint_fits
# ---------------------------------------------------------------------------


def test_project_watts_sums_present_roles():
    assert project_watts(2, 3, 700, 1200) == 2 * 700 + 3 * 1200
    assert project_watts(2, None, 700, 1200) == 1400
    assert project_watts(None, 3, 700, 1200) == 3600
    assert project_watts(2, 3, None, None) == 0


@pytest.mark.parametrize(
    "budget,min_endpoint,p,d,fits",
    [
        (5000, 1, 700, 1200, True),  # 1900 <= 5000
        (1900, 1, 700, 1200, True),  # exactly fits
        (1899, 1, 700, 1200, False),  # 1900 > 1899
        (1000, 2, 700, 1200, False),  # 2*(700+1200) = 3800 > 1000
        (2000, 1, None, 1200, True),  # only decode present
    ],
)
def test_minimum_power_footprint_fits(budget, min_endpoint, p, d, fits):
    assert minimum_power_footprint_fits(budget, min_endpoint, p, d) is fits


# ---------------------------------------------------------------------------
# apply_power_budget — ceiling clamp
# ---------------------------------------------------------------------------


def test_fit_is_a_no_op():
    assert apply_power_budget(2, 2, 2, 2, 700, 1200, 100000, 1, False) == (2, 2, None)


def test_disagg_over_budget_shrinks_proportionally_to_fit():
    # 3*700 + 3*1200 = 5700 > 5000
    new_p, new_d, reason = apply_power_budget(3, 3, 2, 2, 700, 1200, 5000, 1, False)
    assert reason == "power_budget_clamped"
    assert new_p <= 3 and new_d <= 3  # decode-no-upscale: never raised
    assert new_p * 700 + new_d * 1200 <= 5000


def test_ceiling_never_raises_a_proposed_count():
    # Even with a huge budget, a proposal that already fits is untouched.
    assert apply_power_budget(1, 1, 5, 5, 700, 1200, 100000, 1, False) == (1, 1, None)


def test_partial_proposal_does_not_mutate_unproposed_component():
    # Only prefill proposed; decode fixed at current=2 (2*1200=2400).
    # 4*700 + 2400 = 5200 > 5000, so prefill must shrink; decode stays None.
    new_p, new_d, reason = apply_power_budget(4, None, 2, 2, 700, 1200, 5000, 1, False)
    assert new_d is None  # unproposed decode untouched
    assert new_p * 700 + 2 * 1200 <= 5000
    assert reason is not None


def test_partial_proposal_suppressed_when_unproposed_alone_over_budget():
    # decode current=5 → 5*1200 = 6000 already over the 5500 budget; prefill
    # cannot fit even min_endpoint without changing decode → scale-up refused.
    new_p, new_d, reason = apply_power_budget(4, None, 1, 5, 700, 1200, 5500, 1, False)
    assert new_d is None
    assert new_p == 1  # held at current (no scale-up)
    assert reason == "power_budget_scale_up_suppressed"


def test_scale_up_blocked_holds_at_current():
    # Blocked: scale-up (5,5 vs current 2,2) is held at current.
    assert apply_power_budget(5, 5, 2, 2, 700, 1200, 100000, 1, True) == (
        2,
        2,
        "power_scale_up_blocked",
    )


def test_scale_up_blocked_still_allows_scale_down():
    # Blocked: a scale-down (1,1 vs current 3,3) is honored.
    assert apply_power_budget(1, 1, 3, 3, 700, 1200, 100000, 1, True) == (1, 1, None)


def test_already_over_budget_baseline_with_no_proposal_is_left_alone():
    # Nothing proposed (both None); baseline over budget but no lever this tick.
    assert apply_power_budget(None, None, 10, 10, 700, 1200, 100, 1, False) == (
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# engine_adapter final-boundary ordering (GPU budget then power budget)
# ---------------------------------------------------------------------------


def _bare_adapter(config, capabilities) -> OrchestratorEngineAdapter:
    """An adapter with only the fields the budget path reads (skips heavy init)."""
    adapter = object.__new__(OrchestratorEngineAdapter)
    adapter._config = config
    adapter._capabilities = capabilities
    return adapter


def _agg_config(**overrides):
    base = dict(
        enable_power_awareness=True,
        total_gpu_power_limit=1200,
        min_endpoint=1,
        min_gpu_budget=8,  # GPU floor: 8 GPUs
        max_gpu_budget=-1,  # no GPU ceiling
        mode="agg",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_gpu_budget_then_power_budget_order_is_not_commutative():
    """The GPU floor raises decode to 8, then the power ceiling lowers it to fit
    1200 W — power wins. Applying power first would fit the proposal, then the
    GPU floor would raise it back above budget: the two orders disagree."""
    caps = WorkerCapabilities(
        prefill=None,
        decode=EngineCapabilities(num_gpu=1, power_watts_per_replica=400),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(_agg_config(), caps)
    wc = WorkerCounts(ready_num_decode=2, expected_num_decode=2)  # stable

    # Pipeline order (GPU floor first, then power ceiling): power wins → 3
    # (3 × 400 W = 1200 W fits; the GPU floor of 8 is violated and reported).
    assert adapter._apply_final_budget(None, 2, wc) == (None, 3)

    # Reverse order: power fit leaves 2, then the GPU floor raises to 8 — which
    # is 3200 W, over the 1200 W budget. Different result → non-commutative.
    power_first = adapter._apply_power_final_budget(None, 2, wc)
    reversed_result = adapter._apply_gpu_final_budget(
        power_first[0], power_first[1], wc
    )
    assert reversed_result == (None, 8)
    assert reversed_result != adapter._apply_final_budget(None, 2, wc)


def test_power_clamp_noop_when_awareness_disabled():
    caps = WorkerCapabilities(
        prefill=None,
        decode=EngineCapabilities(num_gpu=1, power_watts_per_replica=400),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(
        _agg_config(enable_power_awareness=False, min_gpu_budget=-1), caps
    )
    wc = WorkerCounts(ready_num_decode=2)
    # No GPU floor, awareness off → proposal passes through untouched.
    assert adapter._apply_final_budget(None, 5, wc) == (None, 5)


def test_power_clamp_blocked_holds_scale_up_at_current():
    caps = WorkerCapabilities(
        prefill=None,
        decode=EngineCapabilities(num_gpu=1, power_watts_per_replica=100),
        power_scale_up_blocked=True,
    )
    adapter = _bare_adapter(
        _agg_config(total_gpu_power_limit=100000, min_gpu_budget=-1), caps
    )
    wc = WorkerCounts(ready_num_decode=2, expected_num_decode=2)  # stable
    # Blocked (not rolling) → scale-up proposal of 6 is held at current 2 by the
    # blocked flag, isolating that path from the rollout hold.
    assert adapter._apply_final_budget(None, 6, wc) == (None, 2)


def _mode_config(mode, **overrides):
    base = dict(
        enable_power_awareness=True,
        total_gpu_power_limit=1200,
        min_endpoint=1,
        min_gpu_budget=-1,
        max_gpu_budget=-1,
        mode=mode,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_final_boundary_clamps_prefill_mode():
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=1, power_watts_per_replica=400),
        decode=None,
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(_mode_config("prefill", total_gpu_power_limit=1200), caps)
    wc = WorkerCounts(ready_num_prefill=2, expected_num_prefill=2)  # stable
    # 6 × 400 W = 2400 W > 1200 W → clamp prefill to floor(1200/400)=3; decode
    # is None (not managed in prefill mode) and stays None.
    assert adapter._apply_final_budget(6, None, wc) == (3, None)


def test_final_boundary_clamps_decode_mode():
    caps = WorkerCapabilities(
        prefill=None,
        decode=EngineCapabilities(num_gpu=1, power_watts_per_replica=300),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(_mode_config("decode", total_gpu_power_limit=900), caps)
    wc = WorkerCounts(ready_num_decode=2, expected_num_decode=2)  # stable
    # 6 × 300 W = 1800 W > 900 W → clamp decode to floor(900/300)=3.
    assert adapter._apply_final_budget(None, 6, wc) == (None, 3)


def test_final_boundary_clamps_disagg_mode():
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=2, power_watts_per_replica=700),
        decode=EngineCapabilities(num_gpu=4, power_watts_per_replica=1200),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(_mode_config("disagg", total_gpu_power_limit=5000), caps)
    wc = WorkerCounts(
        ready_num_prefill=2,
        ready_num_decode=2,
        expected_num_prefill=2,
        expected_num_decode=2,
    )  # stable
    # 4*700 + 4*1200 = 7600 W > 5000 W → proportionally shrunk to fit.
    new_p, new_d = adapter._apply_final_budget(4, 4, wc)
    assert new_p <= 4 and new_d <= 4
    assert new_p * 700 + new_d * 1200 <= 5000


def test_final_boundary_clamps_merged_proposal_regardless_of_source():
    """``_project_scale_to`` applies the power clamp to the merged
    ``final_proposal`` — the single point where builtin and external plugin
    proposals converge — so the budget is enforced no matter which plugin
    produced the counts."""
    caps = WorkerCapabilities(
        prefill=None,
        decode=EngineCapabilities(num_gpu=1, power_watts_per_replica=400),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(_mode_config("agg", total_gpu_power_limit=1200), caps)
    outcome = SimpleNamespace(
        execute_action="apply",
        final_proposal=SimpleNamespace(
            targets=[SimpleNamespace(sub_component_type="decode", replicas=6)]
        ),
    )
    wc = WorkerCounts(ready_num_decode=2, expected_num_decode=2)  # stable
    decision = adapter._project_scale_to(outcome, wc)
    # 6 × 400 W = 2400 W over the 1200 W budget → clamped to 3 in the decision.
    assert decision is not None
    assert decision.num_decode == 3


# The Kubernetes environment uses a single deployment-wide stability flag, so a
# rollout of EITHER role marks BOTH roles' ``expected`` (settled) count None.
# These tests use that reachable state (both None during any rollout).


def test_scale_up_held_while_deployment_is_rolling():
    """Fail closed: while any power-relevant role is mid-rollout (deployment
    unstable → both expected None), a proposed scale-up is held at ready, since
    a rolling role can only be charged at its transient ready count."""
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=2, power_watts_per_replica=700),
        decode=EngineCapabilities(num_gpu=4, power_watts_per_replica=1200),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(_mode_config("disagg", total_gpu_power_limit=5000), caps)
    wc = WorkerCounts(
        ready_num_prefill=2,
        ready_num_decode=2,
        expected_num_prefill=None,  # deployment mid-rollout ->
        expected_num_decode=None,  # both settled targets unknown
    )
    # Prefill proposed 2 -> 4 is held at ready 2 while the deployment rolls.
    assert adapter._apply_final_budget(4, None, wc) == (2, None)


def test_scale_up_allowed_when_deployment_stable():
    """When the deployment is stable (both expected known == ready), a proposal
    is budgeted normally against the known footprint."""
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=2, power_watts_per_replica=700),
        decode=EngineCapabilities(num_gpu=4, power_watts_per_replica=1200),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(_mode_config("disagg", total_gpu_power_limit=6000), caps)
    wc = WorkerCounts(
        ready_num_prefill=2,
        ready_num_decode=2,
        expected_num_prefill=2,
        expected_num_decode=2,  # deployment stable
    )
    # decode at 2 (2400 W); prefill 2->4 (2800 W); total 5200 <= 6000 → allowed.
    assert adapter._apply_final_budget(4, None, wc) == (4, None)


def test_scale_down_allowed_while_deployment_is_rolling():
    """A scale-down is always safe (reduces power), even mid-rollout."""
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=2, power_watts_per_replica=700),
        decode=EngineCapabilities(num_gpu=4, power_watts_per_replica=1200),
        power_scale_up_blocked=False,
    )
    adapter = _bare_adapter(_mode_config("disagg", total_gpu_power_limit=5000), caps)
    wc = WorkerCounts(
        ready_num_prefill=4,
        ready_num_decode=2,
        expected_num_prefill=None,  # deployment mid-rollout
        expected_num_decode=None,
    )
    # Prefill scale-down 4 -> 2 is honored despite the rollout.
    assert adapter._apply_final_budget(2, None, wc) == (2, None)


def test_project_scale_to_holds_scale_up_during_rollout_full_merge():
    """Production-shaped reproduction of the Kubernetes runtime path.

    ``type_aware_merge`` fills a role a plugin omitted from the baseline, so the
    final proposal carries a target for BOTH roles (neither is None). The K8s
    environment marks BOTH ``expected`` counts None during any rollout, so the
    guard must key off that deployment-wide rollout state — not per-role or
    None-target detection — and hold the decode scale-up. Without the fix this
    returned ``ScalingDecision(num_prefill=2, num_decode=2)``."""
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=2, power_watts_per_replica=700),
        decode=EngineCapabilities(num_gpu=4, power_watts_per_replica=1200),
        power_scale_up_blocked=False,
    )
    # Huge budget so only the rollout guard (not the budget clamp) can act.
    adapter = _bare_adapter(_mode_config("disagg", total_gpu_power_limit=100000), caps)
    # Deployment mid-rollout: BOTH expected counts unknown (the reachable state).
    # Merged proposal carries prefill=2 (baseline) and decode=2 (a scale-up).
    wc = WorkerCounts(
        ready_num_prefill=2,
        ready_num_decode=1,
        expected_num_prefill=None,
        expected_num_decode=None,
    )
    outcome = SimpleNamespace(
        execute_action="apply",
        final_proposal=SimpleNamespace(
            targets=[
                SimpleNamespace(sub_component_type="prefill", replicas=2),
                SimpleNamespace(sub_component_type="decode", replicas=2),
            ]
        ),
    )
    decision = adapter._project_scale_to(outcome, wc)
    # decode must NOT be scaled up to 2 while the deployment rolls (held at 1);
    # the held result equals the current counts, so no new decision is issued.
    assert decision is None or (decision.num_decode or 0) <= 1


def test_project_scale_to_masks_ready_echo_during_rollout_scale_down():
    """A scale-down of one role must not drag the other role's ready echo.

    Production-shaped: prefill is rolling from ready=2 toward a larger desired
    (deployment unstable → both ``expected`` None), while decode legitimately
    scales down 3 -> 2. ``type_aware_merge`` echoes prefill's baseline == ready 2
    into the merged proposal, so before masking ``_project_scale_to`` returned
    ``ScalingDecision(num_prefill=2, num_decode=2)``; DisaggPlanner applies every
    non-None target, writing prefill's DGD desired back to 2 and cancelling its
    in-flight rollout. The ready-echo mask drops prefill to None while preserving
    the decode scale-down."""
    caps = WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=2, power_watts_per_replica=700),
        decode=EngineCapabilities(num_gpu=4, power_watts_per_replica=1200),
        power_scale_up_blocked=False,
    )
    # Budget large enough that the decode scale-down is never the constraint.
    adapter = _bare_adapter(_mode_config("disagg", total_gpu_power_limit=100000), caps)
    wc = WorkerCounts(
        ready_num_prefill=2,
        ready_num_decode=3,
        expected_num_prefill=None,  # deployment mid-rollout (prefill 2 -> desired)
        expected_num_decode=None,
    )
    outcome = SimpleNamespace(
        execute_action="apply",
        final_proposal=SimpleNamespace(
            targets=[
                # prefill baseline echo == ready 2 (not an intended change)
                SimpleNamespace(sub_component_type="prefill", replicas=2),
                # decode scale-down 3 -> 2
                SimpleNamespace(sub_component_type="decode", replicas=2),
            ]
        ),
    )
    decision = adapter._project_scale_to(outcome, wc)
    # prefill's ready echo is masked to None (its rollout desired is left
    # untouched); the decode scale-down survives.
    assert decision is not None
    assert decision.num_prefill is None
    assert decision.num_decode == 2
