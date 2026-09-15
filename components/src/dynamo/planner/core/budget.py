# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU budget enforcement primitives.

Two layers:

* **Pure math** (``compute_tolerance``, ``bounds_for_total``, the directional
  scaling guards, the proportional clamps, and the power-budget helpers
  below): no I/O, no state, no logging. Shared by
  the builtin local planner state (where the budget is enforced
  intra-DGD by clamping the joint ``(num_prefill, num_decode)`` desired
  counts), the orchestrator engine adapter's final budget clamp, and the
  centralized GlobalPlanner (where it is enforced across DGDs by
  accepting/rejecting incoming ScaleRequests). Callers share the same
  ``tolerance`` / in-band check; only the action taken on a breach
  differs (local transforms counts, GlobalPlanner decides).

* ``_initialize_gpu_counts`` remains a deployment-bootstrap helper: it
  populates per-engine GPU counts from the DGD spec or CLI flags, with
  a virtual-mode fallback.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.errors import DeploymentValidationError
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Pure primitives — no I/O, shared between local and global planner.           #
# ---------------------------------------------------------------------------- #


def compute_tolerance(gpu_per_replicas: Iterable[int]) -> int:
    """Tolerance for a budget band when the pools that are actually changing
    have different ``gpu_per_replica`` step sizes.

    Returns ``max(gpu_per_replicas)`` over positive entries, or ``0`` if the
    iterable is empty / all non-positive.

    Why: integer worker steps from one pool can't always exactly cancel the
    integer worker steps from another pool. Example with prefill=2 GPU/worker
    and decode=2 GPU/worker, ``min == max == 5`` is unreachable — totals
    can only be 0, 2, 4, 6, ... — so a strict bounds check would oscillate.
    Allowing the result to land within ``±tolerance`` lets the algorithm
    converge in a single pass.
    """
    gpus = [g for g in gpu_per_replicas if g > 0]
    return max(gpus, default=0)


def bounds_for_total(
    total: int,
    min_gpus: int,
    max_gpus: int,
    tolerance: int,
) -> tuple[bool, str]:
    """Pure check: does ``total`` fit ``[min_gpus - tolerance, max_gpus]``?

    A negative ``min_gpus`` disables the floor. A negative ``max_gpus``
    disables the ceiling. ``tolerance == 0`` enforces a strict floor.

    ``max_gpus`` is a hard hardware/capacity bound and is **never** relaxed —
    overshooting it would risk pending pods or over-admission. Tolerance
    relaxes only the lower bound, to handle integer-step granularity where
    pool changes can't always exactly cancel.

    Returns ``(in_bounds, reason_if_out)``. ``reason`` is empty when in bounds.
    """
    if max_gpus >= 0:
        if total > max_gpus:
            return (False, f"total {total} exceeds ceiling ({max_gpus})")
    if min_gpus >= 0:
        lo = min_gpus - tolerance
        if total < lo:
            return (
                False,
                f"total {total} below floor "
                f"({min_gpus}{f' - tol {tolerance}' if tolerance else ''})",
            )
    return (True, "")


def proportional_clamp_pair(
    num_p: int,
    num_d: int,
    p_gpu: int,
    d_gpu: int,
    min_gpus: int,
    max_gpus: int,
    prefill_min_endpoint: int,
    decode_min_endpoint: Optional[int] = None,
) -> tuple[int, int]:
    """Clamp ``(num_p, num_d)`` so total GPUs lands in the budget band.

    The band is ``[min_gpus - tolerance, max_gpus]`` when both bounds are
    active, and strictly ``[0, max_gpus]`` or ``[min_gpus, +inf)`` when only
    one bound is active. ``tolerance`` is computed internally as
    ``max(p_gpu, d_gpu)`` and only relaxes the lower bound — ``max_gpus`` is
    a hard hardware/capacity bound and is never relaxed.

    Distribution policy is proportional in both directions (mirror of the
    historical proportional shrink). SLA-pressure-aware split is a future
    enhancement.

    Negative ``min_gpus`` or ``max_gpus`` disables the corresponding bound.
    Returns ``(num_p, num_d)`` unchanged if both are disabled or if either
    per-replica GPU count is non-positive (caller hasn't initialized
    capabilities yet). Returns ``(0, 0)`` when the component-specific minimum
    footprint would overshoot the hard ceiling (configuration is infeasible).
    """
    if decode_min_endpoint is None:
        decode_min_endpoint = prefill_min_endpoint

    if min_gpus < 0 and max_gpus < 0:
        return num_p, num_d
    if p_gpu <= 0 or d_gpu <= 0:
        return num_p, num_d

    total = num_p * p_gpu + num_d * d_gpu
    tolerance = (
        compute_tolerance([p_gpu, d_gpu]) if (min_gpus >= 0 and max_gpus >= 0) else 0
    )

    in_band, _ = bounds_for_total(total, min_gpus, max_gpus, tolerance)
    if in_band:
        return num_p, num_d

    # Ceiling path — strict shrink. ``max_gpus`` is a hard cap; if even
    # component-specific minimum footprint overshoots it, the deployment is infeasible
    # and we zero out (the caller is responsible for surfacing the config
    # error). Otherwise proportionally shrink to fit under ``max_gpus``.
    if max_gpus >= 0 and total > max_gpus:
        min_req = prefill_min_endpoint * p_gpu + decode_min_endpoint * d_gpu
        if max_gpus < min_req:
            return 0, 0
        target = max_gpus
        scale = target / total
        max_p = math.floor((target - decode_min_endpoint * d_gpu) / p_gpu)
        new_p = max(prefill_min_endpoint, min(max_p, math.floor(num_p * scale)))
        remaining = target - new_p * p_gpu
        new_d = max(decode_min_endpoint, math.floor(remaining / d_gpu))
        return new_p, new_d

    # Floor path — proportional grow toward min_gpus.
    floor = min_gpus
    if total <= 0:
        # No prior allocation — split the floor roughly evenly across the
        # two pools, biasing the remainder toward decode.
        new_p = max(prefill_min_endpoint, math.ceil(floor / 2 / p_gpu))
        remaining = max(0, floor - new_p * p_gpu)
        new_d = max(decode_min_endpoint, math.ceil(remaining / d_gpu))
    else:
        scale = floor / total
        new_p = max(prefill_min_endpoint, math.ceil(num_p * scale))
        remaining = max(0, floor - new_p * p_gpu)
        new_d = max(decode_min_endpoint, math.ceil(remaining / d_gpu))

    # If the floor push would blow past the strict ceiling, the configuration
    # is infeasible (tight bounds incompatible with the step sizes). Best
    # effort: keep the inputs unchanged and let the caller log; this
    # function stays pure.
    if max_gpus >= 0 and (new_p * p_gpu + new_d * d_gpu) > max_gpus:
        return num_p, num_d

    return new_p, new_d


def fit_directional_budget_pair(
    current_p: int,
    current_d: int,
    proposed_p: int,
    proposed_d: int,
    p_gpu: int,
    d_gpu: int,
    min_gpus: int,
    max_gpus: int,
    prefill_min_endpoint: int,
    decode_min_endpoint: int,
) -> tuple[int, int]:
    """Fit a bounded proposal to the GPU band without changing its directions.

    Every returned component lies between its current and proposed counts. For
    an opposing rebalance, both legs move or neither moves so a GPU-width
    mismatch can never leave only the donor removal. Callers bound throughput
    deltas before invoking this helper, keeping the candidate set small.
    """
    if p_gpu <= 0 or d_gpu <= 0:
        return proposed_p, proposed_d

    tolerance = (
        compute_tolerance([p_gpu, d_gpu]) if min_gpus >= 0 and max_gpus >= 0 else 0
    )
    p_down = proposed_p < current_p
    p_up = proposed_p > current_p
    d_down = proposed_d < current_d
    d_up = proposed_d > current_d
    opposing_rebalance = (p_down and d_up) or (p_up and d_down)
    requested_p = abs(proposed_p - current_p)
    requested_d = abs(proposed_d - current_d)

    def _ceil_div(numerator: int, denominator: int) -> int:
        return -((-numerator) // denominator)

    p_low = max(min(current_p, proposed_p), prefill_min_endpoint)
    p_high = max(current_p, proposed_p)
    d_low = max(min(current_d, proposed_d), decode_min_endpoint)
    d_high = max(current_d, proposed_d)

    # Counts outside these bounds cannot possibly satisfy the band, so do not
    # let an extreme configured cap or stale cached floor expand the scan past
    # the actual resource envelope.
    if max_gpus >= 0:
        p_high = min(p_high, (max_gpus - d_low * d_gpu) // p_gpu)
        d_high = min(d_high, (max_gpus - p_low * p_gpu) // d_gpu)
    if min_gpus >= 0:
        lower_budget = min_gpus - tolerance
        p_low = max(p_low, _ceil_div(lower_budget - d_high * d_gpu, p_gpu))
        d_low = max(d_low, _ceil_div(lower_budget - p_high * p_gpu, d_gpu))
    if p_low > p_high or d_low > d_high:
        return current_p, current_d

    proposed_in_band, _ = bounds_for_total(
        proposed_p * p_gpu + proposed_d * d_gpu,
        min_gpus,
        max_gpus,
        tolerance,
    )
    if (
        proposed_p >= prefill_min_endpoint
        and proposed_d >= decode_min_endpoint
        and proposed_in_band
    ):
        return proposed_p, proposed_d

    best: Optional[tuple[tuple[int, int, int, int, int], tuple[int, int]]] = None

    def _consider(candidate_p: int, candidate_d: int) -> None:
        nonlocal best
        total = candidate_p * p_gpu + candidate_d * d_gpu
        in_band, _ = bounds_for_total(total, min_gpus, max_gpus, tolerance)
        if not in_band:
            return
        distance = abs(proposed_p - candidate_p) + abs(proposed_d - candidate_d)
        progress_p = abs(candidate_p - current_p)
        progress_d = abs(candidate_d - current_d)
        proportional_skew = abs(progress_p * requested_d - progress_d * requested_p)
        score = (
            distance,
            proportional_skew,
            -(progress_p + progress_d),
            candidate_p,
            candidate_d,
        )
        if best is None or score < best[0]:
            best = (score, (candidate_p, candidate_d))

    effective_min: Optional[int] = min_gpus - tolerance if min_gpus >= 0 else None

    # Enumerate only the shorter component range. For each value, the budget
    # band defines one interval for the other component, whose closest point
    # to the proposal is the only candidate that can minimize the primary
    # distance score. This reduces the old Cartesian O(P*D) search to
    # O(min(P,D)); the normal cap=8 path is at most nine iterations.
    if p_high - p_low <= d_high - d_low:
        for candidate_p in range(p_low, p_high + 1):
            candidate_d_low, candidate_d_high = d_low, d_high
            p_cost = candidate_p * p_gpu
            if effective_min is not None:
                candidate_d_low = max(
                    candidate_d_low,
                    _ceil_div(effective_min - p_cost, d_gpu),
                )
            if max_gpus >= 0:
                candidate_d_high = min(
                    candidate_d_high,
                    (max_gpus - p_cost) // d_gpu,
                )
            if opposing_rebalance:
                if candidate_p == current_p:
                    if candidate_d_low <= current_d <= candidate_d_high:
                        _consider(candidate_p, current_d)
                    continue
                if proposed_d > current_d:
                    candidate_d_low = max(candidate_d_low, current_d + 1)
                else:
                    candidate_d_high = min(candidate_d_high, current_d - 1)
            if candidate_d_low <= candidate_d_high:
                candidate_d = min(
                    max(proposed_d, candidate_d_low),
                    candidate_d_high,
                )
                _consider(candidate_p, candidate_d)
    else:
        for candidate_d in range(d_low, d_high + 1):
            candidate_p_low, candidate_p_high = p_low, p_high
            d_cost = candidate_d * d_gpu
            if effective_min is not None:
                candidate_p_low = max(
                    candidate_p_low,
                    _ceil_div(effective_min - d_cost, p_gpu),
                )
            if max_gpus >= 0:
                candidate_p_high = min(
                    candidate_p_high,
                    (max_gpus - d_cost) // p_gpu,
                )
            if opposing_rebalance:
                if candidate_d == current_d:
                    if candidate_p_low <= current_p <= candidate_p_high:
                        _consider(current_p, candidate_d)
                    continue
                if proposed_p > current_p:
                    candidate_p_low = max(candidate_p_low, current_p + 1)
                else:
                    candidate_p_high = min(candidate_p_high, current_p - 1)
            if candidate_p_low <= candidate_p_high:
                candidate_p = min(
                    max(proposed_p, candidate_p_low),
                    candidate_p_high,
                )
                _consider(candidate_p, candidate_d)

    return (current_p, current_d) if best is None else best[1]


def guard_single_scaling_budget(
    current: int,
    proposed: int,
    gpu_cost: int,
    min_gpus: int,
    max_gpus: int,
    min_endpoint: int,
) -> tuple[int, Optional[str]]:
    """Apply a GPU band to one bounded proposal without reversing direction."""
    if gpu_cost <= 0:
        return proposed, None

    tolerance = gpu_cost if min_gpus >= 0 and max_gpus >= 0 else 0
    current_total = current * gpu_cost
    current_in_band, _ = bounds_for_total(current_total, min_gpus, max_gpus, tolerance)
    if not current_in_band:
        reconciled = proportional_clamp_single(
            current, gpu_cost, min_gpus, max_gpus, min_endpoint
        )
        return reconciled, "gpu_budget_reconcile"

    if min_gpus >= 0 and current_total <= min_gpus and proposed < current:
        return current, "gpu_budget_guard_hold"
    if max_gpus >= 0 and current_total >= max_gpus and proposed > current:
        return current, "gpu_budget_guard_hold"

    final = proportional_clamp_single(
        proposed, gpu_cost, min_gpus, max_gpus, min_endpoint
    )
    if not min(current, proposed) <= final <= max(current, proposed):
        return current, "gpu_budget_guard_hold"
    return final, None


def guard_disagg_scaling_budget(
    current_p: int,
    current_d: int,
    proposed_p: int,
    proposed_d: int,
    p_gpu: int,
    d_gpu: int,
    min_gpus: int,
    max_gpus: int,
    prefill_min_endpoint: int,
    decode_min_endpoint: int,
) -> tuple[int, int, Optional[str]]:
    """Apply GPU budgets to a bounded disaggregated scaling proposal safely.

    ``min_gpu_budget`` is a scale-down guard, not an independent source of
    scale-up intent. Once the current allocation has reached that floor, a
    down/down or down/hold proposal is held instead of being proportionally
    grown into a direction-reversing action. At a full (maximum-budget)
    allocation, a scale-up is admitted only as an explicit opposing rebalance:
    one pool must propose up while the other proposes down, and the proposed
    pair itself must fit the budget band.

    If the observed allocation is already outside the budget band, recovery is
    kept explicit: the current allocation, rather than the load proposal, is
    reconciled back toward the band and ``"gpu_budget_reconcile"`` is returned.
    A negative minimum disables only the floor; the hard ceiling remains
    direction-preserving and never invents a donor.

    The returned optional reason is suitable for load-decision diagnostics.
    """
    if p_gpu <= 0 or d_gpu <= 0:
        new_p, new_d = proportional_clamp_pair(
            proposed_p,
            proposed_d,
            p_gpu,
            d_gpu,
            min_gpus,
            max_gpus,
            prefill_min_endpoint,
            decode_min_endpoint,
        )
        return new_p, new_d, None

    tolerance = compute_tolerance([p_gpu, d_gpu]) if max_gpus >= 0 else 0
    current_total = current_p * p_gpu + current_d * d_gpu
    current_in_band, _ = bounds_for_total(current_total, min_gpus, max_gpus, tolerance)

    # Budget recovery is not an ordinary load decision. Reconcile from the
    # observed allocation so a stale down proposal cannot dictate which pool
    # receives recovery capacity (or turn the recovery into churn).
    if not current_in_band:
        new_p, new_d = proportional_clamp_pair(
            current_p,
            current_d,
            p_gpu,
            d_gpu,
            min_gpus,
            max_gpus,
            prefill_min_endpoint,
            decode_min_endpoint,
        )
        return new_p, new_d, "gpu_budget_reconcile"

    p_down = proposed_p < current_p
    p_up = proposed_p > current_p
    d_down = proposed_d < current_d
    d_up = proposed_d > current_d
    any_down = p_down or d_down
    any_up = p_up or d_up
    opposing_rebalance = (p_down and d_up) or (p_up and d_down)

    # The nominal floor is the configured safety intent. Tolerance only makes
    # integer GPU-width targets feasible; it must not permit repeated
    # scale-down-only actions once the allocation has reached the floor.
    if current_total <= min_gpus and any_down and not any_up:
        return current_p, current_d, "gpu_budget_guard_hold"

    # At the floor, keep opposing movements atomic. Fit the largest useful
    # direction-preserving sub-pair; if no receiver can move with its donor,
    # fail closed instead of executing only the removal.
    if current_total <= min_gpus and opposing_rebalance:
        fitted_p, fitted_d = fit_directional_budget_pair(
            current_p,
            current_d,
            proposed_p,
            proposed_d,
            p_gpu,
            d_gpu,
            min_gpus,
            max_gpus,
            prefill_min_endpoint,
            decode_min_endpoint,
        )
        if (fitted_p, fitted_d) == (current_p, current_d):
            return current_p, current_d, "gpu_budget_guard_hold"
        return fitted_p, fitted_d, None

    # At the hard ceiling there is no unallocated capacity for a scale-up.
    # Refuse to invent a donor from a pool whose load signal said hold/up.
    # For an explicit opposing proposal, fit both legs together so a clamp can
    # never suppress the receiver and execute only the donor removal.
    if max_gpus >= 0 and current_total >= max_gpus and any_up:
        if not opposing_rebalance:
            return current_p, current_d, "gpu_budget_guard_hold"
        fitted_p, fitted_d = fit_directional_budget_pair(
            current_p,
            current_d,
            proposed_p,
            proposed_d,
            p_gpu,
            d_gpu,
            min_gpus,
            max_gpus,
            prefill_min_endpoint,
            decode_min_endpoint,
        )
        if (fitted_p, fitted_d) == (current_p, current_d):
            return current_p, current_d, "gpu_budget_guard_hold"
        return fitted_p, fitted_d, None

    new_p, new_d = proportional_clamp_pair(
        proposed_p,
        proposed_d,
        p_gpu,
        d_gpu,
        min_gpus,
        max_gpus,
        prefill_min_endpoint,
        decode_min_endpoint,
    )

    # Constraints may suppress a requested movement, but must not reverse or
    # amplify it. An explicit recovery path above handles out-of-band current
    # allocations; ordinary in-band load decisions fail closed to HOLD.
    def _within_requested_direction(current: int, proposed: int, final: int) -> bool:
        return min(current, proposed) <= final <= max(current, proposed)

    if not _within_requested_direction(current_p, proposed_p, new_p) or not (
        _within_requested_direction(current_d, proposed_d, new_d)
    ):
        return current_p, current_d, "gpu_budget_guard_hold"

    new_total = new_p * p_gpu + new_d * d_gpu
    new_in_band, _ = bounds_for_total(new_total, min_gpus, max_gpus, tolerance)
    if not new_in_band:
        return current_p, current_d, "gpu_budget_guard_hold"

    return new_p, new_d, None


def proportional_clamp_single(
    desired: int,
    engine_gpu: int,
    min_gpus: int,
    max_gpus: int,
    min_endpoint: int,
) -> int:
    """Single-pool variant for agg mode.

    Tolerance equals ``engine_gpu`` automatically when both bounds are
    active, and relaxes only the lower bound. ``max_gpus`` is a hard cap.

    Negative ``min_gpus`` or ``max_gpus`` disables the corresponding bound.
    Returns ``0`` when even ``min_endpoint`` replicas would overshoot the
    hard ceiling (configuration is infeasible).
    """
    if min_gpus < 0 and max_gpus < 0:
        return desired
    if engine_gpu <= 0:
        return desired

    total = desired * engine_gpu
    tolerance = engine_gpu if (min_gpus >= 0 and max_gpus >= 0) else 0

    in_band, _ = bounds_for_total(total, min_gpus, max_gpus, tolerance)
    if in_band:
        return desired

    if max_gpus >= 0 and total > max_gpus:
        min_req = min_endpoint * engine_gpu
        if max_gpus < min_req:
            return 0
        return max(min_endpoint, math.floor(max_gpus / engine_gpu))

    # total < min_gpus - tolerance
    return max(min_endpoint, math.ceil(min_gpus / engine_gpu))


# ---------------------------------------------------------------------------- #
# Power budget — pure ceiling clamp on PROJECTED watts (no floor).             #
#                                                                              #
# The per-GPU caps are DGD-owned; the planner reads ``watts_per_replica`` per  #
# role (from the *requested* annotation) and a ``total_gpu_power_limit`` and   #
# clamps proposed replica counts so projected watts fit the budget. This is a  #
# ceiling on the projected draw of the requested caps — not a proven hardware  #
# limit (the Power Agent may clamp a cap up to the GPU minimum or fail to      #
# apply it, and does not feed the effective cap back here). Within that model  #
# it is treated as a hard constraint: it only ever *lowers* counts and,        #
# applied after the GPU-budget clamp, wins over the GPU floor when the two     #
# conflict (the floor violation is reported, not enforced).                    #
# ---------------------------------------------------------------------------- #


def project_watts(
    num_p: Optional[int],
    num_d: Optional[int],
    p_watts: Optional[int],
    d_watts: Optional[int],
) -> int:
    """Projected watts = Σ count × per-replica watts. Missing count/watts = 0."""
    total = 0
    if num_p is not None and p_watts is not None:
        total += num_p * p_watts
    if num_d is not None and d_watts is not None:
        total += num_d * d_watts
    return total


def peak_parallel_watts(
    current_p: Optional[int],
    current_d: Optional[int],
    proposed_p: Optional[int],
    proposed_d: Optional[int],
    p_watts: Optional[int],
    d_watts: Optional[int],
) -> int:
    """Worst-case draw if both roles move toward their targets in parallel."""
    p_ready = current_p or 0
    d_ready = current_d or 0
    p_peak = max(p_ready, proposed_p if proposed_p is not None else p_ready)
    d_peak = max(d_ready, proposed_d if proposed_d is not None else d_ready)
    return project_watts(p_peak, d_peak, p_watts, d_watts)


def _is_opposing_rebalance(
    proposed_p: Optional[int],
    proposed_d: Optional[int],
    current_p: Optional[int],
    current_d: Optional[int],
) -> bool:
    """True when one role scales up and the other scales down.

    ``None`` current means no replicas are seated yet, treated as 0 — matching
    ``_hold_at_current`` and ``peak_parallel_watts``.  ``None`` proposed means
    the role is not part of this tick's proposal; direction is indeterminate so
    the function returns False and the caller must handle it.
    """
    if proposed_p is None or proposed_d is None:
        return False
    cp = 0 if current_p is None else current_p
    cd = 0 if current_d is None else current_d
    p_up = proposed_p > cp
    p_down = proposed_p < cp
    d_up = proposed_d > cd
    d_down = proposed_d < cd
    return (p_up and d_down) or (p_down and d_up)


def minimum_power_footprint_fits(
    total_budget: int,
    prefill_min_endpoint: int,
    p_watts: Optional[int],
    d_watts: Optional[int],
    decode_min_endpoint: Optional[int] = None,
) -> bool:
    """True when every present role's minimum replicas fit the budget.

    Startup feasibility gate: if even the minimum footprint overshoots the
    total power budget the deployment can never satisfy the ceiling, so the
    planner must fail closed rather than clamp to an impossible target.
    """
    if decode_min_endpoint is None:
        decode_min_endpoint = prefill_min_endpoint

    required = 0
    if p_watts is not None:
        required += prefill_min_endpoint * p_watts
    if d_watts is not None:
        required += decode_min_endpoint * d_watts
    return required <= total_budget


def _hold_at_current(
    proposed: Optional[int], current: Optional[int]
) -> tuple[Optional[int], bool]:
    """Cap a proposal at the current count (block scale-up, allow scale-down).

    ``current is None`` means no replicas are seated yet — treated as 0 so a
    create-from-nothing proposal is a scale-up and can be refused when the
    fixed peer already exhausts the budget.
    """
    if proposed is None:
        return None, False
    baseline = 0 if current is None else current
    held = min(proposed, baseline)
    return held, held < proposed


def apply_power_budget(
    proposed_p: Optional[int],
    proposed_d: Optional[int],
    current_p: Optional[int],
    current_d: Optional[int],
    p_watts: Optional[int],
    d_watts: Optional[int],
    total_budget: int,
    prefill_min_endpoint: int,
    decode_min_endpoint: Optional[int] = None,
) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Clamp proposed replica counts so projected power fits ``total_budget``.

    ``None`` proposals preserve the proposal mask — an unproposed component is
    never mutated; its *current* count is charged against the budget when
    sizing the proposed component(s). Returns ``(new_p, new_d, reason)`` where
    ``reason`` is a short diagnostic when the clamp changed a proposal (or
    suppressed a scale-up), else ``None``.

    Power is ceiling-only and never raises a count above what was proposed.
    """
    p_adjustable = proposed_p is not None and p_watts is not None and p_watts > 0
    d_adjustable = proposed_d is not None and d_watts is not None and d_watts > 0

    if (
        p_adjustable
        and d_adjustable
        and _is_opposing_rebalance(proposed_p, proposed_d, current_p, current_d)
        and peak_parallel_watts(
            current_p, current_d, proposed_p, proposed_d, p_watts, d_watts
        )
        > total_budget
    ):
        # Settled target may fit, but parallel rollouts can transiently exceed
        # the ceiling (e.g. (1,4)->(4,1) peaks at (4,4)). Stage scale-downs
        # first by deferring scale-up legs to a later stable tick.
        new_p, capped_p = _hold_at_current(proposed_p, current_p)
        new_d, capped_d = _hold_at_current(proposed_d, current_d)
        if capped_p or capped_d:
            return new_p, new_d, "power_rebalance_staged"

    if decode_min_endpoint is None:
        decode_min_endpoint = prefill_min_endpoint

    eff_p = proposed_p if proposed_p is not None else current_p
    eff_d = proposed_d if proposed_d is not None else current_d
    if project_watts(eff_p, eff_d, p_watts, d_watts) <= total_budget:
        return proposed_p, proposed_d, None

    if p_adjustable and d_adjustable:
        assert proposed_p is not None and proposed_d is not None
        assert p_watts is not None and d_watts is not None
        current_p_count = 0 if current_p is None else current_p
        current_d_count = 0 if current_d is None else current_d
        if (
            project_watts(
                current_p_count,
                current_d_count,
                p_watts,
                d_watts,
            )
            <= total_budget
        ):
            # An ordinary in-band proposal may use only the directions it
            # requested. Proportional shrinking can otherwise invent a donor
            # (and exceed a throughput delta cap) merely because another role
            # asked to scale up. Explicit opposing swaps were staged above;
            # out-of-band current allocations retain the recovery path below.
            new_p, new_d = fit_directional_budget_pair(
                current_p_count,
                current_d_count,
                proposed_p,
                proposed_d,
                p_watts,
                d_watts,
                -1,
                total_budget,
                prefill_min_endpoint,
                decode_min_endpoint,
            )
            return new_p, new_d, "power_budget_clamped"

        new_p, new_d = _shrink_pair(
            proposed_p,
            proposed_d,
            p_watts,
            d_watts,
            total_budget,
            prefill_min_endpoint,
            decode_min_endpoint,
        )
        # Ceiling never raises a proposed count (decode-no-upscale invariant).
        new_p = min(new_p, proposed_p)
        new_d = min(new_d, proposed_d)
        # The clamp itself can synthesize an opposing rebalance vs current even
        # when the proposal was not one (e.g. current (4,1), proposal (5,5) ->
        # clamped (2,3)): the settled target fits, but parallel actuation peaks
        # at (4,3). Stage that peak the same way as a proposed rebalance by
        # holding the scale-up leg(s) at current so the transient stays under
        # the ceiling; the scale-up is admitted on a later stable tick.
        if (
            _is_opposing_rebalance(new_p, new_d, current_p, current_d)
            and peak_parallel_watts(
                current_p, current_d, new_p, new_d, p_watts, d_watts
            )
            > total_budget
        ):
            staged_p, capped_p = _hold_at_current(new_p, current_p)
            staged_d, capped_d = _hold_at_current(new_d, current_d)
            if capped_p or capped_d:
                return staged_p, staged_d, "power_rebalance_staged"
        return new_p, new_d, "power_budget_clamped"

    if p_adjustable != d_adjustable:
        # Exactly one proposed adjustable component; charge the other at its
        # current count and never mutate it.
        if p_adjustable:
            assert proposed_p is not None and p_watts is not None
            fixed = eff_d * d_watts if (eff_d is not None and d_watts) else 0
            new_p, suppressed = _shrink_single(
                proposed_p,
                current_p,
                p_watts,
                total_budget - fixed,
                prefill_min_endpoint,
            )
            if new_p == proposed_p:
                return new_p, proposed_d, None
            reason = (
                "power_budget_scale_up_suppressed"
                if suppressed
                else "power_budget_clamped"
            )
            return new_p, proposed_d, reason
        assert proposed_d is not None and d_watts is not None
        fixed = eff_p * p_watts if (eff_p is not None and p_watts) else 0
        new_d, suppressed = _shrink_single(
            proposed_d,
            current_d,
            d_watts,
            total_budget - fixed,
            decode_min_endpoint,
        )
        if new_d == proposed_d:
            return proposed_p, new_d, None
        reason = (
            "power_budget_scale_up_suppressed" if suppressed else "power_budget_clamped"
        )
        return proposed_p, new_d, reason

    # Over budget but nothing adjustable is proposed (baseline over budget with
    # no lever this tick). Do not mutate unproposed components.
    return proposed_p, proposed_d, None


def _shrink_pair(
    num_p: int,
    num_d: int,
    p_watts: int,
    d_watts: int,
    budget: int,
    prefill_min_endpoint: int,
    decode_min_endpoint: Optional[int] = None,
) -> tuple[int, int]:
    """Proportionally shrink a disagg pair so watts fit the budget ceiling."""
    if decode_min_endpoint is None:
        decode_min_endpoint = prefill_min_endpoint
    projected = num_p * p_watts + num_d * d_watts
    if projected <= budget:
        return num_p, num_d
    # minimum_power_footprint_fits() at startup guarantees this is unreachable
    # in a correctly configured deployment. Use RuntimeError (not assert) so
    # the guard is never stripped by -O optimized-mode Python.
    minimum_watts = prefill_min_endpoint * p_watts + decode_min_endpoint * d_watts
    if budget < minimum_watts:
        raise RuntimeError(
            f"Power budget infeasible: {budget=} < {minimum_watts=} "
            f"({prefill_min_endpoint=} * {p_watts=} + "
            f"{decode_min_endpoint=} * {d_watts=}); startup validation missed this"
        )
    scale = budget / projected
    max_p = math.floor((budget - decode_min_endpoint * d_watts) / p_watts)
    new_p = max(prefill_min_endpoint, min(max_p, math.floor(num_p * scale)))
    remaining = budget - new_p * p_watts
    # Cap new_d at num_d: when p_watts >> d_watts the proportional shrink lands
    # new_p near the prefill minimum, leaving a large remaining budget that
    # would push floor(remaining / d_watts) above num_d. Callers guarantee num_d >=
    # decode minimum, so min(num_d, max(decode_min_endpoint, ...)) returns <= num_d
    # while still respecting the floor for valid inputs.
    new_d = min(num_d, max(decode_min_endpoint, math.floor(remaining / d_watts)))
    return new_p, new_d


def _shrink_single(
    proposed: int,
    current: Optional[int],
    watts: int,
    avail: int,
    min_endpoint: int,
) -> tuple[int, bool]:
    """Fit a single adjustable pool into ``avail`` watts.

    Returns ``(new_count, suppressed)``. ``suppressed`` is True when the fixed
    (unproposed) component alone leaves no room to even seat ``min_endpoint``,
    so the proposed scale-up is refused (held at ``min(proposed, current)``)
    rather than the unproposed component being silently mutated.
    """
    if avail < min_endpoint * watts:
        held, capped = _hold_at_current(proposed, current)
        # ``proposed`` is non-optional, so ``_hold_at_current`` never returns
        # ``None`` here. ``current is None`` is treated as baseline 0, so a
        # create-from-nothing proposal is suppressed when the fixed peer
        # alone leaves no room for ``min_endpoint``.
        assert held is not None
        return held, capped
    max_fit = math.floor(avail / watts)
    return max(min_endpoint, min(proposed, max_fit)), False


# ---------------------------------------------------------------------------- #
# Deployment bootstrap — GPU counts from DGD / CLI.                            #
# ---------------------------------------------------------------------------- #


def _initialize_gpu_counts(
    config: PlannerConfig,
    connector,
    require_prefill: bool,
    require_decode: bool,
) -> None:
    """Initialize GPU counts from DGD (Kubernetes) or config (virtual).

    In Kubernetes mode: reads from DGD, falls back to CLI flags if not found
    (useful for mockers that don't specify GPU resources).
    In virtual mode: requires CLI flags, errors if not provided.

    Raises:
        DeploymentValidationError: If GPU counts cannot be determined
    """
    # Try to read from DGD in Kubernetes mode
    if hasattr(connector, "get_gpu_counts"):
        try:
            prefill_gpu, decode_gpu = connector.get_gpu_counts(
                require_prefill=require_prefill,
                require_decode=require_decode,
            )
            config.prefill_engine_num_gpu = prefill_gpu
            config.decode_engine_num_gpu = decode_gpu
            logger.info(
                f"Detected GPU counts from DGD: prefill={prefill_gpu}, decode={decode_gpu}"
            )
            return
        except Exception as e:
            # Fall back to CLI flags (e.g., for mockers without GPU resources in DGD)
            logger.warning(
                f"Could not read GPU counts from DGD ({e}), falling back to CLI flags"
            )

    # Use CLI flags (virtual mode, or K8s fallback when DGD lacks GPU resources)
    errors = []
    if require_prefill and config.prefill_engine_num_gpu is None:
        errors.append("Missing prefill_engine_num_gpu in config")
    if require_decode and config.decode_engine_num_gpu is None:
        errors.append("Missing decode_engine_num_gpu in config")
    if errors:
        raise DeploymentValidationError(errors)
    logger.info(
        f"Using GPU counts from CLI: prefill={config.prefill_engine_num_gpu}, "
        f"decode={config.decode_engine_num_gpu}"
    )
