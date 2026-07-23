# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU budget enforcement primitives.

Two layers:

* **Pure math** (``compute_tolerance``, ``bounds_for_total``,
  ``proportional_clamp_pair``, ``proportional_clamp_single``, plus the
  power-budget helpers below): no I/O, no state, no logging. Shared by
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
    min_endpoint: int,
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
    capabilities yet). Returns ``(0, 0)`` when even ``min_endpoint`` of each
    pool would overshoot the hard ceiling (configuration is infeasible).
    """
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
    # min_endpoint of each pool overshoots it, the deployment is infeasible
    # and we zero out (the caller is responsible for surfacing the config
    # error). Otherwise proportionally shrink to fit under ``max_gpus``.
    if max_gpus >= 0 and total > max_gpus:
        min_req = min_endpoint * p_gpu + min_endpoint * d_gpu
        if max_gpus < min_req:
            return 0, 0
        target = max_gpus
        scale = target / total
        max_p = math.floor((target - min_endpoint * d_gpu) / p_gpu)
        new_p = max(min_endpoint, min(max_p, math.floor(num_p * scale)))
        remaining = target - new_p * p_gpu
        new_d = max(min_endpoint, math.floor(remaining / d_gpu))
        return new_p, new_d

    # Floor path — proportional grow toward min_gpus.
    floor = min_gpus
    if total <= 0:
        # No prior allocation — split the floor roughly evenly across the
        # two pools, biasing the remainder toward decode.
        new_p = max(min_endpoint, math.ceil(floor / 2 / p_gpu))
        remaining = max(0, floor - new_p * p_gpu)
        new_d = max(min_endpoint, math.ceil(remaining / d_gpu))
    else:
        scale = floor / total
        new_p = max(min_endpoint, math.ceil(num_p * scale))
        remaining = max(0, floor - new_p * p_gpu)
        new_d = max(min_endpoint, math.ceil(remaining / d_gpu))

    # If the floor push would blow past the strict ceiling, the configuration
    # is infeasible (tight bounds incompatible with the step sizes). Best
    # effort: keep the inputs unchanged and let the caller log; this
    # function stays pure.
    if max_gpus >= 0 and (new_p * p_gpu + new_d * d_gpu) > max_gpus:
        return num_p, num_d

    return new_p, new_d


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
    """True when one role scales up and the other scales down."""
    if (
        proposed_p is None
        or proposed_d is None
        or current_p is None
        or current_d is None
    ):
        return False
    p_up = proposed_p > current_p
    p_down = proposed_p < current_p
    d_up = proposed_d > current_d
    d_down = proposed_d < current_d
    return (p_up and d_down) or (p_down and d_up)


def minimum_power_footprint_fits(
    total_budget: int,
    min_endpoint: int,
    p_watts: Optional[int],
    d_watts: Optional[int],
) -> bool:
    """True when ``min_endpoint`` replicas of every present role fit the budget.

    Startup feasibility gate: if even the minimum footprint overshoots the
    total power budget the deployment can never satisfy the ceiling, so the
    planner must fail closed rather than clamp to an impossible target.
    """
    required = 0
    if p_watts is not None:
        required += min_endpoint * p_watts
    if d_watts is not None:
        required += min_endpoint * d_watts
    return required <= total_budget


def _hold_at_current(
    proposed: Optional[int], current: Optional[int]
) -> tuple[Optional[int], bool]:
    """Cap a proposal at the current count (block scale-up, allow scale-down)."""
    if proposed is None:
        return None, False
    if current is None:
        return proposed, False
    held = min(proposed, current)
    return held, held < proposed


def apply_power_budget(
    proposed_p: Optional[int],
    proposed_d: Optional[int],
    current_p: Optional[int],
    current_d: Optional[int],
    p_watts: Optional[int],
    d_watts: Optional[int],
    total_budget: int,
    min_endpoint: int,
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

    eff_p = proposed_p if proposed_p is not None else current_p
    eff_d = proposed_d if proposed_d is not None else current_d
    if project_watts(eff_p, eff_d, p_watts, d_watts) <= total_budget:
        return proposed_p, proposed_d, None

    if p_adjustable and d_adjustable:
        assert proposed_p is not None and proposed_d is not None
        assert p_watts is not None and d_watts is not None
        new_p, new_d = _shrink_pair(
            proposed_p, proposed_d, p_watts, d_watts, total_budget, min_endpoint
        )
        # Ceiling never raises a proposed count (decode-no-upscale invariant).
        new_p = min(new_p, proposed_p)
        new_d = min(new_d, proposed_d)
        return new_p, new_d, "power_budget_clamped"

    if p_adjustable != d_adjustable:
        # Exactly one proposed adjustable component; charge the other at its
        # current count and never mutate it.
        if p_adjustable:
            assert proposed_p is not None and p_watts is not None
            fixed = eff_d * d_watts if (eff_d is not None and d_watts) else 0
            new_p, suppressed = _shrink_single(
                proposed_p, current_p, p_watts, total_budget - fixed, min_endpoint
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
            proposed_d, current_d, d_watts, total_budget - fixed, min_endpoint
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
    min_endpoint: int,
) -> tuple[int, int]:
    """Proportionally shrink a disagg pair so watts fit the budget ceiling."""
    projected = num_p * p_watts + num_d * d_watts
    if projected <= budget:
        return num_p, num_d
    if budget < min_endpoint * (p_watts + d_watts):
        # Infeasible under the ceiling (startup validation should have caught
        # it). Best effort: hold each pool at the floor.
        return min_endpoint, min_endpoint
    scale = budget / projected
    max_p = math.floor((budget - min_endpoint * d_watts) / p_watts)
    new_p = max(min_endpoint, min(max_p, math.floor(num_p * scale)))
    remaining = budget - new_p * p_watts
    new_d = max(min_endpoint, math.floor(remaining / d_watts))
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
        # ``suppressed`` tracks whether the proposal was actually held back.
        # When ``current`` is None, ``_hold_at_current`` leaves the proposal
        # unchanged and ``capped`` is False — do not claim a suppression.
        return (held if held is not None else min_endpoint), capped
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
