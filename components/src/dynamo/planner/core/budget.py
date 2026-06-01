# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU budget enforcement primitives.

Two layers:

* **Pure math** (``compute_tolerance``, ``bounds_for_total``,
  ``proportional_clamp_pair``, ``proportional_clamp_single``): no I/O, no
  state, no logging. Shared by the local PlannerStateMachine (where the
  budget is enforced intra-DGD by clamping the joint
  ``(num_prefill, num_decode)`` desired counts) and the centralized
  GlobalPlanner (where it is enforced across DGDs by accepting/rejecting
  incoming ScaleRequests). Both layers compute the same ``tolerance`` and
  the same in-band check; only the action taken on a breach differs (the
  local planner transforms counts, the GlobalPlanner decides).

* **Config-aware wrappers** (``_apply_global_gpu_budget``,
  ``_apply_component_gpu_budget``): pull ``min_gpu_budget``,
  ``max_gpu_budget``, ``min_endpoint``, and per-engine GPU counts off a
  ``PlannerConfig`` and delegate to the pure primitives. These are what
  ``state_machine.py`` and friends call.

* ``_initialize_gpu_counts`` remains a deployment-bootstrap helper: it
  populates per-engine GPU counts from the DGD spec or CLI flags, with
  a virtual-mode fallback. Untouched by this refactor.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

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
# Config-aware wrappers — what state_machine.py and friends call.              #
# ---------------------------------------------------------------------------- #


def _apply_global_gpu_budget(
    next_num_p: int, next_num_d: int, config: PlannerConfig
) -> tuple[int, int]:
    """Apply GPU budget band to disagg ``(num_p, num_d)``.

    Honors ``config.max_gpu_budget`` (hard ceiling) and ``config.min_gpu_budget``
    (floor; ``-1`` disables). When both are active, allows the result to
    land in ``[min - tolerance, max]`` where ``tolerance =
    max(prefill_engine_num_gpu, decode_engine_num_gpu)`` — see
    ``proportional_clamp_pair``. ``max_gpu_budget`` is never relaxed.

    Returns ``(0, 0)`` if the ceiling is below the per-pool minima
    (configuration error).
    """
    if config.max_gpu_budget < 0 and config.min_gpu_budget < 0:
        return next_num_p, next_num_d
    assert config.prefill_engine_num_gpu is not None
    assert config.decode_engine_num_gpu is not None

    p_gpu = config.prefill_engine_num_gpu
    d_gpu = config.decode_engine_num_gpu

    new_p, new_d = proportional_clamp_pair(
        next_num_p,
        next_num_d,
        p_gpu,
        d_gpu,
        config.min_gpu_budget,
        config.max_gpu_budget,
        config.min_endpoint,
    )

    if (new_p, new_d) != (next_num_p, next_num_d):
        old_total = next_num_p * p_gpu + next_num_d * d_gpu
        new_total = new_p * p_gpu + new_d * d_gpu
        logger.warning(
            f"GPU budget band [min={config.min_gpu_budget}, max={config.max_gpu_budget}] "
            f"clamped ({next_num_p}P + {next_num_d}D = {old_total} GPUs) -> "
            f"({new_p}P + {new_d}D = {new_total} GPUs)"
        )

    return new_p, new_d


def _apply_component_gpu_budget(
    desired_replicas: int, engine_num_gpu: int, config: PlannerConfig
) -> int:
    """Apply GPU budget band to a single component (agg, or
    prefill-only / decode-only mode)."""
    if config.max_gpu_budget < 0 and config.min_gpu_budget < 0:
        return desired_replicas

    new_replicas = proportional_clamp_single(
        desired_replicas,
        engine_num_gpu,
        config.min_gpu_budget,
        config.max_gpu_budget,
        config.min_endpoint,
    )

    if new_replicas != desired_replicas:
        logger.warning(
            f"GPU budget band [min={config.min_gpu_budget}, max={config.max_gpu_budget}] "
            f"clamped {desired_replicas} replicas (= {desired_replicas * engine_num_gpu} GPUs) "
            f"-> {new_replicas} replicas (= {new_replicas * engine_num_gpu} GPUs)"
        )

    return new_replicas


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
