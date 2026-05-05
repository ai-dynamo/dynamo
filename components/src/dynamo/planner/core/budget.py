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
    """Pure check: does ``total`` fit ``[min_gpus - tolerance, max_gpus + tolerance]``?

    A negative ``min_gpus`` disables the floor. A negative ``max_gpus``
    disables the ceiling. ``tolerance == 0`` enforces strict bounds.

    Returns ``(in_bounds, reason_if_out)``. ``reason`` is empty when in bounds.
    """
    if max_gpus >= 0:
        hi = max_gpus + tolerance
        if total > hi:
            return (
                False,
                f"total {total} exceeds ceiling "
                f"({max_gpus}{f' + tol {tolerance}' if tolerance else ''})",
            )
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

    The band is ``[min_gpus - tolerance, max_gpus + tolerance]`` when both
    bounds are active, and strictly ``[0, max_gpus]`` or ``[min_gpus, +inf)``
    when only one bound is active. ``tolerance`` is computed internally as
    ``max(p_gpu, d_gpu)`` and only applied when both bounds are active —
    callers with only a ceiling get the historical strict behavior of
    ``_apply_global_budget`` preserved exactly.

    Distribution policy is proportional in both directions (mirror of the
    historical proportional shrink). SLA-pressure-aware split is a future
    enhancement.

    Negative ``min_gpus`` or ``max_gpus`` disables the corresponding bound.
    Returns ``(num_p, num_d)`` unchanged if both are disabled or if either
    per-replica GPU count is non-positive (caller hasn't initialized
    capabilities yet).
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

    # Ceiling path — proportional shrink. Mirrors the historical
    # _apply_global_budget logic for backward compatibility.
    if max_gpus >= 0 and total > max_gpus + tolerance:
        min_req = min_endpoint * p_gpu + min_endpoint * d_gpu
        # Compare against the *band* upper edge (max + tolerance), not the
        # strict ceiling: in fixed-budget configs the band is the contract,
        # and (1, 1) at min_endpoint each can land above max but inside
        # max + tolerance. Zeroing the deployment in that case is wrong.
        # When tolerance is 0 (ceiling-only mode) this collapses to the
        # historical strict check.
        if max_gpus + tolerance < min_req:
            return 0, 0
        # Shrink target is the band upper edge so min_endpoint*per_pool can
        # actually fit when max_gpus alone is a hair too small.
        target = max(max_gpus, min_req)
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

    # If the floor push would blow past the ceiling band, the configuration
    # is infeasible (tight bounds incompatible with the step sizes). Best
    # effort: keep the inputs unchanged and let the caller log; this
    # function stays pure.
    if max_gpus >= 0 and (new_p * p_gpu + new_d * d_gpu) > max_gpus + tolerance:
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
    active. When only one bound is set, falls back to strict enforcement
    matching historical ``_budget_clamp``.

    Negative ``min_gpus`` or ``max_gpus`` disables the corresponding bound.
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

    if max_gpus >= 0 and total > max_gpus + tolerance:
        min_req = min_endpoint * engine_gpu
        # Compare against band upper edge (max + tolerance), not strict
        # ceiling — in fixed-budget configs the band is the contract.
        # E.g. engine_gpu=4, min_endpoint=1, min=max=3 should let
        # 1 replica (=4 GPUs) survive inside the [-1, 7] band rather than
        # being torn down. tolerance=0 in ceiling-only mode preserves
        # historical strict behavior.
        if max_gpus + tolerance < min_req:
            return 0
        target = max(max_gpus, min_req)
        return max(min_endpoint, math.floor(target / engine_gpu))

    # total < min_gpus - tolerance
    return max(min_endpoint, math.ceil(min_gpus / engine_gpu))


# ---------------------------------------------------------------------------- #
# Config-aware wrappers — what state_machine.py and friends call.              #
# ---------------------------------------------------------------------------- #


def _apply_global_gpu_budget(
    next_num_p: int, next_num_d: int, config: PlannerConfig
) -> tuple[int, int]:
    """Apply GPU budget band to disagg ``(num_p, num_d)``.

    Honors ``config.max_gpu_budget`` (ceiling) and ``config.min_gpu_budget``
    (floor; ``-1`` disables). When both are active, allows the result to
    land in ``[min - tolerance, max + tolerance]`` where ``tolerance =
    max(prefill_engine_num_gpu, decode_engine_num_gpu)`` — see
    ``proportional_clamp_pair``.

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
