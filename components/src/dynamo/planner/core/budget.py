# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.errors import DeploymentValidationError
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _apply_global_gpu_budget(
    next_num_p: int, next_num_d: int, config: PlannerConfig
) -> tuple[int, int]:
    """Apply GPU budget constraint to both prefill and decode replicas.

    When total GPUs required (num_p * prefill_gpus + num_d * decode_gpus) exceeds the
    budget, scale down both proportionally using scale = budget / total_required. Prefill
    replicas are clamped to [min_endpoint, max_prefill] where max_prefill reserves enough
    GPUs for min_endpoint decode replicas. Remaining budget is then allocated to decode.
    Returns (0, 0) if budget cannot satisfy min_endpoint for both components.
    """
    if config.max_gpu_budget < 0:
        return next_num_p, next_num_d
    assert config.prefill_engine_num_gpu is not None
    assert config.decode_engine_num_gpu is not None
    total_gpu_required = (
        next_num_p * config.prefill_engine_num_gpu
        + next_num_d * config.decode_engine_num_gpu
    )
    if total_gpu_required <= config.max_gpu_budget:
        return next_num_p, next_num_d
    min_required = (
        config.min_endpoint * config.prefill_engine_num_gpu
        + config.min_endpoint * config.decode_engine_num_gpu
    )
    if config.max_gpu_budget < min_required:
        logger.warning(
            f"max_gpu_budget ({config.max_gpu_budget}) is below the minimum required "
            f"for min_endpoint ({min_required}); enforcing zero replicas"
        )
        return 0, 0
    scale = config.max_gpu_budget / total_gpu_required
    max_prefill = math.floor(
        (config.max_gpu_budget - config.min_endpoint * config.decode_engine_num_gpu)
        / config.prefill_engine_num_gpu
    )
    next_num_p = max(
        config.min_endpoint, min(max_prefill, math.floor(next_num_p * scale))
    )
    remaining = config.max_gpu_budget - next_num_p * config.prefill_engine_num_gpu
    next_num_d = max(
        config.min_endpoint, math.floor(remaining / config.decode_engine_num_gpu)
    )
    logger.warning(
        f"Total number of GPUs required ({total_gpu_required}) exceeds the max GPU budget ({config.max_gpu_budget}), "
        f"scaling down to {next_num_p} prefill and {next_num_d} decode replicas"
    )
    return next_num_p, next_num_d


def _apply_component_gpu_budget(
    desired_replicas: int, engine_num_gpu: int, config: PlannerConfig
) -> int:
    """Apply GPU budget constraint to a single component (prefill-only or decode-only).

    When total GPUs required (replicas * gpus_per_replica) exceeds the budget, scale down
    using scale = budget / total_required, floored and clamped to at least min_endpoint.
    Returns 0 if budget cannot satisfy min_endpoint replicas.
    """
    if config.max_gpu_budget < 0:
        return desired_replicas
    total_gpu_required = desired_replicas * engine_num_gpu
    if total_gpu_required <= config.max_gpu_budget:
        return desired_replicas
    min_required = config.min_endpoint * engine_num_gpu
    if config.max_gpu_budget < min_required:
        logger.warning(
            f"max_gpu_budget ({config.max_gpu_budget}) is below the minimum required "
            f"for min_endpoint ({min_required}); enforcing zero replicas"
        )
        return 0
    scale = config.max_gpu_budget / total_gpu_required
    next_num = max(config.min_endpoint, math.floor(desired_replicas * scale))
    logger.warning(
        f"Total number of GPUs required ({total_gpu_required}) exceeds the max GPU budget ({config.max_gpu_budget}), "
        f"scaling down to {next_num} replicas"
    )
    return next_num


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
