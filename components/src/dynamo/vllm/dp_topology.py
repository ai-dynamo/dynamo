# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM data-parallel topology helpers."""

import logging

from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


def get_dp_range_for_worker(vllm_config: VllmConfig) -> tuple[int, int]:
    """Return the global DP start rank and number of ranks managed by this worker."""
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_external_lb:
        return (parallel_config.data_parallel_rank, 1)
    if parallel_config.data_parallel_hybrid_lb:
        return (
            parallel_config.data_parallel_rank,
            parallel_config.data_parallel_size_local,
        )

    logger.warning(
        "vLLM selects internal DP load balancing. If you are launching multiple "
        "workers for DP deployment, hybrid or external load balancing is recommended."
    )
    return (
        parallel_config.data_parallel_rank,
        parallel_config.data_parallel_size,
    )
