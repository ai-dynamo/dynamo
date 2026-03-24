# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for determining parallelization from ``optimizationType``."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
    ParallelizationMapping,
)
from dynamo.profiler.utils.dgdr_v1beta1_types import (
    DynamoGraphDeploymentRequestSpec,
    OptimizationType,
)
from dynamo.profiler.utils.model_info import (
    ModelInfo,
    calculate_min_gpus_for_model,
    get_default_parallelization_for_architecture,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationTypeConfig:
    """Resolved parallelization and concurrency strategy for optimization-type mode.

    Attributes
    ----------
    prefill_mapping:
        Parallelization mapping for the prefill engine.
    decode_mapping:
        Parallelization mapping for the decode engine.
    num_gpus:
        Number of GPUs per engine (always a power of two).
    use_max_concurrency:
        ``True`` for throughput optimisation (highest concurrency / full KV
        cache); ``False`` for latency optimisation (single-stream).
    """

    prefill_mapping: ParallelizationMapping
    decode_mapping: ParallelizationMapping
    num_gpus: int
    use_max_concurrency: bool


def _mapping_dict_to_parallelization(mapping: dict) -> ParallelizationMapping:
    """Convert a ``{"tp": N}`` / ``{"tep": N}`` / ``{"dep": N}`` dict to a
    :class:`ParallelizationMapping` instance."""
    return ParallelizationMapping(**mapping)


def resolve_optimization_type_config(
    dgdr: DynamoGraphDeploymentRequestSpec,
    model_info: ModelInfo,
) -> OptimizationTypeConfig:
    """Determine parallelization and concurrency for ``optimizationType`` mode.

    Parameters
    ----------
    dgdr:
        The validated DGDR spec.  Must have ``dgdr.sla.optimizationType`` set.
    model_info:
        Pre-fetched model architecture information.

    Returns
    -------
    OptimizationTypeConfig
        The resolved config with parallelization mappings, GPU count, and
        concurrency strategy.

    Raises
    ------
    ValueError
        If ``dgdr.sla.optimizationType`` is not set.
    """
    if dgdr.sla is None or dgdr.sla.optimizationType is None:
        raise ValueError(
            "resolve_optimization_type_config requires sla.optimizationType to be set"
        )

    opt_type: OptimizationType = dgdr.sla.optimizationType

    # --- Determine number of GPUs -------------------------------------------
    vram_mb = dgdr.hardware.vramMb
    if vram_mb is not None and vram_mb > 0:
        num_gpus = calculate_min_gpus_for_model(model_info.model_size, vram_mb)
    else:
        # VRAM not available — fall back to total GPUs (single-node assumption)
        logger.warning(
            "vramMb not set in hardware spec; falling back to totalGpus=%s",
            dgdr.hardware.totalGpus,
        )
        num_gpus = dgdr.hardware.totalGpus or 1

    # Clamp to available GPUs
    total_gpus = dgdr.hardware.totalGpus or num_gpus
    if num_gpus > total_gpus:
        logger.warning(
            "Computed num_gpus=%d exceeds totalGpus=%d; clamping.",
            num_gpus,
            total_gpus,
        )
        num_gpus = total_gpus

    # --- Determine parallelization strategy ---------------------------------
    prefill_dict, decode_dict = get_default_parallelization_for_architecture(
        architecture=model_info.architecture,
        is_moe=model_info.is_moe,
        num_gpus=num_gpus,
        optimization_type=opt_type.value,  # "throughput" or "latency"
    )

    prefill_mapping = _mapping_dict_to_parallelization(prefill_dict)
    decode_mapping = _mapping_dict_to_parallelization(decode_dict)

    # --- Concurrency strategy -----------------------------------------------
    use_max_concurrency = opt_type == OptimizationType.Throughput

    logger.info(
        "Resolved optimizationType=%s: num_gpus=%d, prefill=%s, decode=%s, "
        "use_max_concurrency=%s",
        opt_type.value,
        num_gpus,
        prefill_mapping.label(),
        decode_mapping.label(),
        use_max_concurrency,
    )

    return OptimizationTypeConfig(
        prefill_mapping=prefill_mapping,
        decode_mapping=decode_mapping,
        num_gpus=num_gpus,
        use_max_concurrency=use_max_concurrency,
    )
