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

"""Shared helpers and configuration for the profiler pipeline."""

import logging
import os
from dataclasses import dataclass

import pandas as pd

from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
    PickedParallelConfig,
)
from dynamo.profiler.utils.dgdr_v1beta1_types import DynamoGraphDeploymentRequestSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Published container image naming conventions
# ---------------------------------------------------------------------------

# The image-name component of the published Dynamo frontend image.
# e.g. nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.0
FRONTEND_IMAGE_NAME = "dynamo-frontend"

# Mapping from backend name to the image-name component of the published
# backend runtime image.
# e.g. vllm → nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0
BACKEND_IMAGE_NAMES: dict[str, str] = {
    "vllm": "vllm-runtime",
    "sglang": "sglang-runtime",
    "trtllm": "tensorrtllm-runtime",
}


def derive_backend_image(frontend_image: str, backend: str) -> str:
    """Derive the backend worker image from the frontend image.

    The frontend image must contain ``dynamo-frontend`` in its name.  The
    backend image is derived by replacing that component with the appropriate
    backend runtime image name, preserving the registry path and tag.

    Example::

        derive_backend_image(
            "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.0", "vllm"
        )
        # → "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0"

    Args:
        frontend_image: The frontend container image.  Must contain
            ``'dynamo-frontend'`` (e.g.
            ``nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.0``).
        backend: The resolved backend type (``'vllm'``, ``'sglang'``, or
            ``'trtllm'``).

    Returns:
        The backend container image string.

    Raises:
        ValueError: If *backend* is not a recognised backend, or if
            *frontend_image* does not follow the expected naming convention.
    """
    backend_image_name = BACKEND_IMAGE_NAMES.get(backend)
    if backend_image_name is None:
        raise ValueError(
            f"Cannot derive backend image for unknown backend '{backend}'. "
            f"Supported backends: {list(BACKEND_IMAGE_NAMES.keys())}"
        )
    if FRONTEND_IMAGE_NAME not in frontend_image:
        raise ValueError(
            f"Image '{frontend_image}' does not contain '{FRONTEND_IMAGE_NAME}'. "
            f"The profiler expects the 'image' field to reference the published "
            f"Dynamo frontend image "
            f"(e.g. nvcr.io/nvidia/ai-dynamo/{FRONTEND_IMAGE_NAME}:<tag>)."
        )
    return frontend_image.replace(FRONTEND_IMAGE_NAME, backend_image_name)


# ---------------------------------------------------------------------------
# Operational defaults not part of DynamoGraphDeploymentRequestSpec
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "profiling_results"
DEFAULT_NAMESPACE = os.environ.get("DGDR_NAMESPACE", "dynamo-sla-profiler")
DEFAULT_DEPLOYMENT_TIMEOUT = 3600
DEFAULT_PREFILL_INTERPOLATION_GRANULARITY = 16
DEFAULT_DECODE_INTERPOLATION_GRANULARITY = 6
DEFAULT_DRY_RUN = False


@dataclass
class ProfilerOperationalConfig:
    """Operational knobs that are not part of the DGDR spec."""

    output_dir: str = DEFAULT_OUTPUT_DIR
    k8s_namespace: str = DEFAULT_NAMESPACE
    deployment_timeout: int = DEFAULT_DEPLOYMENT_TIMEOUT
    prefill_interpolation_granularity: int = DEFAULT_PREFILL_INTERPOLATION_GRANULARITY
    decode_interpolation_granularity: int = DEFAULT_DECODE_INTERPOLATION_GRANULARITY
    dry_run: bool = DEFAULT_DRY_RUN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def picked_config_from_row(prefix: str, row: pd.Series) -> PickedParallelConfig:
    """Extract a PickedParallelConfig from a picked ColumnsDisagg DataFrame row."""
    return PickedParallelConfig(
        tp=int(row.get(f"{prefix}tp", 1)),
        pp=int(row.get(f"{prefix}pp", 1)),
        dp=int(row.get(f"{prefix}dp", 1)),
        moe_tp=int(row.get(f"{prefix}moe_tp", 1)),
        moe_ep=int(row.get(f"{prefix}moe_ep", 1)),
    )


def resolve_model_path(dgdr: DynamoGraphDeploymentRequestSpec) -> str:
    """Resolve the model path, preferring local PVC mount over HF ID."""
    if (
        dgdr.modelCache
        and dgdr.modelCache.pvcName
        and dgdr.modelCache.pvcMountPath
        and dgdr.modelCache.pvcModelPath
    ):
        mount = dgdr.modelCache.pvcMountPath.rstrip("/")
        sub = dgdr.modelCache.pvcModelPath.strip("/")
        local_path = f"{mount}/{sub}"
        if os.path.isdir(local_path):
            return local_path
    return dgdr.model


def is_planner_enabled(dgdr: DynamoGraphDeploymentRequestSpec) -> bool:
    """True when the DGDR spec has a planner config with scaling enabled."""
    return (
        dgdr.features is not None
        and dgdr.features.planner is not None
        and dgdr.features.planner.scaling_enabled()
    )


def determine_picking_mode(dgdr: DynamoGraphDeploymentRequestSpec) -> str:
    target_load_provided = dgdr.workload is not None and (
        dgdr.workload.requestRate is not None or dgdr.workload.concurrency is not None
    )
    if is_planner_enabled(dgdr):
        return "autoscale"
    elif target_load_provided:
        return "load_match"
    return "default"


def warn_and_update_sla(
    best_latencies: dict,
    target_ttft: float,
    target_tpot: float,
) -> tuple[float, float]:
    """Warn if SLA is unachievable; return (possibly updated) targets."""
    achieved_ttft = best_latencies.get("ttft", 0.0)
    achieved_tpot = best_latencies.get("tpot", 0.0)

    if achieved_ttft > target_ttft:
        logger.warning(
            "TTFT SLA %.1fms is unachievable. Best achievable: %.1fms. Updating SLA.",
            target_ttft,
            achieved_ttft,
        )
        target_ttft = achieved_ttft

    if achieved_tpot > target_tpot:
        logger.warning(
            "ITL SLA %.1fms is unachievable. Best achievable: %.1fms. Updating SLA.",
            target_tpot,
            achieved_tpot,
        )
        target_tpot = achieved_tpot

    return target_ttft, target_tpot


def warn_gpu_shortage(
    picking_mode: str,
    best_latencies: dict,
    total_gpus: int,
) -> None:
    if picking_mode != "load_match":
        return
    gpus_needed = best_latencies.get("total_gpus_needed")
    if gpus_needed is not None and gpus_needed > total_gpus:
        logger.warning(
            "Load target requires %d GPUs but only %d available. "
            "Consider adding more GPUs or reducing the load target.",
            gpus_needed,
            total_gpus,
        )
