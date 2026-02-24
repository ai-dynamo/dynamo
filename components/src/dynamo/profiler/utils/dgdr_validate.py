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

"""
Profiler-side validation for DynamoGraphDeploymentRequestSpec.

The auto-generated Pydantic types in ``dgdr_v1beta1_types.py`` mirror the
Go API and mark most fields as ``Optional``.  The profiler requires a
stricter contract.  This module validates those requirements and normalises
fields (e.g. populating defaults, resolving SLA modes) so that downstream
code can access them without ``None`` checks.
"""

from __future__ import annotations

import logging

from dynamo.planner.utils.planner_config import PlannerPreDeploymentSweepMode
from dynamo.profiler.utils.dgdr_v1beta1_types import (
    DynamoGraphDeploymentRequestSpec,
    SLASpec,
    WorkloadSpec,
)
from dynamo.profiler.utils.profile_common import FRONTEND_IMAGE_NAME

logger = logging.getLogger(__name__)


def validate_dgdr_for_profiler(
    dgdr: DynamoGraphDeploymentRequestSpec,
) -> DynamoGraphDeploymentRequestSpec:
    """Validate and normalise a DGDR spec for the profiler.

    After this function returns successfully the caller can safely access:

    - ``dgdr.image`` (str, non-empty)
    - ``dgdr.hardware.gpuSku`` (str, non-empty)
    - ``dgdr.hardware.numGpusPerNode`` (int > 0)
    - ``dgdr.workload.isl``, ``dgdr.workload.osl`` (int)
    - ``dgdr.sla.ttft``, ``dgdr.sla.itl`` (float) **or** ``dgdr.sla.e2eLatency`` (float)

    without additional ``None`` guards.

    The function mutates ``dgdr`` in-place (e.g. populating defaults) and
    returns it for convenience.

    Raises:
        ValueError: If a required field is missing or invalid.
    """
    _validate_required_fields(dgdr)
    _validate_workload(dgdr.workload)
    _validate_sla(dgdr.sla)
    _validate_features(dgdr)
    return dgdr


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------


def _validate_required_fields(dgdr: DynamoGraphDeploymentRequestSpec) -> None:
    """Check fields the profiler treats as required."""
    if not dgdr.image:
        raise ValueError("'image' is required in the DGDR spec.")

    if FRONTEND_IMAGE_NAME not in dgdr.image:
        raise ValueError(
            f"'image' must reference the published Dynamo frontend image. "
            f"Expected the image name to contain '{FRONTEND_IMAGE_NAME}' "
            f"(e.g. nvcr.io/nvidia/ai-dynamo/{FRONTEND_IMAGE_NAME}:<tag>), "
            f"got: '{dgdr.image}'."
        )

    if not dgdr.hardware:
        raise ValueError("'hardware' is required in the DGDR spec.")
    if not dgdr.hardware.gpuSku:
        raise ValueError("'hardware.gpuSku' is required in the DGDR spec.")
    if not dgdr.hardware.numGpusPerNode or dgdr.hardware.numGpusPerNode <= 0:
        raise ValueError("'hardware.numGpusPerNode' must be a positive integer.")

    # Populate defaults for optional sub-objects so callers don't need None checks
    if dgdr.workload is None:
        dgdr.workload = WorkloadSpec()
    if dgdr.sla is None:
        dgdr.sla = SLASpec()


def _validate_workload(workload: WorkloadSpec) -> None:
    """Concurrency and requestRate are mutually exclusive."""
    if workload.concurrency is not None and workload.requestRate is not None:
        raise ValueError(
            "Only one of 'concurrency' or 'requestRate' can be provided, not both."
        )


def _validate_sla(sla: SLASpec) -> None:
    """Validate SLA targets and normalise e2eLatency mode."""
    for name, val in [
        ("ttft", sla.ttft),
        ("itl", sla.itl),
        ("e2eLatency", sla.e2eLatency),
    ]:
        if val is not None and val <= 0:
            raise ValueError(f"SLA '{name}' must be positive (got {val}).")

    has_e2e = sla.e2eLatency is not None

    # When e2eLatency is provided it takes precedence — null out the per-token defaults
    if has_e2e:
        sla.ttft = None
        sla.itl = None
        return

    has_ttft_itl = sla.ttft is not None and sla.itl is not None
    if not has_ttft_itl:
        raise ValueError(
            "Either both 'ttft' and 'itl', or 'e2eLatency', must be provided in the SLA spec."
        )


def _validate_features(dgdr: DynamoGraphDeploymentRequestSpec) -> None:
    """Cross-field validation for features."""
    if not dgdr.features:
        return

    # Mocker requires pre-deployment sweeping
    if dgdr.features.mocker and dgdr.features.mocker.enabled and dgdr.features.planner:
        sweep_mode = dgdr.features.planner.pre_deployment_sweeping_mode
        if sweep_mode is None or sweep_mode == PlannerPreDeploymentSweepMode.None_:
            raise ValueError(
                "pre_deployment_sweeping_mode cannot be 'none' when mocker is enabled. "
                "Mocker backend requires pre-deployment sweeping to generate simulated "
                "performance profiles."
            )
