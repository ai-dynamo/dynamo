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

import copy
import logging
import os
from dataclasses import dataclass, field
from stat import S_ISREG

import pandas as pd

from dynamo.common.container_images import BACKEND_IMAGE_NAMES as BACKEND_IMAGE_NAMES
from dynamo.common.container_images import PLANNER_IMAGE_NAME as PLANNER_IMAGE_NAME
from dynamo.common.container_images import derive_backend_image as derive_backend_image
from dynamo.common.container_images import derive_planner_image as derive_planner_image
from dynamo.planner.config.planner_config import PlannerPreDeploymentSweepMode
from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
    PickedParallelConfig,
)
from dynamo.profiler.utils.dgdr_v1beta1_types import (
    DynamoGraphDeploymentRequestSpec,
    ProfilingPhase,
    SearchStrategy,
)
from dynamo.profiler.utils.model_cache_paths import normalize_model_cache_path

logger = logging.getLogger(__name__)

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
    current_phase: ProfilingPhase = field(default=ProfilingPhase.Initializing)


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
        local_path = normalize_model_cache_path(
            dgdr.modelCache.pvcMountPath,
            dgdr.modelCache.pvcModelPath,
        )
        config_path = os.path.join(local_path, "config.json")
        try:
            if S_ISREG(os.stat(config_path).st_mode):
                return local_path
        except (FileNotFoundError, NotADirectoryError):
            pass
        except OSError as e:
            raise RuntimeError(
                f"Cannot inspect PVC model config {config_path!r}: {e}. "
                "Check directory permissions and symlink ownership for the "
                "profiler user, or set modelCache.pvcModelPath to an accessible "
                "snapshot directory."
            ) from e
    return dgdr.model


def pick_decode_component(client) -> str:
    """Pick the decode worker component name from a deployment client.

    Returns the first entry in ``client.components`` that is not the frontend
    (case-insensitive), falling back to the literal ``"decode"`` if every
    component is frontend or the list is empty. The previous fallback
    (``client.components[-1]``) could resolve to ``"frontend"`` in degenerate
    component lists, which routed log-path lookups at the frontend service.
    """
    for svc in getattr(client, "components", None) or []:
        if str(svc).lower() != "frontend":
            return svc
    return "decode"


def is_planner_enabled(dgdr: DynamoGraphDeploymentRequestSpec) -> bool:
    """True when the DGDR spec has a planner config with scaling enabled."""
    return (
        dgdr.features is not None
        and dgdr.features.planner is not None
        and dgdr.features.planner.scaling_enabled()
    )


def is_mocker_enabled(dgdr: DynamoGraphDeploymentRequestSpec) -> bool:
    """True when the DGDR spec has mocker explicitly enabled."""
    return (
        dgdr.features is not None
        and dgdr.features.mocker is not None
        and dgdr.features.mocker.enabled is True
    )


def is_kv_router_enabled(dgdr: DynamoGraphDeploymentRequestSpec) -> bool:
    """True when the DGDR spec explicitly enables KV-cache-aware routing."""
    return (
        dgdr.features is not None
        and dgdr.features.kvRouter is not None
        and dgdr.features.kvRouter.enabled is True
    )


def needs_mocker_ais_perf_model(dgdr: DynamoGraphDeploymentRequestSpec) -> bool:
    """True when mocker workers should load performance data from AIC.

    Requests with Planner configuration use its pre-deployment sweep mode. For
    mocker-only requests there is no Planner mode, so the DGDR search strategy
    determines whether the profiler uses rapid AIC data or thorough NPZ data.
    """
    if not is_mocker_enabled(dgdr):
        return False
    if dgdr.features is not None and dgdr.features.planner is not None:
        return (
            dgdr.features.planner.pre_deployment_sweeping_mode
            == PlannerPreDeploymentSweepMode.Rapid
        )
    return dgdr.searchStrategy == SearchStrategy.Rapid


def needs_profile_data(dgdr: DynamoGraphDeploymentRequestSpec) -> bool:
    """True when the DGDR requires profiling interpolation data *at this stage*.

    Profile data (NPZ/JSON on disk) is consumed by:

    * **Mocker workers** for latency simulation — required for thorough mode.
      Requests with Planner configuration use its sweep mode; mocker-only
      requests use the DGDR search strategy. In rapid mode the mocker pulls
      latency data directly from the AIConfigurator SDK via
      ``--ais-perf-model`` flags injected by the profiler, so no NPZ is
      emitted.
    * **Planner** when thorough-mode bootstrap data is requested. In rapid
      mode the planner receives ``ais_perf_model`` and can also run AIC
      interpolation in-process at bootstrap; in none mode it starts from
      native AIC or live FPM regression warmup.
    """
    if is_mocker_enabled(dgdr):
        return not needs_mocker_ais_perf_model(dgdr)
    sweep_mode = (
        dgdr.features.planner.pre_deployment_sweeping_mode
        if dgdr.features is not None and dgdr.features.planner is not None
        else None
    )
    if (
        dgdr.features is not None
        and dgdr.features.planner is not None
        and dgdr.features.planner.enable_throughput_scaling
    ):
        return sweep_mode == PlannerPreDeploymentSweepMode.Thorough
    return False


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


def get_profiling_job_tolerations(dgdr: DynamoGraphDeploymentRequestSpec) -> list:
    """Return tolerations from overrides.profilingJob.template.spec.tolerations."""
    try:
        if dgdr.overrides is None or dgdr.overrides.profilingJob is None:
            return []
        return (
            dgdr.overrides.profilingJob.get("template", {})
            .get("spec", {})
            .get("tolerations", [])
        )
    except (AttributeError, KeyError):
        return []


def inject_tolerations_into_dgd(dgd_config: dict, tolerations: list) -> dict:
    """Add tolerations to every component pod template in a DGD config.

    Tolerations already present in a component are preserved; only new entries
    (by identity) are appended.  Returns a deep copy with tolerations applied.
    """
    result = copy.deepcopy(dgd_config)
    components = result.get("spec", {}).get("components", [])
    if not isinstance(components, list):
        return result
    for component in components:
        if not isinstance(component, dict):
            continue
        pod_spec = component.setdefault("podTemplate", {}).setdefault("spec", {})
        existing = pod_spec.get("tolerations") or []
        new_entries = [t for t in tolerations if t not in existing]
        if new_entries:
            pod_spec["tolerations"] = list(existing) + new_entries
    return result
