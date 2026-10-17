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

import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.errors import (
    DuplicateSubComponentError,
    GPUShapeUnavailableError,
    PowerAnnotationInvalidError,
    PowerAnnotationMissingError,
    SubComponentNotFoundError,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

V1BETA1_COMPONENT_TYPES = {"prefill", "decode"}
V1BETA1_GENERIC_WORKER_COMPONENT_TYPE = "worker"

# Per-GPU power-limit annotation key (watts, positive integer).
#
# Ownership: this value is *authored* on the DGD worker component
# ``podTemplate.metadata.annotations`` by a human or the profiler. The operator
# renders it onto every worker Pod at create time; the Power Agent DaemonSet
# reads the *live Pod* annotation and applies the NVML/DCGM cap. The operator
# also projects the effective value into component status; the Planner reads
# only that status to project a power budget and never writes it onto Pods. The Power Agent
# keeps its own copy of this literal (deploy/power-agent/power_agent.py); the
# two are asserted identical by a contract test rather than shared as a package
# import, because the agent image does not install the ``dynamo`` package.
POWER_ANNOTATION_KEY = "dynamo.nvidia.com/gpu-power-limit"


@dataclass(frozen=True)
class ComponentGPUShape:
    """Inference-engine GPU width and unique allocation per replica."""

    gpus_per_engine: int
    gpus_per_replica: int


def get_components_by_name(deployment: dict) -> dict[str, dict]:
    """Return v1beta1 DGD components keyed by logical name.

    v1beta1 exposes components as ``spec.components[]`` with ``name`` and
    ``type``. The planner consumes this map so the rest of the code does not
    have to work with list traversal.
    """
    components = deployment.get("spec", {}).get("components") or []
    return {component["name"]: component for component in components}


def get_component_type(component: dict) -> str:
    return component.get("type", "")


def get_planner_component_role(component: dict) -> str:
    component_type = get_component_type(component)
    if component_type in V1BETA1_COMPONENT_TYPES:
        return component_type
    return ""


def _can_use_explicit_component_name(
    component: dict, component_type: SubComponentType
) -> bool:
    explicit_type = get_component_type(component)
    return explicit_type in (
        "",
        V1BETA1_GENERIC_WORKER_COMPONENT_TYPE,
        component_type.value,
    )


class Service(BaseModel):
    name: str
    service: dict

    def number_replicas(self) -> int:
        return self.service.get("replicas", 0)

    def _current_component_status(self, deployment: dict) -> dict:
        deployment_status = deployment.get("status", {})
        generation = deployment.get("metadata", {}).get("generation")
        if generation is None or generation != deployment_status.get(
            "observedGeneration"
        ):
            return {}
        return deployment_status.get("components", {}).get(self.name, {})

    def get_model_name(self, deployment: dict) -> Optional[str]:
        """Return the operator-projected primary served model name."""
        return self._current_component_status(deployment).get("servedModelName")

    def get_runtime_component_name(self, deployment: dict) -> Optional[str]:
        """Return the operator-projected Dynamo runtime component identity."""
        return self._current_component_status(deployment).get("runtimeComponentName")

    def get_gpu_shape(self, deployment: dict) -> ComponentGPUShape:
        """Return the current operator-projected GPU shape."""
        deployment_status = deployment.get("status", {})
        component_status = deployment_status.get("components", {}).get(self.name, {})
        engine_raw = component_status.get("gpusPerEngine")
        replica_raw = component_status.get("gpusPerReplica")
        generation = deployment.get("metadata", {}).get("generation")
        observed_generation = deployment_status.get("observedGeneration")
        if generation is None or observed_generation != generation:
            raise GPUShapeUnavailableError(
                self.name,
                f"Resolved GPU shape for component '{self.name}' is not current: "
                f"metadata.generation={generation}, "
                f"status.observedGeneration={observed_generation}.",
            )
        if engine_raw is None or replica_raw is None:
            deployment_state = deployment_status.get("state", "unknown")
            raise GPUShapeUnavailableError(
                self.name,
                "operator status has no complete gpusPerEngine/gpusPerReplica "
                f"shape for component '{self.name}' (deployment state {deployment_state!r})",
            )
        try:
            engine = int(engine_raw)
            replica = int(replica_raw)
        except (TypeError, ValueError) as err:
            raise GPUShapeUnavailableError(
                self.name,
                f"Invalid GPU shape for component '{self.name}': "
                f"gpusPerEngine={engine_raw!r}, gpusPerReplica={replica_raw!r}.",
            ) from err
        if engine < 0 or replica < 0 or (replica == 0 and engine != 0):
            raise GPUShapeUnavailableError(
                self.name,
                f"Invalid GPU shape for component '{self.name}': "
                f"gpusPerEngine={engine}, gpusPerReplica={replica}.",
            )
        return ComponentGPUShape(engine, replica)

    def get_gpu_power_limit_watts(self, deployment: dict) -> int:
        """Return the operator-projected per-GPU power limit."""
        raw = self._current_component_status(deployment).get("gpuPowerLimitWatts")
        if raw is None:
            raise PowerAnnotationMissingError(self.name)
        try:
            watts = int(raw)
        except (ValueError, TypeError) as err:
            raise PowerAnnotationInvalidError(self.name, str(raw)) from err
        if watts <= 0:
            raise PowerAnnotationInvalidError(self.name, str(raw))
        return watts


def get_component_from_type_or_name(
    deployment: dict,
    component_type: SubComponentType,
    component_name: Optional[str] = None,
) -> Service:
    """
    Get the current replicas for a component in a graph deployment

    Returns: Service object

    Raises:
        SubComponentNotFoundError: If no component with the specified role is found
        DuplicateSubComponentError: If multiple components have the same role
    """
    components = get_components_by_name(deployment)

    matching_components = []

    for curr_name, curr_component in components.items():
        component_role = get_planner_component_role(curr_component)
        if component_role == component_type.value:
            matching_components.append((curr_name, curr_component))

    # Check for duplicates
    if len(matching_components) > 1:
        component_names = [name for name, _ in matching_components]
        raise DuplicateSubComponentError(component_type.value, component_names)

    if not matching_components and component_type == SubComponentType.DECODE:
        generic_workers = [
            (name, component)
            for name, component in components.items()
            if get_component_type(component) == V1BETA1_GENERIC_WORKER_COMPONENT_TYPE
        ]
        if len(generic_workers) == 1:
            matching_components = generic_workers

    if not matching_components and component_name in components:
        component = components[component_name]
        if not _can_use_explicit_component_name(component, component_type):
            raise SubComponentNotFoundError(component_type.value)
        matching_components.append((component_name, component))
    elif not matching_components:
        raise SubComponentNotFoundError(component_type.value)

    name, component = matching_components[0]
    return Service(name=name, service=component)


@dataclass(frozen=True)
class ComponentPowerConfig:
    """Resolved per-role power facts for one worker component.

    Built by :func:`resolve_component_power_configs` from the DGD-owned per-GPU
    annotation and the component's per-replica GPU total. ``watts_per_replica``
    is the value the power-budget projection and clamp consume.
    """

    component_name: str
    role: str  # prefill | decode | worker
    gpu_power_limit_watts: int
    gpus_per_replica: int

    @property
    def watts_per_replica(self) -> int:
        return self.gpu_power_limit_watts * self.gpus_per_replica


def _resolve_one_power_service(
    deployment: dict,
    sub_component_type: SubComponentType,
    component_name: Optional[str],
) -> Service:
    """Resolve a single role's worker Service without reading the power annotation.

    Role/name resolution delegates to ``get_component_from_type_or_name``, which
    already handles the unique-generic-worker fallback for agg (DECODE role with
    no typed decode component).  The except clause here covers only the case where
    multiple generic ``type: worker`` components exist: the shared resolver returns
    ``SubComponentNotFoundError`` in that situation (it cannot distinguish them),
    so this function converts it to a ``DuplicateSubComponentError`` to give callers
    an actionable diagnostic.
    """
    try:
        return get_component_from_type_or_name(
            deployment, sub_component_type, component_name=component_name
        )
    except SubComponentNotFoundError:
        if sub_component_type != SubComponentType.DECODE:
            raise
        components = get_components_by_name(deployment)
        generic_workers = [
            (curr_name, curr_component)
            for curr_name, curr_component in components.items()
            if get_component_type(curr_component)
            == V1BETA1_GENERIC_WORKER_COMPONENT_TYPE
        ]
        if len(generic_workers) == 1:
            name, component = generic_workers[0]
            return Service(name=name, service=component)
        if len(generic_workers) > 1:
            component_names = [name for name, _ in generic_workers]
            raise DuplicateSubComponentError(sub_component_type.value, component_names)
        raise


def _resolve_one_power_config(
    deployment: dict,
    sub_component_type: SubComponentType,
    component_name: Optional[str],
) -> ComponentPowerConfig:
    """Resolve a single role's power config, or raise a typed error."""
    service = _resolve_one_power_service(deployment, sub_component_type, component_name)
    watts = service.get_gpu_power_limit_watts(deployment)
    gpus_per_replica = service.get_gpu_shape(deployment).gpus_per_replica
    if gpus_per_replica <= 0:
        raise ValueError(
            f"Invalid operator-projected GPU count '{gpus_per_replica}' for "
            f"component '{service.name}'. GPU count must be a positive integer."
        )
    role = get_component_type(service.service) or sub_component_type.value
    return ComponentPowerConfig(
        component_name=service.name,
        role=role,
        gpu_power_limit_watts=watts,
        gpus_per_replica=gpus_per_replica,
    )


def resolve_power_component_names(
    deployment: dict,
    *,
    require_prefill: bool,
    require_decode: bool,
    prefill_name: Optional[str] = None,
    decode_name: Optional[str] = None,
) -> list[str]:
    """Return DGD component names whose Pods must carry the current power annotation.

    Uses the same role/name resolution as :func:`resolve_component_power_configs`
    (typed roles, explicit-name fallback for untyped workers, unique generic
    ``type: worker`` for agg) but does not read the power annotation — so the
    pod-annotation settlement gate can run before cap validation.
    """
    names: list[str] = []
    if require_prefill:
        names.append(
            _resolve_one_power_service(
                deployment, SubComponentType.PREFILL, prefill_name
            ).name
        )
    if require_decode:
        names.append(
            _resolve_one_power_service(
                deployment, SubComponentType.DECODE, decode_name
            ).name
        )
    # Preserve order but drop duplicates (agg decode-only should be unique).
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def resolve_component_power_configs(
    deployment: dict,
    *,
    require_prefill: bool,
    require_decode: bool,
    prefill_name: Optional[str] = None,
    decode_name: Optional[str] = None,
) -> tuple[Optional[ComponentPowerConfig], Optional[ComponentPowerConfig]]:
    """Resolve (prefill, decode) power configs from a DGD dict.

    Returns ``None`` for a role that is not required. Aggregate mode follows
    existing Planner semantics — ``require_prefill=False, require_decode=True``
    — and resolves the unique generic ``type: worker`` component as the decode
    slot; it does not manufacture a prefill config for that single worker.

    Raises the typed parser errors (``SubComponentNotFoundError``,
    ``DuplicateSubComponentError``, ``PowerAnnotationMissingError``,
    ``PowerAnnotationInvalidError``, or ``ValueError`` for a bad GPU count) so
    the caller can decide startup-fail vs runtime-conservative handling.
    """
    prefill_config = None
    decode_config = None
    if require_prefill:
        prefill_config = _resolve_one_power_config(
            deployment, SubComponentType.PREFILL, prefill_name
        )
    if require_decode:
        decode_config = _resolve_one_power_config(
            deployment, SubComponentType.DECODE, decode_name
        )
    return prefill_config, decode_config
