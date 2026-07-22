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
import shlex
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from dynamo.common.utils.runtime import parse_endpoint
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.errors import (
    DuplicateSubComponentError,
    PowerAnnotationInvalidError,
    PowerAnnotationMissingError,
    SubComponentNotFoundError,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

MAIN_CONTAINER_NAME = "main"
V1BETA1_COMPONENT_TYPES = {"prefill", "decode"}
V1BETA1_GENERIC_WORKER_COMPONENT_TYPE = "worker"
GPU_RESOURCE_KEY = "nvidia.com/gpu"

# Per-GPU power-limit annotation key (watts, positive integer).
#
# Ownership: this value is *authored* on the DGD worker component
# ``podTemplate.metadata.annotations`` by a human or the profiler. The operator
# renders it onto every worker Pod at create time; the Power Agent DaemonSet
# reads the *live Pod* annotation and applies the NVML/DCGM cap. The Planner
# only *reads* this value from the DGD to project a power budget — it never
# writes it onto Pods (see power-budget-architecture-rework.md). The Power Agent
# keeps its own copy of this literal (deploy/power-agent/power_agent.py); the
# two are asserted identical by a contract test rather than shared as a package
# import, because the agent image does not install the ``dynamo`` package.
POWER_ANNOTATION_KEY = "dynamo.nvidia.com/gpu-power-limit"


def break_arguments(args: list[str] | None) -> list[str]:
    ans: list[str] = []
    if args is None:
        return ans
    if isinstance(args, str):
        # Use shlex.split to properly handle quoted arguments and JSON values
        ans = shlex.split(args)
    else:
        for arg in args:
            if arg is not None:
                # Use shlex.split to properly handle quoted arguments
                ans.extend(shlex.split(arg))
    return ans


def _main_container_from_pod_template(component: dict) -> dict:
    containers = (
        component.get("podTemplate", {}).get("spec", {}).get("containers", []) or []
    )
    for container in containers:
        if container.get("name") == MAIN_CONTAINER_NAME:
            return container
    return {}


def get_main_container(component: dict) -> dict:
    """Return the planner-relevant v1beta1 main container."""
    return _main_container_from_pod_template(component)


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

    def get_model_name(self) -> Optional[str]:
        args = get_main_container(self.service).get("args", [])

        args = break_arguments(args)
        if (
            "--served-model-name" in args
            and len(args) > args.index("--served-model-name") + 1
        ):
            return args[args.index("--served-model-name") + 1]
        if (
            "--model-name" in args and len(args) > args.index("--model-name") + 1
        ):  # mocker use --model-name
            return args[args.index("--model-name") + 1]
        if "--model" in args and len(args) > args.index("--model") + 1:
            return args[args.index("--model") + 1]

        return None

    def get_component_name_from_endpoint_arg(self) -> Optional[str]:
        """Return the component name from ``--endpoint`` in the container args.

        Worker backends (vLLM, SGLang, TRT-LLM) accept
        ``--endpoint <namespace>.<component>.<endpoint_name>`` (optionally
        prefixed with ``dyn://``) which overrides the default component
        name written to the MDC ``component`` field. When the user sets
        this, the Planner's MDC filter must match the user's value, not
        the backend default. Returns ``None`` if ``--endpoint`` is not
        present or malformed.
        """
        args = get_main_container(self.service).get("args", [])
        args = break_arguments(args)
        if "--endpoint" not in args:
            return None
        idx = args.index("--endpoint")
        if len(args) <= idx + 1:
            return None
        try:
            _, component, _ = parse_endpoint(args[idx + 1])
            return component
        except ValueError:
            return None

    def get_gpu_count(self) -> int:
        """Get the GPU count from the component's resource specification.

        GPU count is read from the v1beta1 main container resources
        (``nvidia.com/gpu``).

        Returns:
            The number of GPUs configured for this component

        Raises:
            ValueError: If GPU count is not specified or invalid
        """
        resources = get_main_container(self.service).get("resources", {})
        limits = resources.get("limits", {})
        requests = resources.get("requests", {})

        # Prefer limits, fall back to requests. For GPUs, Kubernetes device plugins
        # typically treat requests and limits as equivalent since GPUs are
        # non-compressible and allocated exclusively (no fractional sharing).
        gpu_str = limits.get(GPU_RESOURCE_KEY) or requests.get(GPU_RESOURCE_KEY)

        if gpu_str is None:
            raise ValueError(
                f"No GPU count specified for component '{self.name}'. "
                f"Please set main container resources.limits.{GPU_RESOURCE_KEY} "
                f"or resources.requests.{GPU_RESOURCE_KEY} in the DGD."
            )

        try:
            gpu_count = int(gpu_str)
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Invalid GPU count '{gpu_str}' for component '{self.name}'. "
                f"GPU count must be a positive integer."
            ) from err
        # A zero/negative GPU count is nonsensical and, for power projection,
        # would make watts_per_replica zero — silently disabling enforcement for
        # this role. Reject it so the deployment fails loudly instead.
        if gpu_count <= 0:
            raise ValueError(
                f"Invalid GPU count '{gpu_str}' for component '{self.name}'. "
                f"GPU count must be a positive integer."
            )
        return gpu_count

    def get_node_count(self) -> int:
        """Return multinode.nodeCount from the component spec, defaulting to 1.

        The operator CRD defines total GPUs as nodeCount × per-pod GPU request.
        Single-node components either omit the field or set it to 1.

        Raises:
            ValueError: nodeCount is present but not a positive integer (a
                zero/negative value would zero out watts_per_replica and
                silently disable power enforcement for the role).
        """
        raw = self.service.get("multinode", {}).get("nodeCount", 1)
        try:
            node_count = int(raw)
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Invalid multinode.nodeCount '{raw}' for component "
                f"'{self.name}'. nodeCount must be a positive integer."
            ) from err
        if node_count <= 0:
            raise ValueError(
                f"Invalid multinode.nodeCount '{raw}' for component "
                f"'{self.name}'. nodeCount must be a positive integer."
            )
        return node_count

    def get_total_gpu_count(self) -> int:
        """Return total GPUs consumed by one replica: get_gpu_count() × get_node_count().

        For single-node components this equals get_gpu_count(). For multinode
        components the operator allocates nodeCount pods per replica each
        carrying the same per-pod GPU request, so power projection must
        multiply both factors.
        """
        return self.get_gpu_count() * self.get_node_count()

    def get_gpu_power_limit_watts(self) -> int:
        """Return ``dynamo.nvidia.com/gpu-power-limit`` from podTemplate annotations.

        The per-GPU cap is read from the worker component's
        ``podTemplate.metadata.annotations``. This is the *desired* static cap
        the operator stamps onto Pods and the Power Agent enforces; the Planner
        only reads it for power projection.

        Raises:
            PowerAnnotationMissingError: annotation key is absent.
            PowerAnnotationInvalidError: value is empty, non-integer, or <= 0.
        """
        annotations = (
            self.service.get("podTemplate", {}).get("metadata", {}).get("annotations")
            or {}
        )
        raw = annotations.get(POWER_ANNOTATION_KEY)
        if raw is None:
            raise PowerAnnotationMissingError(self.name)
        try:
            watts = int(str(raw).strip())
        except (ValueError, TypeError):
            raise PowerAnnotationInvalidError(self.name, str(raw))
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

    if not matching_components and component_name in components:
        component = components[component_name]
        if not _can_use_explicit_component_name(component, component_type):
            raise SubComponentNotFoundError(component_type.value)
        matching_components.append((component_name, component))
    elif not matching_components and component_type == SubComponentType.DECODE:
        generic_workers = [
            (curr_name, curr_component)
            for curr_name, curr_component in components.items()
            if get_component_type(curr_component)
            == V1BETA1_GENERIC_WORKER_COMPONENT_TYPE
        ]
        if len(generic_workers) == 1:
            matching_components.append(generic_workers[0])
        elif len(generic_workers) > 1:
            component_names = [name for name, _ in generic_workers]
            raise DuplicateSubComponentError(component_type.value, component_names)
        else:
            raise SubComponentNotFoundError(component_type.value)
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
    gpus_per_replica: int  # Service.get_total_gpu_count() (nodeCount × per-pod GPUs)

    @property
    def watts_per_replica(self) -> int:
        return self.gpu_power_limit_watts * self.gpus_per_replica


def _resolve_one_power_config(
    deployment: dict,
    sub_component_type: SubComponentType,
    component_name: Optional[str],
) -> ComponentPowerConfig:
    """Resolve a single role's power config, or raise a typed error.

    Role/name resolution reuses ``get_component_from_type_or_name`` so disagg
    (typed ``prefill``/``decode``) and agg (generic ``type: worker``) resolve
    the same way GPU-count and model-name lookups already do.
    """
    service = get_component_from_type_or_name(
        deployment, sub_component_type, component_name=component_name
    )
    watts = service.get_gpu_power_limit_watts()
    gpus_per_replica = service.get_total_gpu_count()
    role = get_component_type(service.service) or sub_component_type.value
    return ComponentPowerConfig(
        component_name=service.name,
        role=role,
        gpu_power_limit_watts=watts,
        gpus_per_replica=gpus_per_replica,
    )


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
