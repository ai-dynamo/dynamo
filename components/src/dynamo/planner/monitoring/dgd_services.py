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
from typing import Optional

from pydantic import BaseModel

from dynamo.common.utils.runtime import parse_endpoint
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.errors import DuplicateSubComponentError, SubComponentNotFoundError
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

MAIN_CONTAINER_NAME = "main"
V1BETA1_COMPONENT_TYPES = {"prefill", "decode"}
V1BETA1_GENERIC_WORKER_COMPONENT_TYPE = "worker"
GPU_RESOURCE_KEY = "nvidia.com/gpu"


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
        """Get the accelerator count for this component.

        Resolution order:
        1. ``resources.limits["nvidia.com/gpu"]`` or the matching request
           (NVIDIA device-plugin style)
        2. ``--tensor-parallel-size`` / ``--tp`` from the main container args
           (Intel DRA deployments omit GPU resources and use resource claims instead;
           TP size equals the number of devices per worker)

        Returns:
            The number of GPUs/XPUs configured for this component

        Raises:
            ValueError: If GPU count cannot be determined
        """
        main_container = get_main_container(self.service)
        resources = main_container.get("resources", {})
        limits = resources.get("limits", {})
        requests = resources.get("requests", {})

        # 1. Prefer limits, fall back to requests. For GPUs, Kubernetes device plugins
        # typically treat requests and limits as equivalent since GPUs are
        # non-compressible and allocated exclusively (no fractional sharing).
        gpu_str = limits.get(GPU_RESOURCE_KEY) or requests.get(GPU_RESOURCE_KEY)
        if gpu_str is not None:
            try:
                return int(gpu_str)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Invalid GPU count '{gpu_str}' for component '{self.name}'. "
                    f"GPU count must be an integer."
                ) from err

        # 2. DRA fallback: Intel XPU deployments use resourceClaims and do not set
        # nvidia.com/gpu. Infer device count from --tensor-parallel-size / --tp in
        # the container args (TP size == number of devices per worker).
        args = break_arguments(main_container.get("args", []))
        for i, arg in enumerate(args):
            if arg in ("--tensor-parallel-size", "--tp") and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except (ValueError, TypeError) as err:
                    raise ValueError(
                        f"Invalid --tensor-parallel-size value '{args[i + 1]}' "
                        f"for component '{self.name}'."
                    ) from err

        raise ValueError(
            f"No GPU/XPU count specified for component '{self.name}'. "
            f"Set main container resources.limits.{GPU_RESOURCE_KEY}, "
            f"resources.requests.{GPU_RESOURCE_KEY}, or --tensor-parallel-size/--tp."
        )


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
    elif not matching_components:
        raise SubComponentNotFoundError(component_type.value)

    name, component = matching_components[0]
    return Service(name=name, service=component)
