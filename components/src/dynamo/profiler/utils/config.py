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

from __future__ import annotations

import json
import logging
import math
import shlex
from typing import Any, Optional

from pydantic import BaseModel, Field

from dynamo.common.utils.paths import get_workspace_dir
from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class DgdModel(BaseModel):
    """Pydantic base that preserves Kubernetes field aliases on serialization."""

    model_config = {"extra": "allow", "populate_by_name": True}

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("by_alias", True)
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)


class Container(DgdModel):
    name: str = "main"
    image: Optional[str] = None
    workingDir: Optional[str] = None
    command: Optional[list[str]] = None
    args: Optional[list[str]] = None
    resources: Optional[dict] = None  # For RDMA/custom resources
    env: Optional[list[dict[str, Any]]] = None
    volumeMounts: Optional[list[dict[str, Any]]] = None


class PodSpec(DgdModel):
    containers: list[Container] = Field(default_factory=list)
    volumes: Optional[list[dict[str, Any]]] = None


class PodTemplate(DgdModel):
    spec: PodSpec = Field(default_factory=PodSpec)


class MultinodeConfig(DgdModel):
    nodeCount: int


class Component(DgdModel):
    name: str
    component_type: str = Field(alias="type")
    replicas: Optional[int] = None
    podTemplate: PodTemplate = Field(default_factory=PodTemplate)
    multinode: Optional[MultinodeConfig | dict[str, Any]] = None
    scalingAdapter: Optional[dict[str, Any]] = None
    runtimeVersionOverride: Optional[str] = None


class Spec(DgdModel):
    components: list[Component]


class Metadata(DgdModel):
    name: str
    namespace: Optional[str] = None


class Config(DgdModel):
    apiVersion: str = "nvidia.com/v1beta1"
    kind: str = "DynamoGraphDeployment"
    metadata: Metadata
    spec: Spec


class DgdPlannerComponentConfig(Component):
    """Planner component configuration.

    Planner reads profiling data from a ConfigMap (planner-profile-data)
    automatically created and mounted by the profiler; no PVC dependencies
    """

    name: str = "Planner"
    component_type: str = Field(default="planner", alias="type")
    replicas: int = 1
    podTemplate: PodTemplate = Field(
        default_factory=lambda: PodTemplate(
            spec=PodSpec(
                containers=[
                    Container(
                        image="my-registry/dynamo-planner:my-tag",  # placeholder
                        workingDir=f"{get_workspace_dir()}/components/src/dynamo/planner",
                        command=["python3", "-m", "dynamo.planner"],
                        args=[],
                    )
                ]
            )
        )
    )


def get_component_by_name(config: Config, name: str) -> Component | None:
    """Return a component by its stable DGD name."""
    return next(
        (component for component in config.spec.components if component.name == name),
        None,
    )


def find_main_container(component: Component) -> Container | None:
    """Return the component's ``main`` container when one is defined."""
    return next(
        (
            container
            for container in component.podTemplate.spec.containers
            if container.name == "main"
        ),
        None,
    )


def get_main_container(component: Component) -> Container:
    """Return the component's required ``main`` container.

    Raises:
        ValueError: If the component does not define a ``main`` container.
    """
    container = find_main_container(component)
    if container is not None:
        return container
    raise ValueError(f"Component {component.name!r} does not define a main container")


def get_component_dict(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Return a raw v1beta1 component dictionary by name."""
    components = config.get("spec", {}).get("components", [])
    if not isinstance(components, list):
        return None
    return next(
        (
            component
            for component in components
            if isinstance(component, dict) and component.get("name") == name
        ),
        None,
    )


def get_main_container_dict(component: dict[str, Any]) -> dict[str, Any] | None:
    """Return a raw component's ``main`` container dictionary."""
    containers = component.get("podTemplate", {}).get("spec", {}).get("containers", [])
    if not isinstance(containers, list):
        return None
    return next(
        (
            container
            for container in containers
            if isinstance(container, dict) and container.get("name") == "main"
        ),
        None,
    )


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
                # If the arg looks like it might be JSON (starts with { or [) or is already a single token,
                # don't split it further. Only split if it contains spaces AND doesn't look like JSON.
                if (
                    isinstance(arg, str)
                    and (" " in arg or "\t" in arg)
                    and not (arg.strip().startswith(("{", "[")))
                ):
                    # Use shlex.split to properly handle quoted arguments
                    ans.extend(shlex.split(arg))
                else:
                    ans.append(arg)
    return ans


def remove_valued_arguments(args: list[str], key: str) -> list[str]:
    """Remove a valued argument (e.g., --key value) from the arguments list if exists."""
    if key in args:
        idx = args.index(key)
        if idx + 1 < len(args):
            del args[idx : idx + 2]

    return args


def sanitize_cli_args(args: list[str]) -> list[str]:
    """Strip valued arguments whose value is the literal string ``"None"``.

    AIC's rule engine uses Jinja2 ``compile_expression`` which converts
    undefined variables to Python ``None``.  When that ``None`` is
    serialized into CLI args it becomes the four-character string
    ``"None"``, which is never a valid CLI value and causes backends
    (e.g. sglang ``--kv-cache-dtype None``) to reject the argument.
    """
    result = list(args)
    i = 0
    while i < len(result) - 1:
        if result[i].startswith("--") and result[i + 1] == "None":
            logger.warning(
                "Stripping CLI arg %s with invalid value 'None'",
                result[i],
            )
            del result[i : i + 2]
        else:
            i += 1
    return result


def append_argument(args: list[str], to_append: str | list[str]) -> list[str]:
    idx = find_arg_index(args)
    if isinstance(to_append, list):
        args[idx:idx] = to_append
    else:
        args.insert(idx, to_append)
    return args


def find_arg_index(args: list[str]) -> int:
    # find the correct index to insert an argument
    idx = len(args)

    try:
        new_idx = args.index("|")
        idx = min(idx, new_idx)
    except ValueError:
        pass

    try:
        new_idx = args.index("2>&1")
        idx = min(idx, new_idx)
    except ValueError:
        pass

    return idx


def parse_override_engine_args(args: list[str]) -> tuple[dict, list[str]]:
    """
    Parse and extract --override-engine-args from argument list.

    Returns:
        tuple: (override_dict, modified_args) where override_dict is the parsed JSON
               and modified_args is the args list with --override-engine-args removed
    """
    override_dict = {}
    try:
        idx = args.index("--override-engine-args")
        if idx + 1 < len(args):
            # Parse existing override
            override_dict = json.loads(args[idx + 1])
            # Remove the old override args
            del args[idx : idx + 2]
    except (ValueError, json.JSONDecodeError):
        pass  # No existing override or invalid JSON

    return override_dict, args


def get_requested_total_gpus(total_gpus_needed: Any) -> int | None:
    """Normalize a picked total GPU request from AIC output."""
    if total_gpus_needed is None:
        return None
    try:
        requested_total_gpus = int(total_gpus_needed)
    except (TypeError, ValueError):
        return None
    return requested_total_gpus if requested_total_gpus > 0 else None


def clamp_total_gpus_to_budget(
    requested_total_gpus: Any,
    total_gpu_budget: int,
) -> tuple[int, bool]:
    """Clamp a requested total GPU count to the deployment budget."""
    normalized_request = get_requested_total_gpus(requested_total_gpus)
    if normalized_request is None:
        return total_gpu_budget, False
    return (
        min(normalized_request, total_gpu_budget),
        normalized_request > total_gpu_budget,
    )


def _get_per_instance_gpus(worker_component: Component) -> int | None:
    """Derive per-instance GPU count from worker CLI args (TP x PP).

    Data-parallel workers are independent replicas, so multinode placement
    must be based on the GPUs required by a single instance rather than the
    total GPUs consumed across all replicas.
    """
    args: list[str] | None = None
    main_container = get_main_container(worker_component)
    if main_container.args:
        args = break_arguments(main_container.args)

    if not args:
        return None

    def _match_flag(
        arg: str, next_arg: str | None, names: tuple[str, ...]
    ) -> str | None:
        """Return the value for `arg` if it matches any of `names` in either
        `--name value` or `--name=value` form, else None."""
        for name in names:
            if arg == name:
                return next_arg
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
        return None

    TP_FLAGS = ("--tensor-parallel-size", "--tp")
    PP_FLAGS = ("--pipeline-parallel-size", "--pp")
    DP_FLAGS = ("--data-parallel-size", "--data-parallel-size-local", "--dp")

    tp = 1
    pp = 1
    saw_parallelism_flag = False
    for index, arg in enumerate(args):
        next_arg = args[index + 1] if index + 1 < len(args) else None

        tp_value = _match_flag(arg, next_arg, TP_FLAGS)
        if tp_value is not None:
            try:
                tp = int(tp_value)
                saw_parallelism_flag = True
            except ValueError:
                pass
            continue

        pp_value = _match_flag(arg, next_arg, PP_FLAGS)
        if pp_value is not None:
            try:
                pp = int(pp_value)
                saw_parallelism_flag = True
            except ValueError:
                pass
            continue

        if _match_flag(arg, next_arg, DP_FLAGS) is not None:
            saw_parallelism_flag = True

    if not saw_parallelism_flag:
        return None

    return tp * pp


def set_multinode_config(
    worker_component: Component, gpu_count: int, num_gpus_per_node: int
) -> None:
    """Set multinode configuration based on per-instance GPU placement needs."""
    effective_gpu_count = _get_per_instance_gpus(worker_component) or gpu_count

    if effective_gpu_count <= num_gpus_per_node:
        # Single node: remove multinode configuration if present
        if worker_component.multinode is not None:
            worker_component.multinode = None
    else:
        # Multi-node: set nodeCount = math.ceil(per-instance GPUs / GPUs per node)
        node_count = math.ceil(effective_gpu_count / num_gpus_per_node)
        if worker_component.multinode is None:
            # Create multinode configuration if it doesn't exist
            worker_component.multinode = MultinodeConfig(nodeCount=node_count)
        else:
            # Handle both dict (from YAML) and MultinodeConfig object cases
            if isinstance(worker_component.multinode, dict):
                worker_component.multinode["nodeCount"] = node_count
            else:
                worker_component.multinode.nodeCount = node_count


def get_component_name_by_type(
    config: Config, backend: str, sub_component_type: SubComponentType
) -> str:
    """Return a component name by its v1beta1 component type.

    First match ``spec.components[].type``, then fall back to the backend's
    canonical component name.

    Args:
        config: Configuration object
        backend: Backend name (e.g., "sglang", "vllm", "trtllm")
        sub_component_type: The type of sub-component to look for (PREFILL or DECODE)

    Returns:
        The component name
    """
    if not config.spec or not config.spec.components:
        if sub_component_type == SubComponentType.DECODE:
            return WORKER_COMPONENT_NAMES[backend].decode_worker_k8s_name
        return WORKER_COMPONENT_NAMES[backend].prefill_worker_k8s_name

    for component in config.spec.components:
        if component.component_type == sub_component_type.value:
            return component.name

    generic_workers = [
        component
        for component in config.spec.components
        if component.component_type == "worker"
    ]
    if len(generic_workers) == 1:
        return generic_workers[0].name

    # Fall back to default component names
    if sub_component_type == SubComponentType.DECODE:
        default_name = WORKER_COMPONENT_NAMES[backend].decode_worker_k8s_name
    else:
        default_name = WORKER_COMPONENT_NAMES[backend].prefill_worker_k8s_name

    return default_name


def get_worker_component_from_config(
    config: Config,
    backend: str = "sglang",
    sub_component_type: SubComponentType = SubComponentType.DECODE,
) -> Component:
    """Return a worker component from a v1beta1 config.

    First match the component type, then fall back to the canonical component name.

    Args:
        config: Configuration dictionary
        backend: Backend name (e.g., "sglang", "vllm", "trtllm"). Defaults to "sglang".
        sub_component_type: The type of sub-component to look for (PREFILL or DECODE). Defaults to DECODE.

    Returns:
        The worker component from the configuration
    """
    if backend not in WORKER_COMPONENT_NAMES:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends: {list(WORKER_COMPONENT_NAMES.keys())}"
        )

    component_name = get_component_name_by_type(config, backend, sub_component_type)
    component = get_component_by_name(config, component_name)
    if component is None:
        raise ValueError(f"Missing worker component {component_name!r}")
    return component


def setup_worker_component_resources(
    worker_component: Component,
    gpu_count: int,
    num_gpus_per_node: Optional[int] = None,
) -> None:
    """Set worker GPU resources on the v1beta1 main container."""
    # Handle multinode configuration if num_gpus_per_node is provided
    if num_gpus_per_node is not None:
        set_multinode_config(worker_component, gpu_count, num_gpus_per_node)

    main_container = get_main_container(worker_component)
    if main_container.resources is None:
        main_container.resources = {}
    limits = main_container.resources.setdefault("limits", {})

    # Calculate GPU value
    gpu_value = (
        min(gpu_count, num_gpus_per_node)
        if num_gpus_per_node is not None
        else gpu_count
    )

    def _update_resource_dict(
        resource_dict: dict[str, str | dict[str, Any]], gpu_value: int
    ) -> None:
        """Helper function to update gpu and custom rdma/ib fields in a resource dictionary.

        Args:
            resource_dict: The resource dictionary (either limits or requests) to update
            gpu_value: The GPU value to set
        """
        resource_dict["nvidia.com/gpu"] = str(gpu_value)
        if "rdma/ib" in resource_dict:
            resource_dict["rdma/ib"] = str(gpu_value)

    # Update limits
    _update_resource_dict(limits, gpu_value)
    requests = main_container.resources.get("requests")
    if isinstance(requests, dict):
        _update_resource_dict(requests, gpu_value)


def validate_and_get_worker_args(
    worker_component: Component, backend: str
) -> list[str]:
    """Validate a worker component and return its main-container arguments.

    Args:
        worker_component: Worker component object to validate
        backend: Backend name (e.g., "sglang", "vllm", "trtllm"). Defaults to "sglang".

    Returns:
        List of arguments from the worker service
    """
    if backend not in WORKER_COMPONENT_NAMES:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends: {list(WORKER_COMPONENT_NAMES.keys())}"
        )

    return break_arguments(get_main_container(worker_component).args)


def set_argument_value(args: list[str], arg_name: str, value: str) -> list[str]:
    """Helper function to set an argument value, adding it if not present."""
    try:
        idx = args.index(arg_name)
        args[idx + 1] = value
    except ValueError:
        args = append_argument(args, [arg_name, value])
    return args


def set_unique_argument_value(args: list[str], arg_name: str, value: str) -> list[str]:
    """Set one canonical value after removing every duplicate occurrence.

    Handles both ``--arg value`` and ``--arg=value`` forms. This is intended
    for identity-bearing arguments where appended DGD overrides must not leave
    a later conflicting value for the backend parser to consume.
    """
    filtered: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == arg_name:
            index += 2 if index + 1 < len(args) else 1
            continue
        if isinstance(arg, str) and arg.startswith(f"{arg_name}="):
            index += 1
            continue
        filtered.append(arg)
        index += 1

    return append_argument(filtered, [arg_name, value])


def update_image(config: dict, image: str) -> dict:
    """Update container image for non-planner DGD services.

    This is a shared utility function used by all backend config modifiers.

    Args:
        config: Configuration dictionary
        image: Container image to set for all services

    Returns:
        Updated configuration dictionary
    """
    cfg = Config.model_validate(config)

    for component in cfg.spec.components:
        if component.component_type == "planner":
            continue
        container = find_main_container(component)
        if container is None:
            logger.debug(
                "Skipping image update for component %s without a main container",
                component.name,
            )
            continue
        container.image = image
        logger.debug("Updated image for %s to %s", component.name, image)

    return cfg.model_dump()
