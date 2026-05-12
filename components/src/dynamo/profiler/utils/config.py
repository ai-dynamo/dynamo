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

from pydantic import BaseModel

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


class Container(BaseModel):
    image: Optional[str] = None
    workingDir: Optional[str] = None
    command: Optional[list[str]] = None
    args: Optional[list[str]] = None
    resources: Optional[dict] = None  # For RDMA/custom resources
    model_config = {"extra": "allow"}


class PodSpec(BaseModel):
    mainContainer: Optional[Container] = None
    model_config = {"extra": "allow"}


class ServiceResources(BaseModel):
    requests: Optional[dict[str, str | dict]] = None
    limits: Optional[dict[str, str | dict]] = None


class Service(BaseModel):
    replicas: Optional[int] = None
    resources: Optional[ServiceResources] = None
    extraPodSpec: Optional[PodSpec] = None
    subComponentType: Optional[str] = None
    multinode: Optional[MultinodeConfig | dict[str, Any]] = None
    model_config = {"extra": "allow"}


class Services(BaseModel):
    Frontend: Service
    model_config = {"extra": "allow"}


class Spec(BaseModel):
    services: dict[str, Service]
    model_config = {"extra": "allow"}


class Metadata(BaseModel):
    name: str
    model_config = {"extra": "allow"}


class Config(BaseModel):
    metadata: Metadata
    spec: Spec
    model_config = {"extra": "allow"}


class MultinodeConfig(BaseModel):
    nodeCount: int


class DgdPlannerServiceConfig(BaseModel):
    """Planner service configuration.

    Planner reads profiling data from a ConfigMap (planner-profile-data)
    automatically created and mounted by the profiler; no PVC dependencies
    """

    componentType: str = "planner"
    replicas: int = 1
    extraPodSpec: PodSpec = PodSpec(
        mainContainer=Container(
            image="my-registry/dynamo-planner:my-tag",  # placeholder
            workingDir=f"{get_workspace_dir()}/components/src/dynamo/planner",
            command=["python3", "-m", "dynamo.planner"],
            args=[],
        )
    )
    model_config = {"extra": "allow"}


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


def _get_per_instance_gpus(worker_service: Service) -> int | None:
    """Derive per-instance GPU count from worker CLI args (TP x PP).

    Data-parallel workers are independent replicas, so multinode placement
    must be based on the GPUs required by a single instance rather than the
    total GPUs consumed across all replicas.
    """
    args: list[str] | None = None
    if (
        worker_service.extraPodSpec
        and worker_service.extraPodSpec.mainContainer
        and worker_service.extraPodSpec.mainContainer.args
    ):
        args = break_arguments(worker_service.extraPodSpec.mainContainer.args)

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
    worker_service: Service, gpu_count: int, num_gpus_per_node: int
) -> None:
    """Set multinode configuration based on per-instance GPU placement needs."""
    effective_gpu_count = _get_per_instance_gpus(worker_service) or gpu_count

    if effective_gpu_count <= num_gpus_per_node:
        # Single node: remove multinode configuration if present
        if (
            hasattr(worker_service, "multinode")
            and worker_service.multinode is not None
        ):
            worker_service.multinode = None
    else:
        # Multi-node: set nodeCount = math.ceil(per-instance GPUs / GPUs per node)
        node_count = math.ceil(effective_gpu_count / num_gpus_per_node)
        if not hasattr(worker_service, "multinode") or worker_service.multinode is None:
            # Create multinode configuration if it doesn't exist
            worker_service.multinode = MultinodeConfig(nodeCount=node_count)
        else:
            # Handle both dict (from YAML) and MultinodeConfig object cases
            if isinstance(worker_service.multinode, dict):
                worker_service.multinode["nodeCount"] = node_count
            else:
                worker_service.multinode.nodeCount = node_count


def get_service_name_by_type(
    config: Config, backend: str, sub_component_type: SubComponentType
) -> str:
    """Helper function to get service name by subComponentType.

    First tries to find service by subComponentType, then falls back to component name.

    Args:
        config: Configuration object
        backend: Backend name (e.g., "sglang", "vllm", "trtllm")
        sub_component_type: The type of sub-component to look for (PREFILL or DECODE)

    Returns:
        The service name
    """
    # Check if config has the expected structure
    if not config.spec or not config.spec.services:
        # Fall back to default name if structure is unexpected
        if sub_component_type == SubComponentType.DECODE:
            return WORKER_COMPONENT_NAMES[backend].decode_worker_k8s_name
        else:
            return WORKER_COMPONENT_NAMES[backend].prefill_worker_k8s_name

    # Look through services to find one with matching subComponentType
    services = config.spec.services
    for service_name, service_config in services.items():
        if service_config.subComponentType == sub_component_type.value:
            return service_name

    # Fall back to default component names
    if sub_component_type == SubComponentType.DECODE:
        default_name = WORKER_COMPONENT_NAMES[backend].decode_worker_k8s_name
    else:
        default_name = WORKER_COMPONENT_NAMES[backend].prefill_worker_k8s_name

    # Check if the default name exists in services
    if default_name in services:
        return default_name

    # Last resort: return the default name anyway
    return default_name


def get_worker_service_from_config(
    config: Config,
    backend: str = "sglang",
    sub_component_type: SubComponentType = SubComponentType.DECODE,
) -> Service:
    """Helper function to get a worker service from config.

    First tries to find service by subComponentType, then falls back to component name.

    Args:
        config: Configuration dictionary
        backend: Backend name (e.g., "sglang", "vllm", "trtllm"). Defaults to "sglang".
        sub_component_type: The type of sub-component to look for (PREFILL or DECODE). Defaults to DECODE.

    Returns:
        The worker service from the configuration
    """
    if backend not in WORKER_COMPONENT_NAMES:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends: {list(WORKER_COMPONENT_NAMES.keys())}"
        )

    # Get the service name using the type-aware logic
    service_name = get_service_name_by_type(config, backend, sub_component_type)

    # Get the actual service from the config
    return config.spec.services[service_name]


def setup_worker_service_resources(
    worker_service: Service, gpu_count: int, num_gpus_per_node: Optional[int] = None
) -> None:
    """Helper function to set up worker service resources (requests and limits)."""
    # Handle multinode configuration if num_gpus_per_node is provided
    if num_gpus_per_node is not None:
        set_multinode_config(worker_service, gpu_count, num_gpus_per_node)

    # Ensure resources exists
    if worker_service.resources is None:
        worker_service.resources = ServiceResources()

    # Ensure limits exists
    if worker_service.resources.limits is None:
        worker_service.resources.limits = {}

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
        resource_dict["gpu"] = str(gpu_value)

        # also update custom rdma/ib if it exists (some cluster requires this)
        if "custom" in resource_dict:
            if isinstance(resource_dict["custom"], dict):
                if "rdma/ib" in resource_dict["custom"]:
                    resource_dict["custom"]["rdma/ib"] = str(gpu_value)

    # Update limits
    _update_resource_dict(worker_service.resources.limits, gpu_value)
    # Also update requests if they exist
    if worker_service.resources.requests is not None:
        _update_resource_dict(worker_service.resources.requests, gpu_value)


def validate_and_get_worker_args(worker_service: Service, backend: str) -> list[str]:
    """Helper function to validate worker service and get its arguments.

    Args:
        worker_service: Worker service object to validate
        backend: Backend name (e.g., "sglang", "vllm", "trtllm"). Defaults to "sglang".

    Returns:
        List of arguments from the worker service
    """
    if backend not in WORKER_COMPONENT_NAMES:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends: {list(WORKER_COMPONENT_NAMES.keys())}"
        )

    if not worker_service.extraPodSpec or not worker_service.extraPodSpec.mainContainer:
        raise ValueError(
            f"Missing extraPodSpec or mainContainer in {backend} decode worker service '{WORKER_COMPONENT_NAMES[backend].decode_worker_k8s_name}'"
        )

    args = worker_service.extraPodSpec.mainContainer.args
    return break_arguments(args)


def set_argument_value(args: list[str], arg_name: str, value: str) -> list[str]:
    """Helper function to set an argument value, adding it if not present."""
    try:
        idx = args.index(arg_name)
        args[idx + 1] = value
    except ValueError:
        args = append_argument(args, [arg_name, value])
    return args


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

    for service_name, service_config in cfg.spec.services.items():
        if getattr(service_config, "componentType", None) == "planner":
            continue
        if service_config.extraPodSpec and service_config.extraPodSpec.mainContainer:
            service_config.extraPodSpec.mainContainer.image = image
            logger.debug(f"Updated image for {service_name} to {image}")

    return cfg.model_dump()
