# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
import math
import shlex
from typing import Optional

from pydantic import BaseModel

from dynamo.common.utils.paths import get_workspace_dir
from dynamo.planner.defaults import WORKER_COMPONENT_NAMES, SubComponentType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class VolumeMount(BaseModel):
    name: str = "dynamo-pvc"
    mountPoint: str = "/data"


class Container(BaseModel):
    image: Optional[str] = None
    workingDir: Optional[str] = None
    command: Optional[list[str]] = None
    args: Optional[list[str]] = None
    model_config = {"extra": "allow"}


class PodSpec(BaseModel):
    mainContainer: Optional[Container] = None
    model_config = {"extra": "allow"}


class ServiceResources(BaseModel):
    requests: Optional[dict[str, str]] = None
    limits: Optional[dict[str, str]] = None


class Service(BaseModel):
    replicas: Optional[int] = None
    resources: Optional[ServiceResources] = None
    extraPodSpec: Optional[PodSpec] = None
    subComponentType: Optional[str] = None
    model_config = {"extra": "allow"}


class Services(BaseModel):
    Frontend: Service
    model_config = {"extra": "allow"}


class PVCConfig(BaseModel):
    name: str = "dynamo-pvc"
    create: Optional[bool] = False
    model_config = {"extra": "allow"}


class Spec(BaseModel):
    services: dict[str, Service]
    pvcs: Optional[list[PVCConfig]] = None
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
    dynamoNamespace: str = "dynamo"  # placeholder
    componentType: str = "planner"
    replicas: int = 1
    # Do not attach PVC; we'll mount a ConfigMap for planner data instead.
    volumeMounts: list[VolumeMount] = []
    extraPodSpec: PodSpec = PodSpec(
        mainContainer=Container(
            image="my-registry/dynamo-runtime:my-tag",  # placeholder
            workingDir=f"{get_workspace_dir()}/components/src/dynamo/planner",
            command=["python3", "-m", "planner_sla"],
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


def append_argument(args: list[str], to_append) -> list[str]:
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


def set_multinode_config(worker_service, gpu_count: int, num_gpus_per_node: int):
    """Helper function to set multinode configuration based on GPU count and GPUs per node."""
    if gpu_count <= num_gpus_per_node:
        # Single node: remove multinode configuration if present
        if (
            hasattr(worker_service, "multinode")
            and worker_service.multinode is not None
        ):
            worker_service.multinode = None
    else:
        # Multi-node: set nodeCount = math.ceil(gpu_count / num_gpus_per_node)
        node_count = math.ceil(gpu_count / num_gpus_per_node)
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
):
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
    worker_service, gpu_count: int, num_gpus_per_node: Optional[int] = None
):
    """Helper function to set up worker service resources (requests and limits)."""
    # Handle multinode configuration if num_gpus_per_node is provided
    if num_gpus_per_node is not None:
        set_multinode_config(worker_service, gpu_count, num_gpus_per_node)

    # Ensure resources exists
    if worker_service.resources is None:
        worker_service.resources = ServiceResources()

    # Ensure requests exists
    if worker_service.resources.requests is None:
        worker_service.resources.requests = {}

    # Set GPU requests
    gpu_value = (
        min(gpu_count, num_gpus_per_node)
        if num_gpus_per_node is not None
        else gpu_count
    )
    worker_service.resources.requests["gpu"] = str(gpu_value)

    # Update limits if they exist
    if worker_service.resources.limits is not None:
        worker_service.resources.limits["gpu"] = str(gpu_value)


def validate_and_get_worker_args(worker_service, backend):
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


def set_argument_value(args: list, arg_name: str, value: str):
    """Helper function to set an argument value, adding it if not present."""
    try:
        idx = args.index(arg_name)
        args[idx + 1] = value
    except ValueError:
        args = append_argument(args, [arg_name, value])
    return args


def update_image(config: dict, image: str) -> dict:
    """Update container image for all DGD services (frontend, planner, workers).

    This is a shared utility function used by all backend config modifiers.

    Args:
        config: Configuration dictionary
        image: Container image to set for all services

    Returns:
        Updated configuration dictionary
    """
    cfg = Config.model_validate(config)

    # Update image for all services
    for service_name, service_config in cfg.spec.services.items():
        if service_config.extraPodSpec and service_config.extraPodSpec.mainContainer:
            service_config.extraPodSpec.mainContainer.image = image
            logger.debug(f"Updated image for {service_name} to {image}")

    return cfg.model_dump()
