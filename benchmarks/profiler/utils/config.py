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
import re
import shlex
from typing import Literal, Optional, Protocol

from pydantic import BaseModel

from benchmarks.profiler.utils.defaults import (
    DEFAULT_MODEL_NAME,
    DYNAMO_RUN_DEFAULT_PORT,
)
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


class Container(BaseModel):
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


class Spec(BaseModel):
    services: dict[str, Service]


class Metadata(BaseModel):
    name: str


class Config(BaseModel):
    metadata: Metadata
    spec: Spec
    model_config = {"extra": "allow"}


class MultinodeConfig(BaseModel):
    nodeCount: int


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


def remove_valued_arguments(args: list[str], key: str) -> list[str]:
    """Remove a valued argument (e.g., --key value) from the arguments list if exists."""
    if key in args:
        idx = args.index(key)
        if idx + 1 < len(args):
            del args[idx : idx + 2]

    return args


def join_arguments(args: list[str]) -> list[str]:
    # Use shlex.join to properly quote arguments that contain spaces or special characters
    return [shlex.join(args)]


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
    config: dict, backend: str, sub_component_type: SubComponentType
) -> str:
    """Helper function to get service name by subComponentType.

    First tries to find service by subComponentType, then falls back to component name.

    Args:
        config: Configuration dictionary (with spec.services structure)
        backend: Backend name (e.g., "sglang", "vllm", "trtllm")
        sub_component_type: The type of sub-component to look for (PREFILL or DECODE)

    Returns:
        The service name
    """
    # Check if config has the expected structure
    if (
        not isinstance(config, dict)
        or "spec" not in config
        or "services" not in config.get("spec", {})
    ):
        # Fall back to default name if structure is unexpected
        if sub_component_type == SubComponentType.DECODE:
            return WORKER_COMPONENT_NAMES[backend].decode_worker_k8s_name
        else:
            return WORKER_COMPONENT_NAMES[backend].prefill_worker_k8s_name

    # Look through services to find one with matching subComponentType
    services = config["spec"]["services"]
    for service_name, service_config in services.items():
        if (
            isinstance(service_config, dict)
            and service_config.get("subComponentType") == sub_component_type.value
        ):
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
    config: dict,
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
    cfg = Config.model_validate(config)
    return cfg.spec.services[service_name]


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


class ConfigModifierProtocol(Protocol):
    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: Literal["prefill", "decode"],
        is_moe_model: bool = False,
    ) -> dict:
        ...

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int) -> dict:
        ...

    @classmethod
    def set_config_tep_size(
        cls, config: dict, tep_size: int, num_gpus_per_node: int
    ) -> dict:
        ...

    @classmethod
    def set_config_dep_size(
        cls, config: dict, dep_size: int, num_gpus_per_node: int
    ) -> dict:
        ...

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        ...

    @classmethod
    def get_port(cls, config: dict) -> int:
        ...

    @classmethod
    def get_kv_cache_size_from_dynamo_log(
        cls, dynamo_log_fn: str, attention_dp_size: int = 1
    ) -> int:
        ...


class VllmV1ConfigModifier:
    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: Literal["prefill", "decode"],
        is_moe_model: bool = False,
    ) -> dict:
        if is_moe_model:
            raise NotImplementedError(
                "MoE model support is not implemented for VLLM backend"
            )

        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = "vllm-agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == "prefill":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                config, "vllm", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                config, "vllm", SubComponentType.DECODE
            )

            # convert prefill worker into decode worker
            cfg.spec.services[decode_service_name] = cfg.spec.services[
                prefill_service_name
            ]
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated mode (using decode worker for prefill-only)
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            worker_service = get_worker_service_from_config(
                cfg.model_dump(),
                backend="vllm",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="vllm")
            args = break_arguments(args)

            # remove --is-prefill-worker flag
            args.remove("--is-prefill-worker")

            # disable prefix caching
            if "--enable-prefix-caching" in args:
                args.remove("--enable-prefix-caching")
            if "--no-enable-prefix-caching" not in args:
                args = append_argument(args, "--no-enable-prefix-caching")

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        elif target == "decode":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                config, "vllm", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                config, "vllm", SubComponentType.DECODE
            )

            # delete prefill worker
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated decode-only mode
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            worker_service = get_worker_service_from_config(
                cfg.model_dump(),
                backend="vllm",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="vllm")
            args = break_arguments(args)

            # enable prefix caching
            if "--enable-prefix-caching" not in args:
                args = append_argument(args, "--enable-prefix-caching")
            if "--no-enable-prefix-caching" in args:
                args.remove("--no-enable-prefix-caching")

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        # set num workers to 1
        # Use the inferred decode service name
        final_decode_service_name = get_service_name_by_type(
            cfg.model_dump(), "vllm", SubComponentType.DECODE
        )
        decode_worker_config = cfg.spec.services[final_decode_service_name]
        decode_worker_config.replicas = 1

        return cfg.model_dump()

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(config, backend="vllm")

        # Set up resources
        setup_worker_service_resources(worker_service, tp_size)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        try:
            idx = args.index("--tensor-parallel-size")
            args[idx + 1] = str(tp_size)
        except ValueError:
            args = append_argument(args, ["--tensor-parallel-size", str(tp_size)])

        worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        return cfg.model_dump()

    @classmethod
    def set_config_tep_size(cls, config: dict, tep_size: int, num_gpus_per_node: int):
        raise NotImplementedError(
            "TEP (Tensor Expert Parallelism) is not implemented for VLLM backend"
        )

    @classmethod
    def set_config_dep_size(cls, config: dict, dep_size: int, num_gpus_per_node: int):
        raise NotImplementedError(
            "DEP (Data Expert Parallelism) is not implemented for VLLM backend"
        )

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        try:
            worker_service = get_worker_service_from_config(config, backend="vllm")
            args = validate_and_get_worker_args(worker_service, backend="vllm")
        except (ValueError, KeyError):
            logger.warning(
                f"Worker service missing or invalid, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME

        args = break_arguments(args)
        for i, arg in enumerate(args):
            if arg == "--model" and i + 1 < len(args):
                return args[i + 1]

        logger.warning(
            f"Model name not found in configuration args, using default model name: {DEFAULT_MODEL_NAME}"
        )
        return DEFAULT_MODEL_NAME

    @classmethod
    def get_port(cls, config: dict) -> int:
        cfg = Config.model_validate(config)
        frontend_service = cfg.spec.services.get("Frontend")
        if (
            not frontend_service
            or not frontend_service.extraPodSpec
            or not frontend_service.extraPodSpec.mainContainer
        ):
            logger.warning(
                f"Frontend service or container not found, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

        args = frontend_service.extraPodSpec.mainContainer.args
        if not args:
            logger.warning(
                f"No args found in Frontend configuration, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

        args = break_arguments(args)
        try:
            idx = args.index("--http-port")
            return int(args[idx + 1])
        except (ValueError, IndexError):
            logger.warning(
                f"Port not found in configuration args, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

    @classmethod
    def get_kv_cache_size_from_dynamo_log(
        cls, dynamo_log_fn: str, attention_dp_size: int = 1
    ) -> int:
        try:
            with open(dynamo_log_fn, "r") as f:
                for line in f:
                    if "Maximum concurrency for" in line:
                        line = line.strip().split("Maximum concurrency for ")[1]
                        token_count = int(
                            line.split(" tokens per request: ")[0].replace(",", "")
                        )
                        concurrency = float(line.split(" tokens per request: ")[1][:-1])

                        logger.info(
                            f"Found KV cache info: {token_count} x {concurrency} = {int(token_count * concurrency)}"
                        )
                        return int(token_count * concurrency)
        except Exception as e:
            logger.warning(
                f"Failed to parse KV cache size from line: {line}. Error: {e}"
            )
        return 0


class SGLangConfigModifier:
    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: Literal["prefill", "decode"],
        is_moe_model: bool = False,
    ) -> dict:
        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = "sglang-agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == "prefill":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                config, "sglang", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                config, "sglang", SubComponentType.DECODE
            )

            # convert prefill worker into decode worker
            cfg.spec.services[decode_service_name] = cfg.spec.services[
                prefill_service_name
            ]
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated mode (using decode worker for prefill-only)
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            worker_service = get_worker_service_from_config(
                cfg.model_dump(),
                backend="sglang",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="sglang")
            args = break_arguments(args)

            # remove disagg flags
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-transfer-backend")
            args = remove_valued_arguments(args, "--disaggregation-bootstrap-port")

            # disable prefix caching
            if "--disable-radix-cache" not in args:
                args = append_argument(args, "--disable-radix-cache")

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        elif target == "decode":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                config, "sglang", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                config, "sglang", SubComponentType.DECODE
            )

            # delete prefill worker
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated decode-only mode
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            worker_service = get_worker_service_from_config(
                cfg.model_dump(),
                backend="sglang",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="sglang")
            args = break_arguments(args)

            # remove disagg flags
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-transfer-backend")
            args = remove_valued_arguments(args, "--disaggregation-bootstrap-port")

            # enable prefix caching
            if "--disable-radix-cache" in args:
                args.remove("--disable-radix-cache")

            if is_moe_model:
                # need to use round_robin dp attention routing for MoE models to ensure kv reuse can skip prefill
                if "--load-balance-method" in args:
                    idx = args.index("--load-balance-method")
                    args[idx + 1] = "round_robin"
                else:
                    args = append_argument(
                        args, ["--load-balance-method", "round_robin"]
                    )

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        # set num workers to 1
        # Use the inferred decode service name
        final_decode_service_name = get_service_name_by_type(
            cfg.model_dump(), "sglang", SubComponentType.DECODE
        )
        decode_worker_config = cfg.spec.services[final_decode_service_name]
        decode_worker_config.replicas = 1

        return cfg.model_dump()

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(config, backend="sglang")

        # Set up resources
        setup_worker_service_resources(worker_service, tp_size)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # Set --tp argument
        args = set_argument_value(args, "--tp", str(tp_size))

        worker_service.extraPodSpec.mainContainer.args = join_arguments(args)
        return cfg.model_dump()

    @classmethod
    def set_config_tep_size(cls, config: dict, tep_size: int, num_gpus_per_node: int):
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(config, backend="sglang")

        # Set up resources with multinode configuration
        setup_worker_service_resources(worker_service, tep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # 1. Set --tp=tep_size, if not present add it
        args = set_argument_value(args, "--tp", str(tep_size))

        # 2. Set --ep-size=tep_size, if not present add it
        args = set_argument_value(args, "--ep-size", str(tep_size))

        # 3. Remove --dp if present
        args = remove_valued_arguments(args, "--dp")

        # 4. Remove --enable-dp-attention if present
        if "--enable-dp-attention" in args:
            args.remove("--enable-dp-attention")

        worker_service.extraPodSpec.mainContainer.args = join_arguments(args)
        return cfg.model_dump()

    @classmethod
    def set_config_dep_size(cls, config: dict, dep_size: int, num_gpus_per_node: int):
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(config, backend="sglang")

        # Set up resources with multinode configuration
        setup_worker_service_resources(worker_service, dep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # 1. Set --tp=dep_size
        args = set_argument_value(args, "--tp", str(dep_size))

        # 2. Set --dp=dep_size (data parallelism across experts)
        args = set_argument_value(args, "--dp", str(dep_size))

        # 3. Enable --enable-dp-attention
        if "--enable-dp-attention" not in args:
            args = append_argument(args, "--enable-dp-attention")

        # 4. Set --ep-size=dep_size (expert parallelism size)
        args = set_argument_value(args, "--ep-size", str(dep_size))

        worker_service.extraPodSpec.mainContainer.args = join_arguments(args)
        return cfg.model_dump()

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        try:
            worker_service = get_worker_service_from_config(config, backend="sglang")
            args = validate_and_get_worker_args(worker_service, backend="sglang")
        except (ValueError, KeyError):
            logger.warning(
                f"Worker service missing or invalid, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME

        args = break_arguments(args)
        for i, arg in enumerate(args):
            if arg == "--served-model-name" and i + 1 < len(args):
                return args[i + 1]

        logger.warning(
            f"Model name not found in configuration args, using default model name: {DEFAULT_MODEL_NAME}"
        )
        return DEFAULT_MODEL_NAME

    @classmethod
    def get_port(cls, config: dict) -> int:
        cfg = Config.model_validate(config)
        frontend_service = cfg.spec.services.get("Frontend")
        if (
            not frontend_service
            or not frontend_service.extraPodSpec
            or not frontend_service.extraPodSpec.mainContainer
        ):
            logger.warning(
                f"Frontend service or container not found, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

        args = frontend_service.extraPodSpec.mainContainer.args
        if not args:
            logger.warning(
                f"No args found in Frontend configuration, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

        args = break_arguments(args)
        try:
            idx = args.index("--http-port")
            return int(args[idx + 1])
        except (ValueError, IndexError):
            logger.warning(
                f"Port not found in configuration args, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

    @classmethod
    def get_kv_cache_size_from_dynamo_log(
        cls, dynamo_log_fn: str, attention_dp_size: int = 1
    ) -> int:
        try:
            with open(dynamo_log_fn, "r") as f:
                for line in f:
                    if "KV Cache is allocated" in line and "#tokens:" in line:
                        # Extract the number after "#tokens:"
                        match = re.search(r"#tokens:\s*(\d+)", line)
                        if match:
                            return int(match.group(1)) * attention_dp_size
        except Exception as e:
            logger.warning(f"Failed to parse KV cache size from log file. Error: {e}")
        return 0


class TrtllmConfigModifier:
    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: Literal["prefill", "decode"],
        is_moe_model: bool = False,
    ) -> dict:
        if is_moe_model:
            raise NotImplementedError(
                "MoE model support is not implemented for TrtLLM backend"
            )

        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = "trtllm-agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == "prefill":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                config, "trtllm", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                config, "trtllm", SubComponentType.DECODE
            )

            # Convert to prefill-only aggregated setup
            # Rename prefill worker to decode worker name
            cfg.spec.services[decode_service_name] = cfg.spec.services[
                prefill_service_name
            ]
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated mode (using decode worker for prefill-only)
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            worker_service = get_worker_service_from_config(
                cfg.model_dump(),
                backend="trtllm",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="trtllm")
            args = break_arguments(args)

            # Remove disaggregation args
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-strategy")

            # Keep the original extra-engine-args (prefill.yaml) which may contain user settings
            # Check if user already has override-engine-args and merge with our changes
            override_dict, args = parse_override_engine_args(args)

            # Merge our overrides for converting prefill-only disagg to aggregated:
            # - Disable enable_block_reuse (no KV reuse for prefill-only)
            # - Enable overlap scheduler (disabled in prefill.yaml but needed for agg)
            # - Remove cache_transceiver_config (not needed in agg mode)
            if "kv_cache_config" not in override_dict:
                override_dict["kv_cache_config"] = {}
            override_dict["kv_cache_config"]["enable_block_reuse"] = False
            override_dict[
                "disable_overlap_scheduler"
            ] = False  # Enable overlap scheduler for agg
            override_dict[
                "cache_transceiver_config"
            ] = None  # Remove cache transceiver for agg

            override_str = json.dumps(override_dict)
            args = append_argument(args, ["--override-engine-args", override_str])

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        elif target == "decode":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                config, "trtllm", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                config, "trtllm", SubComponentType.DECODE
            )

            # Convert to decode-only aggregated setup
            # Remove prefill worker if exists
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated decode-only mode
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            # Decode worker already has the correct name
            worker_service = get_worker_service_from_config(
                cfg.model_dump(),
                backend="trtllm",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="trtllm")
            args = break_arguments(args)

            # Remove disaggregation args
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-strategy")

            # Keep the original extra-engine-args (decode.yaml) which may contain user settings
            # Check if user already has override-engine-args and merge with our changes
            override_dict, args = parse_override_engine_args(args)

            # Merge our overrides for converting decode-only disagg to aggregated:
            # - Enable enable_block_reuse (to skip prefill in decode-only)
            # - Remove cache_transceiver_config (not needed in agg mode)
            if "kv_cache_config" not in override_dict:
                override_dict["kv_cache_config"] = {}
            override_dict["kv_cache_config"]["enable_block_reuse"] = True
            override_dict[
                "cache_transceiver_config"
            ] = None  # Remove cache transceiver for agg

            override_str = json.dumps(override_dict)
            args = append_argument(args, ["--override-engine-args", override_str])

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        # Set num workers to 1
        # Use the inferred decode service name
        final_decode_service_name = get_service_name_by_type(
            cfg.model_dump(), "trtllm", SubComponentType.DECODE
        )
        worker_config = cfg.spec.services[final_decode_service_name]
        worker_config.replicas = 1

        return cfg.model_dump()

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        cfg = Config.model_validate(config)

        # Get the worker service using helper function
        # This assumes convert_config has been called, so the service is named decode_worker_k8s_name
        worker_service = get_worker_service_from_config(config, backend="trtllm")

        # Set up resources
        setup_worker_service_resources(worker_service, tp_size)

        # Validate and get args
        args = validate_and_get_worker_args(worker_service, backend="trtllm")

        # Break arguments to handle both joined strings and lists
        args = break_arguments(args)

        # For TRT-LLM, we need to update the override-engine-args
        # to set the tensor_parallel_size
        override_dict, args = parse_override_engine_args(args)

        # Add/update tensor_parallel_size in the override
        override_dict["tensor_parallel_size"] = tp_size
        override_str = json.dumps(override_dict)
        args = append_argument(args, ["--override-engine-args", override_str])

        worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        return cfg.model_dump()

    @classmethod
    def set_config_tep_size(cls, config: dict, tep_size: int, num_gpus_per_node: int):
        raise NotImplementedError(
            "TEP (Tensor Expert Parallelism) is not implemented for TrtLLM backend"
        )

    @classmethod
    def set_config_dep_size(cls, config: dict, dep_size: int, num_gpus_per_node: int):
        raise NotImplementedError(
            "DEP (Data Expert Parallelism) is not implemented for TrtLLM backend"
        )

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        try:
            worker_service = get_worker_service_from_config(config, backend="trtllm")
            args = validate_and_get_worker_args(worker_service, backend="trtllm")
        except (ValueError, KeyError):
            logger.warning(
                f"Worker service missing or invalid, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME

        args = break_arguments(args)
        for i, arg in enumerate(args):
            if arg == "--served-model-name" and i + 1 < len(args):
                return args[i + 1]

        logger.warning(
            f"Model name not found in configuration args, using default model name: {DEFAULT_MODEL_NAME}"
        )
        return DEFAULT_MODEL_NAME

    @classmethod
    def get_port(cls, config: dict) -> int:
        cfg = Config.model_validate(config)
        frontend_service = cfg.spec.services.get("Frontend")
        if (
            not frontend_service
            or not frontend_service.extraPodSpec
            or not frontend_service.extraPodSpec.mainContainer
        ):
            logger.warning(
                f"Frontend service or container not found, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

        # TRT-LLM frontend doesn't have args, it uses the default port
        return DYNAMO_RUN_DEFAULT_PORT

    @classmethod
    def get_kv_cache_size_from_dynamo_log(
        cls, dynamo_log_fn: str, attention_dp_size: int = 1
    ) -> int:
        # TRT-LLM log parsing for KV cache size
        # Format: [TensorRT-LLM][INFO] [MemUsageChange] Allocated XX GiB for max tokens in paged KV cache (XXXXXX).
        try:
            with open(dynamo_log_fn, "r") as f:
                for line in f:
                    # Look for the specific TRT-LLM KV cache allocation log
                    if (
                        "Allocated" in line
                        and "for max tokens in paged KV cache" in line
                    ):
                        # Extract the number in parentheses at the end
                        match = re.search(r"paged KV cache \((\d+)\)", line)
                        if match:
                            max_tokens = int(match.group(1))
                            logger.info(
                                f"Found TRT-LLM KV cache max tokens: {max_tokens}"
                            )
                            return max_tokens
        except Exception as e:
            logger.warning(f"Failed to parse KV cache size from log file. Error: {e}")

        # Return a reasonable default if we couldn't find the KV cache size in logs
        logger.warning(
            "Could not find KV cache size in TRT-LLM logs, using default value of 100000"
        )
        return 100000  # Default fallback value for TRT-LLM


CONFIG_MODIFIERS: dict[str, type[ConfigModifierProtocol]] = {
    "vllm": VllmV1ConfigModifier,
    "sglang": SGLangConfigModifier,
    "trtllm": TrtllmConfigModifier,
}

# Re-export WORKER_COMPONENT_NAMES for profile_sla.py
__all__ = ["CONFIG_MODIFIERS", "WORKER_COMPONENT_NAMES"]
