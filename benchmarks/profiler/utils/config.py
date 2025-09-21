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
import re
import shlex
from typing import Literal, Optional, Protocol

from pydantic import BaseModel

from benchmarks.profiler.utils.defaults import (
    DEFAULT_MODEL_NAME,
    DYNAMO_RUN_DEFAULT_PORT,
)
from dynamo.planner.defaults import WORKER_COMPONENT_NAMES

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


class ConfigModifierProtocol(Protocol):
    @classmethod
    def convert_config(cls, config: dict, target: Literal["prefill", "decode"]) -> dict:
        ...

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int) -> dict:
        ...

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        ...

    @classmethod
    def get_port(cls, config: dict) -> int:
        ...

    @classmethod
    def get_kv_cache_size_from_dynamo_log(cls, dynamo_log_fn: str) -> int:
        ...


class VllmV1ConfigModifier:
    @classmethod
    def convert_config(cls, config: dict, target: Literal["prefill", "decode"]) -> dict:
        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = "vllm-agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == "prefill":
            # convert prefill worker into decode worker
            cfg.spec.services[
                WORKER_COMPONENT_NAMES["vllm"].decode_worker_k8s_name
            ] = cfg.spec.services[
                WORKER_COMPONENT_NAMES["vllm"].prefill_worker_k8s_name
            ]
            del cfg.spec.services[
                WORKER_COMPONENT_NAMES["vllm"].prefill_worker_k8s_name
            ]

            worker_service = cfg.spec.services[
                WORKER_COMPONENT_NAMES["vllm"].decode_worker_k8s_name
            ]
            if (
                not worker_service.extraPodSpec
                or not worker_service.extraPodSpec.mainContainer
            ):
                raise ValueError(
                    f"Missing extraPodSpec or mainContainer in VLLM decode worker service '{WORKER_COMPONENT_NAMES['vllm'].decode_worker_k8s_name}'"
                )
            args = worker_service.extraPodSpec.mainContainer.args

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
            # delete prefill worker
            del cfg.spec.services[
                WORKER_COMPONENT_NAMES["vllm"].prefill_worker_k8s_name
            ]

            worker_service = cfg.spec.services[
                WORKER_COMPONENT_NAMES["vllm"].decode_worker_k8s_name
            ]
            if (
                not worker_service.extraPodSpec
                or not worker_service.extraPodSpec.mainContainer
            ):
                raise ValueError(
                    f"Missing extraPodSpec or mainContainer in VLLM decode worker service '{WORKER_COMPONENT_NAMES['vllm'].decode_worker_k8s_name}'"
                )
            args = worker_service.extraPodSpec.mainContainer.args

            args = break_arguments(args)

            # enable prefix caching
            if "--enable-prefix-caching" not in args:
                args = append_argument(args, "--enable-prefix-caching")
            if "--no-enable-prefix-caching" in args:
                args.remove("--no-enable-prefix-caching")

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        # set num workers to 1
        decode_worker_config = cfg.spec.services[
            WORKER_COMPONENT_NAMES["vllm"].decode_worker_k8s_name
        ]
        decode_worker_config.replicas = 1

        return cfg.model_dump()

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        cfg = Config.model_validate(config)

        worker_service = cfg.spec.services[
            WORKER_COMPONENT_NAMES["vllm"].decode_worker_k8s_name
        ]

        # Ensure resources exists
        if worker_service.resources is None:
            worker_service.resources = ServiceResources()

        # Ensure requests exists
        if worker_service.resources.requests is None:
            worker_service.resources.requests = {}

        worker_service.resources.requests["gpu"] = str(tp_size)

        # Update limits if they exist
        if worker_service.resources.limits is not None:
            worker_service.resources.limits["gpu"] = str(tp_size)

        if (
            not worker_service.extraPodSpec
            or not worker_service.extraPodSpec.mainContainer
        ):
            raise ValueError(
                f"Missing extraPodSpec or mainContainer in VLLM decode worker service '{WORKER_COMPONENT_NAMES['vllm'].decode_worker_k8s_name}'"
            )
        args = worker_service.extraPodSpec.mainContainer.args

        args = break_arguments(args)

        try:
            idx = args.index("--tensor-parallel-size")
            args[idx + 1] = str(tp_size)
        except ValueError:
            args = append_argument(args, ["--tensor-parallel-size", str(tp_size)])

        worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        return cfg.model_dump()

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        cfg = Config.model_validate(config)
        worker_name = WORKER_COMPONENT_NAMES["vllm"].decode_worker_k8s_name
        worker_service = cfg.spec.services[worker_name]
        if (
            not worker_service.extraPodSpec
            or not worker_service.extraPodSpec.mainContainer
        ):
            logger.warning(
                f"Worker service missing extraPodSpec or mainContainer, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME
        args = worker_service.extraPodSpec.mainContainer.args

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
    def get_kv_cache_size_from_dynamo_log(cls, dynamo_log_fn: str) -> int:
        # TODO
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
    def convert_config(cls, config: dict, target: Literal["prefill", "decode"]) -> dict:
        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = "sglang-agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == "prefill":
            # convert prefill worker into decode worker
            cfg.spec.services[
                WORKER_COMPONENT_NAMES["sglang"].decode_worker_k8s_name
            ] = cfg.spec.services[
                WORKER_COMPONENT_NAMES["sglang"].prefill_worker_k8s_name
            ]
            del cfg.spec.services[
                WORKER_COMPONENT_NAMES["sglang"].prefill_worker_k8s_name
            ]

            worker_service = cfg.spec.services[
                WORKER_COMPONENT_NAMES["sglang"].decode_worker_k8s_name
            ]
            if (
                not worker_service.extraPodSpec
                or not worker_service.extraPodSpec.mainContainer
            ):
                raise ValueError(
                    f"Missing extraPodSpec or mainContainer in SGLang decode worker service '{WORKER_COMPONENT_NAMES['sglang'].decode_worker_k8s_name}'"
                )
            args = worker_service.extraPodSpec.mainContainer.args

            args = break_arguments(args)

            # remove `--disaggregation-mode` and `--disaggregation-transfer-backend`
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-transfer-backend")

            # disable prefix caching
            if "--disable-radix-cache" not in args:
                args = append_argument(args, "--disable-radix-cache")

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        elif target == "decode":
            # delete prefill worker
            del cfg.spec.services[
                WORKER_COMPONENT_NAMES["sglang"].prefill_worker_k8s_name
            ]

            worker_service = cfg.spec.services[
                WORKER_COMPONENT_NAMES["sglang"].decode_worker_k8s_name
            ]
            if (
                not worker_service.extraPodSpec
                or not worker_service.extraPodSpec.mainContainer
            ):
                raise ValueError(
                    f"Missing extraPodSpec or mainContainer in SGLang decode worker service '{WORKER_COMPONENT_NAMES['sglang'].decode_worker_k8s_name}'"
                )
            args = worker_service.extraPodSpec.mainContainer.args

            args = break_arguments(args)

            # remove `--disaggregation-mode` and `--disaggregation-transfer-backend`
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-transfer-backend")

            # enable prefix caching
            if "--disable-radix-cache" in args:
                args.remove("--disable-radix-cache")

            worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        # set num workers to 1
        decode_worker_config = config["spec"]["services"][
            WORKER_COMPONENT_NAMES["sglang"].decode_worker_k8s_name
        ]
        decode_worker_config["replicas"] = 1

        return config

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        cfg = Config.model_validate(config)

        worker_service = cfg.spec.services[
            WORKER_COMPONENT_NAMES["sglang"].decode_worker_k8s_name
        ]

        # Ensure resources exists
        if worker_service.resources is None:
            worker_service.resources = ServiceResources()

        # Ensure requests exists
        if worker_service.resources.requests is None:
            worker_service.resources.requests = {}

        worker_service.resources.requests["gpu"] = str(tp_size)

        # Update limits if they exist
        if worker_service.resources.limits is not None:
            worker_service.resources.limits["gpu"] = str(tp_size)

        if (
            not worker_service.extraPodSpec
            or not worker_service.extraPodSpec.mainContainer
        ):
            raise ValueError(
                f"Missing extraPodSpec or mainContainer in SGLang decode worker service '{WORKER_COMPONENT_NAMES['sglang'].decode_worker_k8s_name}'"
            )
        args = worker_service.extraPodSpec.mainContainer.args

        args = break_arguments(args)

        try:
            idx = args.index("--tp")
            args[idx + 1] = str(tp_size)
        except ValueError:
            args = append_argument(args, ["--tp", str(tp_size)])

        worker_service.extraPodSpec.mainContainer.args = join_arguments(args)

        return cfg.model_dump()

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        cfg = Config.model_validate(config)
        worker_name = WORKER_COMPONENT_NAMES["sglang"].decode_worker_k8s_name
        worker_service = cfg.spec.services[worker_name]
        if (
            not worker_service.extraPodSpec
            or not worker_service.extraPodSpec.mainContainer
        ):
            logger.warning(
                f"Worker service missing extraPodSpec or mainContainer, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME
        args = worker_service.extraPodSpec.mainContainer.args

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
    def get_kv_cache_size_from_dynamo_log(cls, dynamo_log_fn: str) -> int:
        # TODO
        try:
            with open(dynamo_log_fn, "r") as f:
                for line in f:
                    if "KV Cache is allocated" in line and "#tokens:" in line:
                        # Extract the number after "#tokens:"
                        match = re.search(r"#tokens:\s*(\d+)", line)
                        if match:
                            return int(match.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse KV cache size from log file. Error: {e}")
        return 0


class TrtllmConfigModifier:
    @classmethod
    def convert_config(cls, config: dict, target: Literal["prefill", "decode"]) -> dict:
        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = "trtllm-agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == "prefill":
            # Convert to prefill-only aggregated setup
            # Merge prefill worker config into a single worker
            if "TRTLLMPrefillWorker" in cfg.spec.services:
                # Rename prefill worker to generic worker
                cfg.spec.services["TRTLLMWorker"] = cfg.spec.services[
                    "TRTLLMPrefillWorker"
                ]
                del cfg.spec.services["TRTLLMPrefillWorker"]

            # Remove decode worker
            del cfg.spec.services["TRTLLMDecodeWorker"]

            worker_service = cfg.spec.services["TRTLLMWorker"]
            if (
                not worker_service.extraPodSpec
                or not worker_service.extraPodSpec.mainContainer
            ):
                raise ValueError(
                    "Missing extraPodSpec or mainContainer in TRTLLM worker service 'TRTLLMWorker'"
                )
            args = worker_service.extraPodSpec.mainContainer.args

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
            # Convert to decode-only aggregated setup
            # Use decode worker as the main worker
            if "TRTLLMDecodeWorker" in cfg.spec.services:
                # Rename decode worker to generic worker
                cfg.spec.services["TRTLLMWorker"] = cfg.spec.services[
                    "TRTLLMDecodeWorker"
                ]
                del cfg.spec.services["TRTLLMDecodeWorker"]

            # Remove prefill worker if exists
            if "TRTLLMPrefillWorker" in cfg.spec.services:
                del cfg.spec.services["TRTLLMPrefillWorker"]

            worker_service = cfg.spec.services["TRTLLMWorker"]
            if (
                not worker_service.extraPodSpec
                or not worker_service.extraPodSpec.mainContainer
            ):
                raise ValueError(
                    "Missing extraPodSpec or mainContainer in TRTLLM worker service 'TRTLLMWorker'"
                )
            args = worker_service.extraPodSpec.mainContainer.args

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
        worker_config = cfg.spec.services["TRTLLMWorker"]
        worker_config.replicas = 1

        return cfg.model_dump()

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        cfg = Config.model_validate(config)

        worker_service = cfg.spec.services["TRTLLMWorker"]

        # Ensure resources exists
        if worker_service.resources is None:
            worker_service.resources = ServiceResources()

        # Ensure requests exists
        if worker_service.resources.requests is None:
            worker_service.resources.requests = {}

        worker_service.resources.requests["gpu"] = str(tp_size)

        # Update limits if they exist
        if worker_service.resources.limits is not None:
            worker_service.resources.limits["gpu"] = str(tp_size)

        if (
            not worker_service.extraPodSpec
            or not worker_service.extraPodSpec.mainContainer
        ):
            raise ValueError(
                "Missing extraPodSpec or mainContainer in TRTLLM worker service 'TRTLLMWorker'"
            )
        args = worker_service.extraPodSpec.mainContainer.args

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
    def get_model_name(cls, config: dict) -> str:
        cfg = Config.model_validate(config)
        worker_name = "TRTLLMWorker"
        worker_service = cfg.spec.services.get(worker_name)

        # Also check for disagg worker names
        if not worker_service:
            worker_name = "TRTLLMPrefillWorker"
            worker_service = cfg.spec.services.get(worker_name)
        if not worker_service:
            worker_name = "TRTLLMDecodeWorker"
            worker_service = cfg.spec.services.get(worker_name)

        if not worker_service:
            logger.warning(
                f"Worker service not found, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME

        if (
            not worker_service.extraPodSpec
            or not worker_service.extraPodSpec.mainContainer
        ):
            logger.warning(
                f"Worker service missing extraPodSpec or mainContainer, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME
        args = worker_service.extraPodSpec.mainContainer.args

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
    def get_kv_cache_size_from_dynamo_log(cls, dynamo_log_fn: str) -> int:
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
