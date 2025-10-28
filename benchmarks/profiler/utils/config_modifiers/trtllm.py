# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from typing import Literal

import yaml

from benchmarks.profiler.utils.config import (
    Config,
    append_argument,
    break_arguments,
    get_service_name_by_type,
    get_worker_service_from_config,
    parse_override_engine_args,
    remove_valued_arguments,
    set_argument_value,
    setup_worker_service_resources,
    update_image,
    validate_and_get_worker_args,
)
from benchmarks.profiler.utils.defaults import (
    DEFAULT_MODEL_NAME,
    DYNAMO_RUN_DEFAULT_PORT,
)
from dynamo.planner.defaults import SubComponentType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


DEFAULT_TRTLLM_CONFIG_PATH = "components/backends/trtllm/deploy/disagg.yaml"


class TrtllmConfigModifier:
    @classmethod
    def load_default_config(cls) -> dict:
        with open(DEFAULT_TRTLLM_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def update_model(cls, config, model_name: str) -> dict:
        # change the model to serve
        cfg = Config.model_validate(config)

        # Update model for both prefill and decode workers
        for sub_component_type in [SubComponentType.PREFILL, SubComponentType.DECODE]:
            try:
                worker_service = get_worker_service_from_config(
                    cfg, backend="trtllm", sub_component_type=sub_component_type
                )
                args = validate_and_get_worker_args(worker_service, backend="trtllm")
                args = break_arguments(args)

                # Update both --model-path and --served-model-name
                args = set_argument_value(args, "--model-path", model_name)
                args = set_argument_value(args, "--served-model-name", model_name)

                worker_service.extraPodSpec.mainContainer.args = args
            except (ValueError, KeyError):
                # Service might not exist (e.g., in aggregated mode)
                logger.debug(
                    f"Skipping {sub_component_type} service as it doesn't exist"
                )
                continue

        return cfg.model_dump()

    @classmethod
    def update_image(cls, config, image: str) -> dict:
        """Update container image for all DGD services (frontend, planner, workers)."""
        return update_image(config, image)

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
                cfg, "trtllm", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                cfg, "trtllm", SubComponentType.DECODE
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
                cfg,
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
            if "kv_cache_config" not in override_dict or not isinstance(
                override_dict["kv_cache_config"], dict
            ):
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

            worker_service.extraPodSpec.mainContainer.args = args

        elif target == "decode":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                cfg, "trtllm", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                cfg, "trtllm", SubComponentType.DECODE
            )

            # Convert to decode-only aggregated setup
            # Remove prefill worker if exists
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated decode-only mode
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            # Decode worker already has the correct name
            worker_service = get_worker_service_from_config(
                cfg,
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
            if "kv_cache_config" not in override_dict or not isinstance(
                override_dict["kv_cache_config"], dict
            ):
                override_dict["kv_cache_config"] = {}
            override_dict["kv_cache_config"]["enable_block_reuse"] = True
            override_dict[
                "cache_transceiver_config"
            ] = None  # Remove cache transceiver for agg

            override_str = json.dumps(override_dict)
            args = append_argument(args, ["--override-engine-args", override_str])

            worker_service.extraPodSpec.mainContainer.args = args

        # Set num workers to 1
        # Use the inferred decode service name
        final_decode_service_name = get_service_name_by_type(
            cfg, "trtllm", SubComponentType.DECODE
        )
        worker_config = cfg.spec.services[final_decode_service_name]
        worker_config.replicas = 1

        return cfg.model_dump()

    @classmethod
    def set_config_tp_size(
        cls,
        config: dict,
        tp_size: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        cfg = Config.model_validate(config)

        # Get the worker service using helper function
        # This assumes convert_config has been called, so the service is named decode_worker_k8s_name
        worker_service = get_worker_service_from_config(
            cfg, backend="trtllm", sub_component_type=component_type
        )

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

        worker_service.extraPodSpec.mainContainer.args = args

        return cfg.model_dump()

    @classmethod
    def set_config_tep_size(
        cls,
        config: dict,
        tep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        raise NotImplementedError(
            "TEP (Tensor Expert Parallelism) is not implemented for TrtLLM backend"
        )

    @classmethod
    def set_config_dep_size(
        cls,
        config: dict,
        dep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        raise NotImplementedError(
            "DEP (Data Expert Parallelism) is not implemented for TrtLLM backend"
        )

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        cfg = Config.model_validate(config)
        try:
            worker_service = get_worker_service_from_config(cfg, backend="trtllm")
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
