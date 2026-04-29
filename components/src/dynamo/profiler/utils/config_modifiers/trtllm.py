# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Tuple
from uuid import uuid4

import yaml

from dynamo.planner.config.defaults import SubComponentType
from dynamo.profiler.utils.config import (
    Config,
    append_argument,
    break_arguments,
    get_service_name_by_type,
    get_worker_service_from_config,
    remove_valued_arguments,
    setup_worker_service_resources,
    update_image,
    validate_and_get_worker_args,
)
from dynamo.profiler.utils.config_modifiers.protocol import BaseConfigModifier
from dynamo.profiler.utils.defaults import (
    DYNAMO_RUN_DEFAULT_PORT,
    EngineType,
    resolve_deploy_path,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

DEFAULT_TRTLLM_DISAGG_CONFIG_PATH = resolve_deploy_path(
    "examples/backends/trtllm/deploy/disagg.yaml"
)
DEFAULT_TRTLLM_AGG_CONFIG_PATH = resolve_deploy_path(
    "examples/backends/trtllm/deploy/agg.yaml"
)


def _trtllm_flags(overrides: dict) -> list[str]:
    """Build a flat ``--trtllm.<dotted.key> <value>`` flag list."""
    flags: list[str] = []
    for key, value in overrides.items():
        flags.append(f"--trtllm.{key}")
        if value is None:
            flags.append("none")
        elif isinstance(value, bool):
            flags.append(str(value).lower())
        else:
            flags.append(str(value))
    return flags


class TrtllmConfigModifier(BaseConfigModifier):
    BACKEND = "trtllm"

    @classmethod
    def load_default_config(cls, mode: str = "disagg") -> dict:
        path = (
            DEFAULT_TRTLLM_AGG_CONFIG_PATH
            if mode == "agg"
            else DEFAULT_TRTLLM_DISAGG_CONFIG_PATH
        )
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def update_image(cls, config, image: str) -> dict:
        """Update container image for all DGD services (frontend, planner, workers)."""
        return update_image(config, image)

    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: EngineType,
        is_moe_model: bool = False,
    ) -> dict:
        if is_moe_model:
            raise NotImplementedError(
                "MoE model support is not implemented for TrtLLM backend"
            )

        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = f"trtllm-agg-{uuid4().hex[:8]}"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == EngineType.PREFILL:
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

            args = append_argument(
                args,
                _trtllm_flags(
                    {
                        "kv_cache_config.enable_block_reuse": False,
                        "disable_overlap_scheduler": False,
                        "cache_transceiver_config": None,
                    }
                ),
            )

            worker_service.extraPodSpec.mainContainer.args = args

        elif target == EngineType.DECODE:
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

            args = append_argument(
                args,
                _trtllm_flags(
                    {
                        "kv_cache_config.enable_block_reuse": True,
                        "cache_transceiver_config": None,
                    }
                ),
            )

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
    ) -> dict:
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

        args = append_argument(args, _trtllm_flags({"tensor_parallel_size": tp_size}))

        worker_service.extraPodSpec.mainContainer.args = args

        return cfg.model_dump()

    @classmethod
    def set_config_tep_size(
        cls,
        config: dict,
        tep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
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
    ) -> dict:
        raise NotImplementedError(
            "DEP (Data Expert Parallelism) is not implemented for TrtLLM backend"
        )

    @classmethod
    def get_model_name(cls, config: dict) -> Tuple[str, str]:
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(cfg, backend="trtllm")
        args = validate_and_get_worker_args(worker_service, backend="trtllm")
        args = break_arguments(args)
        return cls._get_model_name_and_path_from_args(args)

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
        """Return TRT-LLM paged KV cache token capacity parsed from Dynamo logs.

        TRT-LLM may emit multiple memory allocation lines for paged KV cache
        during startup. This parser scans the full file and returns the token
        value from the last matching entry, which reflects the effective
        configured capacity.

        Args:
            dynamo_log_fn: Path to the Dynamo runtime log file.
            attention_dp_size: Unused for TRT-LLM; included for interface parity.

        Returns:
            Parsed max token count for paged KV cache, or ``100000`` when no
            matching log entry is found.
        """
        # TRT-LLM log parsing for KV cache size
        # Format: [TensorRT-LLM][INFO] [MemUsageChange] Allocated XX GiB for max tokens in paged KV cache (XXXXXX).
        max_tokens: int | None = None
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
        except Exception as e:
            logger.warning(f"Failed to parse KV cache size from log file. Error: {e}")

        if max_tokens is not None:
            logger.info(f"Found TRT-LLM KV cache max tokens: {max_tokens}")
            return max_tokens

        # Return a reasonable default if we couldn't find the KV cache size in logs
        logger.warning(
            "Could not find KV cache size in TRT-LLM logs, using default value of 100000"
        )
        return 100000  # Default fallback value for TRT-LLM

    @classmethod
    def set_prefill_config(
        cls,
        config: dict,
        max_batch_size: int,
        max_num_tokens: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """
        Configure prefill-related limits for aggregated prefill runs.
        For TRT-LLM we set these via ``--trtllm.*`` dynamic CLI flags:
        - max_batch_size
        - max_num_tokens
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="trtllm", sub_component_type=component_type
        )
        args = validate_and_get_worker_args(worker_service, backend="trtllm")
        args = break_arguments(args)

        args = append_argument(
            args,
            _trtllm_flags(
                {
                    "max_batch_size": int(max_batch_size),
                    "max_num_tokens": int(max_num_tokens),
                }
            ),
        )

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()
