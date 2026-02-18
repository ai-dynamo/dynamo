# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from typing import Tuple

import yaml

from dynamo.planner.defaults import SubComponentType
from dynamo.profiler.utils.config import (
    Config,
    append_argument,
    break_arguments,
    get_service_name_by_type,
    get_worker_service_from_config,
    parse_override_engine_args,
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
        cfg = Config.model_validate(config)

        # set metadata name (short to avoid Grove 45-char limit for multinode)
        cfg.metadata.name = "agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        # Rename services to shorter names to avoid Grove 45-char naming limit for multinode
        # TRTLLMDecodeWorker (18 chars) -> dec (3 chars)
        # TRTLLMPrefillWorker (18 chars) -> pre (3 chars)
        old_decode_name = "TRTLLMDecodeWorker"
        old_prefill_name = "TRTLLMPrefillWorker"
        new_decode_name = "dec"
        new_prefill_name = "pre"

        if old_decode_name in cfg.spec.services:
            cfg.spec.services[new_decode_name] = cfg.spec.services.pop(old_decode_name)
        if old_prefill_name in cfg.spec.services:
            cfg.spec.services[new_prefill_name] = cfg.spec.services.pop(
                old_prefill_name
            )

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
        """
        Set Tensor Expert Parallelism (TEP) for TensorRT-LLM MoE models.

        TRTLLM uses JSON fields in --override-engine-args.
        All MoE configuration is done via JSON, not command-line args.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="trtllm", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_service_resources(worker_service, tep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="trtllm")
        args = break_arguments(args)

        # Parse existing override-engine-args (if any) and update
        override_dict, args = parse_override_engine_args(args)

        # 1. Set tensor_parallel_size=tep_size (splits KV heads)
        override_dict["tensor_parallel_size"] = tep_size

        # 2. Set moe_expert_parallel_size=tep_size (distributes experts across GPUs)
        override_dict["moe_expert_parallel_size"] = tep_size

        # 3. Set moe_tensor_parallel_size=1 (each expert's weights fully on one GPU)
        override_dict["moe_tensor_parallel_size"] = 1

        # 4. Disable attention DP (TEP uses TP for attention)
        override_dict["enable_attention_dp"] = False

        # 5. Remove WIDEEP backend if present -- WIDEEP requires attention DP
        #    which is incompatible with TEP. Let TRT-LLM use its default backend.
        moe_config = override_dict.get("moe_config")
        if isinstance(moe_config, dict) and moe_config.get("backend") == "WIDEEP":
            del moe_config["backend"]
            if not moe_config:
                del override_dict["moe_config"]

        # Serialize JSON and append to args
        override_str = json.dumps(override_dict)
        args = append_argument(args, ["--override-engine-args", override_str])

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()

    @classmethod
    def set_config_dep_size(
        cls,
        config: dict,
        dep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        """
        Set Data Expert Parallelism (DEP) for TensorRT-LLM MoE models.

        TRTLLM uses JSON fields in --override-engine-args.
        All MoE configuration is done via JSON, not command-line args.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="trtllm", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_service_resources(worker_service, dep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="trtllm")
        args = break_arguments(args)

        # Parse existing override-engine-args (if any) and update
        override_dict, args = parse_override_engine_args(args)

        # 1. Set tensor_parallel_size=dep_size (use all GPUs)
        #    Attention DP below ensures KV heads aren't split
        override_dict["tensor_parallel_size"] = dep_size

        # 2. Set moe_expert_parallel_size=dep_size (distributes experts across GPUs)
        override_dict["moe_expert_parallel_size"] = dep_size

        # 3. Set moe_tensor_parallel_size=1 (each expert's weights fully on one GPU)
        override_dict["moe_tensor_parallel_size"] = 1

        # 4. Enable attention DP (replicates KV heads, partitions requests)
        override_dict["enable_attention_dp"] = True

        # 5. Set WIDEEP MoE backend for DEP on Blackwell (SM100).
        #    Also set moe_config.max_num_tokens to bound the MoE workspace
        #    independently of the top-level max_num_tokens. Without this, the
        #    DeepGemmMoEOp workspace defaults to the top-level max_num_tokens
        #    and can OOM when it is set high for prefill.
        #    Note: The pydantic field is MoeConfig.max_num_tokens (not
        #    moe_max_num_tokens). The top-level max_num_tokens controls sequence
        #    chunking; moe_config.max_num_tokens controls MoE workspace size.
        #    Reference: deepseek-r1/agg/wide_ep/wide_ep_agg.yaml uses
        #    max_num_tokens = max_batch_size * ep_size = 256 * 16 = 4096.
        if dep_size > 1:
            if "moe_config" not in override_dict:
                override_dict["moe_config"] = {}
            override_dict["moe_config"]["backend"] = "WIDEEP"
            # moe_config.max_num_tokens bounds the DeepGemmMoEOp workspace per
            # EP rank. Derive it from the base config's max_num_tokens (e.g.
            # decode.yaml sets max_num_tokens=256 for disagg decode) so this
            # works for any model rather than hardcoding 256.
            base_max_num_tokens = override_dict.get("max_num_tokens", 256)
            override_dict["moe_config"]["max_num_tokens"] = (
                dep_size * base_max_num_tokens
            )

            # Add required environment variables for WIDEEP
            container = worker_service.extraPodSpec.mainContainer
            if container.env is None:
                container.env = []
            existing_env_names = {
                e["name"] if isinstance(e, dict) else e.name for e in container.env
            }
            for name, value in [
                ("TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER", "1"),
                ("TRTLLM_ENABLE_PDL", "1"),
            ]:
                if name not in existing_env_names:
                    container.env.append({"name": name, "value": value})

        # Serialize JSON and append to args
        override_str = json.dumps(override_dict)
        args = append_argument(args, ["--override-engine-args", override_str])

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()

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
                            kv_tokens_per_rank = int(match.group(1))
                            total_kv_tokens = kv_tokens_per_rank * max(
                                1, int(attention_dp_size)
                            )
                            logger.info(
                                f"Found TRT-LLM KV cache: {kv_tokens_per_rank} per rank x {attention_dp_size} = {total_kv_tokens} total"
                            )
                            return total_kv_tokens
        except Exception as e:
            logger.warning(f"Failed to parse KV cache size from log file. Error: {e}")

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
        For TRT-LLM we set these via --override-engine-args JSON:
        - max_batch_size
        - max_num_tokens
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="trtllm", sub_component_type=component_type
        )
        args = validate_and_get_worker_args(worker_service, backend="trtllm")
        args = break_arguments(args)

        # Parse existing override-engine-args (if any) and update
        override_dict, args = parse_override_engine_args(args)

        # Note: max_batch_size here is the attention DP size (== DEP size for
        # MoE), not a literal batch size.  The caller in
        # parallelization_mapping passes mapping.get_attn_dp_size().
        attn_dp_size = max(1, int(max_batch_size))
        max_num_tokens_int = max(1, int(max_num_tokens))

        # TRT-LLM build config is per-rank, so divide the total token cap by
        # the number of attention DP ranks.
        per_rank_max_num_tokens = (
            max(1, max_num_tokens_int // attn_dp_size)
            if attn_dp_size > 1
            else max_num_tokens_int
        )

        override_dict["max_batch_size"] = attn_dp_size
        override_dict["max_num_tokens"] = per_rank_max_num_tokens
        override_str = json.dumps(override_dict)
        args = append_argument(args, ["--override-engine-args", override_str])

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()

    @classmethod
    def set_decode_config(
        cls,
        config: dict,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """
        Configure decode engine limits for decode profiling runs.

        Removes max_batch_size so TRT-LLM auto-tunes for decode concurrency
        sweeps (default 2048). Does not override max_num_tokens -- the base
        config value (e.g. 8192 from qwen3) is sufficient for decode, and
        moe_config.max_num_tokens (set by set_config_dep_size) independently
        bounds the WIDEEP MoE workspace.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="trtllm", sub_component_type=component_type
        )
        args = validate_and_get_worker_args(worker_service, backend="trtllm")
        args = break_arguments(args)

        override_dict, args = parse_override_engine_args(args)

        # Remove max_batch_size to let TRT-LLM auto-tune (default 2048).
        override_dict.pop("max_batch_size", None)

        override_str = json.dumps(override_dict)
        args = append_argument(args, ["--override-engine-args", override_str])

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()
