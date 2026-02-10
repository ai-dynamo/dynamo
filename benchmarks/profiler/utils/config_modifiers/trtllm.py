# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import logging
import re
from typing import Tuple

import yaml

from benchmarks.profiler.utils.config import (
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
from benchmarks.profiler.utils.config_modifiers.protocol import BaseConfigModifier
from benchmarks.profiler.utils.defaults import DYNAMO_RUN_DEFAULT_PORT, EngineType
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


DEFAULT_TRTLLM_CONFIG_PATH = "examples/backends/trtllm/deploy/disagg.yaml"


class TrtllmConfigModifier(BaseConfigModifier):
    BACKEND = "trtllm"

    @classmethod
    def load_default_config(cls) -> dict:
        with open(DEFAULT_TRTLLM_CONFIG_PATH, "r") as f:
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
        # Deep copy to avoid mutating the original config dict
        # (MoE handling below modifies args which may share references with input)
        config = copy.deepcopy(config)
        cfg = Config.model_validate(config)

        if is_moe_model:
            # For MoE models, preserve MoE-related fields in override-engine-args JSON
            for sub_component_type in [SubComponentType.PREFILL, SubComponentType.DECODE]:
                try:
                    worker_service = get_worker_service_from_config(
                        cfg, backend="trtllm", sub_component_type=sub_component_type
                    )
                    args = validate_and_get_worker_args(worker_service, backend="trtllm")
                    args = break_arguments(args)

                    # Parse and preserve MoE fields in override-engine-args
                    override_dict, args = parse_override_engine_args(args)

                    # Ensure MoE fields are preserved (they may already be set)
                    # If not present, they'll be set by set_config_tep_size/set_config_dep_size
                    # Here we just ensure they're not lost during conversion

                    override_str = json.dumps(override_dict)
                    args = append_argument(args, ["--override-engine-args", override_str])

                    worker_service.extraPodSpec.mainContainer.args = args
                except (ValueError, KeyError):
                    logger.debug(
                        f"Skipping {sub_component_type} service as it doesn't exist"
                    )
                    continue

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
            cfg.spec.services[new_prefill_name] = cfg.spec.services.pop(old_prefill_name)

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

        # 5. Set WIDEEP MoE backend for DEP.
        #    WIDEEP is the only MoE backend compatible with DEP (attention DP enabled)
        #    on Blackwell (SM100), since the default CutlassFusedMoE/DeepGEMM is
        #    SM90-only. WIDEEP's DeepGemmMoEOp workspace scales with max_num_tokens
        #    × dep_size (tokens are allgathered before grouped GEMM); set_prefill_config
        #    automatically corrects for this by dividing per-rank tokens by dep_size.
        #    Note: set_config_tep_size strips WIDEEP since TEP is incompatible with it.
        if dep_size > 1:
            if "moe_config" not in override_dict:
                override_dict["moe_config"] = {}
            override_dict["moe_config"]["backend"] = "WIDEEP"

            # Add required environment variables for WIDEEP
            container = worker_service.extraPodSpec.mainContainer
            if container.env is None:
                container.env = []
            dep_envs = [
                {"name": "TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER", "value": "1"},
                {"name": "TRTLLM_ENABLE_PDL", "value": "1"},
            ]
            existing_env_names = {e.get("name") if isinstance(e, dict) else e.name for e in container.env}
            for env in dep_envs:
                if env["name"] not in existing_env_names:
                    container.env.append(env)

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
        max_batch_size_int = max(1, int(max_batch_size))
        max_num_tokens_int = max(1, int(max_num_tokens))

        # For attention-DP style prefill profiling, caller may pass a "total"
        # token cap across ranks; TRT-LLM build config is per-rank, so use a
        # per-rank cap to avoid over-allocation/OOM.
        per_rank_max_num_tokens = (
            max(1, max_num_tokens_int // max_batch_size_int)
            if max_batch_size_int > 1
            else max_num_tokens_int
        )

        # WIDEEP allgather correction: DeepGemmMoEOp's workspace is sized by
        # m_max = per_rank_max_num_tokens × dep_size (all DP ranks' tokens are
        # allgathered before the grouped GEMM). Without this correction the
        # effective token count becomes PREFILL_MAX_NUM_TOKENS × dep_size,
        # causing OOM.  Divide by dep_size so the post-allgather m_max stays
        # at PREFILL_MAX_NUM_TOKENS.
        moe_backend = override_dict.get("moe_config", {}).get("backend", "")
        if moe_backend == "WIDEEP":
            dep_size = max_batch_size_int  # DEP: batch_size = attn_dp = dep
            per_rank_max_num_tokens = max(1, per_rank_max_num_tokens // dep_size)
            logger.info(
                f"WIDEEP allgather correction: max_num_tokens per rank "
                f"reduced to {per_rank_max_num_tokens} (effective m_max "
                f"after allgather: {per_rank_max_num_tokens * dep_size})"
            )

        override_dict["max_batch_size"] = max_batch_size_int
        override_dict["max_num_tokens"] = per_rank_max_num_tokens
        override_str = json.dumps(override_dict)
        args = append_argument(args, ["--override-engine-args", override_str])

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()

    @classmethod
    def set_decode_config(
        cls,
        config: dict,
        max_batch_size: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """
        Configure decode batch limits for decode profiling runs.

        Removes any explicit max_batch_size / max_num_tokens overrides so
        TRT-LLM falls back to its built-in default (2048), which is large
        enough for the profiler's decode concurrency sweep.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="trtllm", sub_component_type=component_type
        )
        args = validate_and_get_worker_args(worker_service, backend="trtllm")
        args = break_arguments(args)

        override_dict, args = parse_override_engine_args(args)

        override_dict.pop("max_batch_size", None)
        override_dict.pop("max_num_tokens", None)

        override_str = json.dumps(override_dict)
        args = append_argument(args, ["--override-engine-args", override_str])

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()
