# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import Any, List, Optional

import sglang as sgl
from sglang.srt.environ import envs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.server_args import ServerArgs

from dynamo._core import Endpoint
from dynamo.common.utils.output_modalities import get_output_modalities
from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_model
from dynamo.sglang._compat import NetworkAddress, get_local_ip_auto
from dynamo.sglang.args import DynamoConfig


SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY = "sglang_hicache_mooncake"


async def _register_model_with_runtime_config(
    engine: sgl.Engine,
    endpoint: Endpoint,
    server_args: ServerArgs,
    dynamo_args: DynamoConfig,
    input_type: ModelInput = ModelInput.Tokens,
    output_type: ModelType = ModelType.Chat | ModelType.Completions,
) -> bool:
    """Register LLM with the Dynamo runtime.

    Args:
        engine: The SGLang engine instance.
        endpoint: The Dynamo endpoint for communication.
        server_args: SGLang server configuration.
        dynamo_args: Dynamo-specific configuration.
        input_type: Expected model input type. Defaults to ModelInput.Tokens.
        output_type: Expected model output type. Defaults to ModelType.Chat | ModelType.Completions.

    Returns:
        True if registration succeeded, False otherwise.
    """
    runtime_config = await _get_runtime_config(engine, server_args, dynamo_args)

    if dynamo_args.use_sglang_tokenizer:
        logging.warning(
            "Using the sglang tokenizer/detokenizer instead. The dynamo tokenizer/detokenizer will not be used and only v1/chat/completions will be available"
        )
        input_type = ModelInput.Text
        # Only override output_type for chat models, not for embeddings
        if output_type != ModelType.Embedding:
            output_type = ModelType.Chat

    try:
        await register_model(
            input_type,
            output_type,
            endpoint,
            server_args.model_path,
            server_args.served_model_name,
            context_length=server_args.context_length,
            kv_cache_block_size=server_args.page_size,
            runtime_config=runtime_config,
            custom_template_path=dynamo_args.custom_jinja_template,
        )
        logging.info("Successfully registered LLM with runtime config")
        return True
    except Exception as e:
        logging.error(f"Failed to register with runtime config: {e}")
        return False


def _get_bootstrap_info_for_config(
    engine: sgl.Engine,
) -> tuple[Optional[str], Optional[int]]:
    """Extract bootstrap host and port from SGLang engine for config registration.

    Args:
        engine: The SGLang engine instance.

    Returns:
        Tuple of (bootstrap_host, bootstrap_port), or (None, None) if not available.
    """
    try:
        inner_tm = engine.tokenizer_manager
        bootstrap_port = getattr(
            inner_tm.server_args, "disaggregation_bootstrap_port", None
        )

        if bootstrap_port is None:
            return None, None

        if inner_tm.server_args.dist_init_addr:
            dist_init = NetworkAddress.parse(inner_tm.server_args.dist_init_addr)
            resolved = dist_init.resolved()
            bootstrap_host = (
                NetworkAddress(resolved.host, bootstrap_port)
                .to_host_port_str()
                .rsplit(":", 1)[0]
            )
            logging.info(
                f"Resolved bootstrap host '{dist_init.host}' -> '{resolved.host}' "
                f"({'IPv6' if resolved.is_ipv6 else 'IPv4'})"
            )
        else:
            # get_local_ip_auto() tries IPv4 first, then IPv6. For explicit control,
            # set SGLANG_HOST_IP env var (use bracketed format for IPv6: [addr])
            local_ip = get_local_ip_auto()
            local_addr = NetworkAddress(local_ip, bootstrap_port)
            bootstrap_host = local_addr.to_host_port_str().rsplit(":", 1)[0]
            logging.info(
                f"Using auto-detected local IP: {local_ip} "
                f"({'IPv6' if local_addr.is_ipv6 else 'IPv4'})"
            )
        return bootstrap_host, bootstrap_port
    except Exception as e:
        logging.warning(f"Failed to get bootstrap info: {e}")
        return None, None


def _parse_hicache_storage_extra_config(
    raw_extra_config: Optional[Any],
) -> dict[str, Any]:
    if raw_extra_config is None:
        return {}

    if isinstance(raw_extra_config, dict):
        return dict(raw_extra_config)

    if isinstance(raw_extra_config, str):
        raw_extra_config = raw_extra_config.strip()
        if not raw_extra_config:
            return {}
        try:
            parsed = json.loads(raw_extra_config)
        except Exception as e:
            logging.warning(
                f"Failed to parse hicache_storage_backend_extra_config JSON: {e}"
            )
            return {}

        if isinstance(parsed, dict):
            return parsed

        logging.warning(
            "hicache_storage_backend_extra_config JSON was not an object; ignoring it."
        )
        return {}

    logging.warning(
        "Unsupported hicache_storage_backend_extra_config type %s; ignoring it.",
        type(raw_extra_config).__name__,
    )
    return {}


def _get_mooncake_runtime_data(server_args: ServerArgs) -> Optional[dict[str, Any]]:
    if getattr(server_args, "hicache_storage_backend", None) != "mooncake":
        return None

    extra_config = _parse_hicache_storage_extra_config(
        getattr(server_args, "hicache_storage_backend_extra_config", None)
    )

    try:
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        if extra_config and (
            extra_config.get("master_server_address") is not None
            or extra_config.get("client_server_address") is not None
        ):
            mooncake_config = MooncakeStoreConfig.load_from_extra_config(extra_config)
        elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            mooncake_config = MooncakeStoreConfig.from_file()
        else:
            mooncake_config = MooncakeStoreConfig.load_from_env()
    except Exception as e:
        logging.warning(f"Failed to resolve Mooncake config for runtime metadata: {e}")
        return None

    tp_size = int(getattr(server_args, "tp_size", 1) or 1)
    pp_size = int(getattr(server_args, "pp_size", 1) or 1)

    try:
        is_mla_model = bool(server_args.use_mla_backend())
    except Exception as e:
        logging.warning(f"Failed to determine whether model uses MLA backend: {e}")
        is_mla_model = False

    spec_algorithm = SpeculativeAlgorithm.from_string(
        getattr(server_args, "speculative_algorithm", None)
    )
    is_eagle = bool(spec_algorithm.is_eagle())

    tp_lcm_size = extra_config.get("tp_lcm_size")
    try:
        tp_lcm_size = int(tp_lcm_size) if tp_lcm_size is not None else None
    except (TypeError, ValueError):
        logging.warning("Ignoring non-integer Mooncake tp_lcm_size=%r", tp_lcm_size)
        tp_lcm_size = None

    should_split_heads = (
        not is_mla_model
        and getattr(server_args, "hicache_mem_layout", None) == "page_head"
        and tp_lcm_size is not None
        and tp_lcm_size > tp_size
        and tp_lcm_size % tp_size == 0
    )

    extra_backend_tag = extra_config.get("extra_backend_tag")
    if not isinstance(extra_backend_tag, str) or not extra_backend_tag:
        extra_backend_tag = None

    master_server_address = getattr(mooncake_config, "master_server_address", None)
    if not isinstance(master_server_address, str) or not master_server_address:
        master_server_address = None

    return {
        "backend": "mooncake",
        "page_size": int(getattr(server_args, "page_size", 1) or 1),
        "tp_size": tp_size,
        "pp_size": pp_size,
        "is_mla_model": is_mla_model,
        "is_eagle": is_eagle,
        "tp_lcm_size": tp_lcm_size,
        "should_split_heads": should_split_heads,
        "extra_backend_tag": extra_backend_tag,
        "master_server_address": master_server_address,
        "master_metrics_port": int(getattr(mooncake_config, "master_metrics_port", 9003)),
    }


async def _get_runtime_config(
    engine: sgl.Engine, server_args: ServerArgs, dynamo_args: DynamoConfig
) -> Optional[ModelRuntimeConfig]:
    """Extract runtime configuration from SGLang engine and args.

    Args:
        engine: The SGLang engine instance.
        server_args: SGLang server configuration.
        dynamo_args: Dynamo-specific configuration.

    Returns:
        ModelRuntimeConfig with extracted values, or None if extraction fails.
    """
    runtime_config = ModelRuntimeConfig()
    # set reasoning parser and tool call parser
    runtime_config.reasoning_parser = dynamo_args.dyn_reasoning_parser
    runtime_config.tool_call_parser = dynamo_args.dyn_tool_call_parser
    runtime_config.exclude_tools_when_tool_choice_none = (
        dynamo_args.exclude_tools_when_tool_choice_none
    )
    # Decode workers don't create the WorkerKvQuery endpoint, so don't advertise local indexer
    is_decode_worker = server_args.disaggregation_mode == "decode"
    runtime_config.enable_local_indexer = (
        dynamo_args.enable_local_indexer and not is_decode_worker
    )

    # Set data_parallel_size for DP attention mode
    # This enables the router to correctly track per-(worker_id, dp_rank) pairs
    dp_size = getattr(server_args, "dp_size", 1) or 1
    runtime_config.data_parallel_size = dp_size
    if dp_size > 1:
        logging.info(f"Registering with data_parallel_size={dp_size}")

    # Set bootstrap endpoint for disaggregated serving (prefill workers)
    bootstrap_host, bootstrap_port = _get_bootstrap_info_for_config(engine)
    if bootstrap_host and bootstrap_port:
        runtime_config.set_disaggregated_endpoint(bootstrap_host, bootstrap_port)
        logging.info(
            f"Publishing disaggregated endpoint to discovery: "
            f"{bootstrap_host}:{bootstrap_port}"
        )
    # In SGLang, these are server_args, not scheduler_info (unlike vLLM)
    # Note: If --max-running-requests is not specified, SGLang uses an internal default
    # undocumented value. The value here will be None if not explicitly set by user.
    max_running_requests = getattr(server_args, "max_running_requests", None)
    if max_running_requests:
        runtime_config.max_num_seqs = max_running_requests

    max_prefill_tokens = getattr(server_args, "max_prefill_tokens", None)
    if max_prefill_tokens:
        runtime_config.max_num_batched_tokens = max_prefill_tokens

    if server_args.speculative_algorithm in ("EAGLE", "NEXTN"):
        runtime_config.enable_eagle = True

    mooncake_runtime_data = _get_mooncake_runtime_data(server_args)
    if mooncake_runtime_data is not None:
        try:
            runtime_config.set_engine_specific(
                SGLANG_HICACHE_MOONCAKE_RUNTIME_KEY,
                json.dumps(mooncake_runtime_data),
            )
            logging.info("Published Mooncake HiCache runtime metadata for router use.")
        except Exception as e:
            logging.warning(
                f"Failed to attach Mooncake HiCache runtime metadata to registration: {e}"
            )

    try:
        # Try to check if the engine has a scheduler attribute with the computed values
        if hasattr(engine, "scheduler_info") and engine.scheduler_info is not None:
            # Get max_total_num_tokens from scheduler_info
            max_total_tokens = engine.scheduler_info.get("max_total_num_tokens")
            if max_total_tokens and hasattr(engine.tokenizer_manager, "server_args"):
                page_size = engine.tokenizer_manager.server_args.page_size
                if page_size:
                    runtime_config.total_kv_blocks = (
                        max_total_tokens + page_size - 1
                    ) // page_size
                    logging.info(
                        f"Got total KV blocks from scheduler: {runtime_config.total_kv_blocks} "
                        f"(max_total_tokens={max_total_tokens}, page_size={page_size})"
                    )

            # When max_prefill_tokens is not explicitly set by the user, fall back
            # to max_total_num_tokens from the scheduler. This ensures the planner
            # always has a prefill load signal for aggregated scaling decisions.
            if not max_prefill_tokens and max_total_tokens:
                runtime_config.max_num_batched_tokens = max_total_tokens
                logging.info(
                    f"max_prefill_tokens not set, using max_total_num_tokens "
                    f"from scheduler as max_num_batched_tokens: {max_total_tokens}"
                )

            return runtime_config

        # If scheduler approach doesn't work, log and return None to indicate we'll skip runtime config
        logging.warning(
            "Could not access runtime config from SGLang engine. "
            "The engine may compute these values internally after initialization. "
            "Proceeding without runtime config - SGLang will use its internal defaults."
        )
        return runtime_config

    except Exception as e:
        logging.warning(f"Failed to get runtime config: {e}. Proceeding without it.")
        return runtime_config


async def register_model_with_readiness_gate(
    engine: sgl.Engine,
    generate_endpoint: Endpoint,
    server_args: ServerArgs,
    dynamo_args: DynamoConfig,
    input_type: ModelInput = ModelInput.Tokens,
    output_type: ModelType = ModelType.Chat | ModelType.Completions,
    readiness_gate: Optional[asyncio.Event] = None,
) -> None:
    """Wrapper function to register LLM with the Dynamo runtime and use optional readiness gate to signal success.

    Args:
        engine: The SGLang engine instance.
        generate_endpoint: The Dynamo endpoint for generation requests.
        server_args: SGLang server configuration.
        dynamo_args: Dynamo-specific configuration.
        input_type: Expected model input type. Defaults to ModelInput.Tokens.
        output_type: Expected model output type. Defaults to ModelType.Chat | ModelType.Completions.
        readiness_gate: Optional event to signal when registration completes.

    Raises:
        RuntimeError: If model registration fails.
    """
    registration_success = await _register_model_with_runtime_config(
        engine,
        generate_endpoint,
        server_args,
        dynamo_args,
        input_type,
        output_type,
    )
    if not registration_success:
        logging.error("Model registration failed; shutting down")
        if engine is not None:
            engine.shutdown()
        raise RuntimeError("Model registration failed")

    if readiness_gate:
        readiness_gate.set()

    logging.info("Model registration succeeded; processing queued requests")


async def register_image_diffusion_model(
    generator: Any,  # DiffGenerator
    endpoint: Endpoint,
    server_args: ServerArgs,
    output_modalities: Optional[List[str]] = None,
    readiness_gate: Optional[asyncio.Event] = None,
) -> None:
    """Register diffusion model with Dynamo runtime.

    Args:
        generator: The SGLang DiffGenerator instance.
        endpoint: The Dynamo endpoint for generation requests.
        server_args: SGLang server configuration.
        output_modalities: Optional list of output modality names to override
            the default ModelType.Images registration.
        readiness_gate: Optional event to signal when registration completes.

    Note:
        Image diffusion models use ModelInput.Text (text prompts) and ModelType.Images
        by default. When output_modalities is provided, the ModelType is derived
        from the given modality names instead.
    """
    model_name = (
        getattr(server_args, "served_model_name", None) or server_args.model_path
    )

    model_type = ModelType.Images
    if output_modalities:
        resolved = get_output_modalities(output_modalities, model_name)
        if resolved is not None:
            model_type = resolved
            logging.info(
                "Using output modalities %s for diffusion model registration",
                output_modalities,
            )
        else:
            logging.warning(
                "No recognized output modalities from %s, defaulting to ModelType.Images",
                output_modalities,
            )

    try:
        await register_model(
            ModelInput.Text,
            model_type,
            endpoint,
            server_args.model_path,
            model_name,
        )
        logging.info(f"Successfully registered diffusion model: {model_name}")
    except Exception as e:
        logging.error(f"Failed to register diffusion model: {e}")
        raise RuntimeError("Image diffusion model registration failed")

    # Signal readiness
    if readiness_gate:
        readiness_gate.set()

    logging.info(f"Image diffusion model ready: {model_name}")


async def register_video_generation_model(
    generator: Any,  # DiffGenerator
    endpoint: Endpoint,
    server_args: ServerArgs,
    readiness_gate: Optional[asyncio.Event] = None,
) -> None:
    """Register video generation model with Dynamo runtime.

    Args:
        generator: The SGLang DiffGenerator instance (used for video generation).
        endpoint: The Dynamo endpoint for generation requests.
        server_args: SGLang server configuration.
        readiness_gate: Optional event to signal when registration completes.

    Note:
        Video generation models use ModelInput.Text (text prompts) and ModelType.Videos.
    """
    model_name = (
        getattr(server_args, "served_model_name", None) or server_args.model_path
    )

    try:
        await register_model(
            ModelInput.Text,
            ModelType.Videos,
            endpoint,
            server_args.model_path,
            model_name,
        )
        logging.info(f"Successfully registered video generation model: {model_name}")
    except Exception as e:
        logging.error(f"Failed to register video generation model: {e}")
        raise RuntimeError("Video generation model registration failed")

    # Signal readiness
    if readiness_gate:
        readiness_gate.set()

    logging.info(f"Video generation model ready: {model_name}")
