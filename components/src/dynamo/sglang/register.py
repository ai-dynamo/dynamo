# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import socket
from typing import Any, Optional

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_auto

from dynamo._core import Endpoint
from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_llm
from dynamo.sglang.args import DynamoArgs


async def _register_llm_with_runtime_config(
    engine: sgl.Engine,
    endpoint: Endpoint,
    server_args: ServerArgs,
    dynamo_args: DynamoArgs,
    input_type: Optional[ModelInput] = ModelInput.Tokens,
    output_type: Optional[ModelType] = ModelType.Chat | ModelType.Completions,
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
    input_type = input_type

    if not server_args.skip_tokenizer_init:
        logging.warning(
            "The skip-tokenizer-init flag was not set. Using the sglang tokenizer/detokenizer instead. The dynamo tokenizer/detokenizer will not be used and only v1/chat/completions will be available"
        )
        input_type = ModelInput.Text
        # Only override output_type for chat models, not for embeddings
        if output_type != ModelType.Embedding:
            output_type = ModelType.Chat

    try:
        await register_llm(
            input_type,
            output_type,
            endpoint,
            server_args.model_path,
            server_args.served_model_name,
            kv_cache_block_size=server_args.page_size,
            migration_limit=dynamo_args.migration_limit,
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
            # IPv6-ready host extraction and resolution:
            # 1) Extract raw host from "host:port" or "[IPv6]:port"/"[IPv6]".
            # 2) Resolve via AF_UNSPEC to accept A/AAAA and literals.
            # 3) Bracket-wrap IPv6 for safe "{host}:{port}" URL formatting.
            addr = inner_tm.server_args.dist_init_addr.strip()
            if addr.startswith("["):
                end = addr.find("]")
                host_core = addr[1:end] if end != -1 else addr.strip("[]")
            else:
                # Only treat single ':' with numeric suffix as host:port; otherwise it's an IPv6/FQDN host.
                if addr.count(":") == 1:
                    host_candidate, maybe_port = addr.rsplit(":", 1)
                    host_core = host_candidate if maybe_port.isdigit() else addr
                else:
                    host_core = addr
            try:
                infos = socket.getaddrinfo(
                    host_core,
                    None,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                )
                resolved = infos[0][4][0]  # let OS policy pick v4/v6
                bootstrap_host = resolved
                addr_family = infos[0][0]
                logging.info(
                    f"Resolved bootstrap host '{host_core}' -> '{resolved}' "
                    f"({'IPv6' if addr_family == socket.AF_INET6 else 'IPv4'})"
                )
            except socket.gaierror as e:
                # Fallback: keep literal/FQDN as-is (still wrap IPv6 below)
                bootstrap_host = host_core
                logging.warning(
                    f"Failed to resolve bootstrap host '{host_core}': {e}, using as-is"
                )
        else:
            # get_local_ip_auto() tries IPv4 first, then IPv6. For explicit control,
            # set SGLANG_HOST_IP env var (use bracketed format for IPv6: [addr])
            bootstrap_host = get_local_ip_auto()
            is_ipv6 = ":" in bootstrap_host
            logging.info(
                f"Using auto-detected local IP: {bootstrap_host} "
                f"({'IPv6' if is_ipv6 else 'IPv4'})"
            )

        # Wrap IPv6 literal with brackets so f"{host}:{port}" stays valid.
        if ":" in bootstrap_host and not bootstrap_host.startswith("["):
            bootstrap_host = f"[{bootstrap_host}]"
            logging.info(f"Wrapped IPv6 address with brackets: {bootstrap_host}")

        return bootstrap_host, bootstrap_port
    except Exception as e:
        logging.warning(f"Failed to get bootstrap info: {e}")
        return None, None


async def _get_runtime_config(
    engine: sgl.Engine, server_args: ServerArgs, dynamo_args: DynamoArgs
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
    runtime_config.reasoning_parser = dynamo_args.reasoning_parser
    runtime_config.tool_call_parser = dynamo_args.tool_call_parser
    runtime_config.enable_local_indexer = dynamo_args.enable_local_indexer

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

    try:
        # Try to check if the engine has a scheduler attribute with the computed values
        if hasattr(engine, "scheduler_info") and engine.scheduler_info is not None:
            # Get max_total_num_tokens from scheduler_info
            if "max_total_num_tokens" in engine.scheduler_info:
                max_total_tokens = engine.scheduler_info["max_total_num_tokens"]
                if max_total_tokens and hasattr(
                    engine.tokenizer_manager, "server_args"
                ):
                    page_size = engine.tokenizer_manager.server_args.page_size
                    if page_size:
                        runtime_config.total_kv_blocks = (
                            max_total_tokens + page_size - 1
                        ) // page_size
                        logging.info(
                            f"Got total KV blocks from scheduler: {runtime_config.total_kv_blocks} "
                            f"(max_total_tokens={max_total_tokens}, page_size={page_size})"
                        )

            # Note: max_running_requests and max_prefill_tokens are NOT available in scheduler_info.
            # SGLang separates configuration (server_args) from runtime stats (scheduler_info).
            # In contrast, vLLM exposes both config and runtime values through engine config.
            # These are config parameters, so they must be retrieved from server_args only.

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


async def register_llm_with_readiness_gate(
    engine: sgl.Engine,
    generate_endpoint: Endpoint,
    server_args: ServerArgs,
    dynamo_args: DynamoArgs,
    input_type: Optional[ModelInput] = ModelInput.Tokens,
    output_type: Optional[ModelType] = ModelType.Chat | ModelType.Completions,
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
    registration_success = await _register_llm_with_runtime_config(
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
    readiness_gate: Optional[asyncio.Event] = None,
) -> None:
    """Register diffusion model with Dynamo runtime.

    Args:
        generator: The SGLang DiffGenerator instance.
        endpoint: The Dynamo endpoint for generation requests.
        server_args: SGLang server configuration.
        readiness_gate: Optional event to signal when registration completes.

    Note:
        Image diffusion models use ModelInput.Text (text prompts) and ModelType.Images.
    """
    # Use model_path as the model name (diffusion workers don't have served_model_name)
    model_name = server_args.model_path

    try:
        await register_llm(
            ModelInput.Text,
            ModelType.Images,
            endpoint,
            model_name,
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
