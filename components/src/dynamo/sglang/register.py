# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Optional

import sglang as sgl
from sglang.srt.server_args import ServerArgs

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
