# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Diffusion Language Model Worker Handler for SGLang.

This handler enables text generation using diffusion LMs (e.g., LLaDA2.0)
through SGLang's native diffusion algorithm support.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

logger = logging.getLogger(__name__)


class DiffusionWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher = None,
        generate_endpoint=None,
    ) -> None:
        """Initialize diffusion worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: SGLang engine with diffusion algorithm configured.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher.
            generate_endpoint: The endpoint handle for discovery.
        """
        super().__init__(component, engine, config, publisher, generate_endpoint)
        
        # Validate that diffusion algorithm is configured
        if not hasattr(engine.tokenizer_manager.server_args, 'dllm_algorithm') or \
           engine.tokenizer_manager.server_args.dllm_algorithm is None:
            logger.warning(
                "SGLang engine does not have dllm_algorithm configured. "
                "Diffusion LM behavior may not be active."
            )
        else:
            logger.info(
                f"Diffusion worker initialized with algorithm: "
                f"{engine.tokenizer_manager.server_args.dllm_algorithm}"
            )

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # Get input parameters (tokens or text)
        input_param = self._get_input_param(request)
        
        # Build sampling parameters
        sampling_params = self._build_sampling_params(request)
        
        # Generate trace info if tracing is enabled
        trace_header = self._get_trace_header(context) if self.enable_trace else None
        trace_id = context.id() if trace_header else None
        
        logger.debug(
            f"Starting diffusion generation for request {context.id()}, "
            f"input_tokens={len(request.get('token_ids', []))}"
        )
        
        # Call SGLang engine - it handles diffusion internally!
        # From our perspective, this looks identical to autoregressive generation
        async_gen = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,  # Always stream for Dynamo
            external_trace_header=trace_header,
            rid=trace_id,
        )
        
        # Process token stream - reuses existing BaseWorkerHandler method
        # This works for both autoregressive and diffusion models!
        async for out in self._process_token_stream(async_gen, context):
            yield out

    def _build_sampling_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build SGLang sampling parameters from request.
        
        Args:
            request: Request dict with sampling_options and stop_conditions.
            
        Returns:
            Dict of sampling parameters for SGLang engine.
        """
        if self.skip_tokenizer_init:
            # Token-based request format
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            
            sampling_params = {}
            
            # Only add params if they have valid values (not None)
            if "temperature" in sampling_opts and sampling_opts["temperature"] is not None:
                sampling_params["temperature"] = sampling_opts["temperature"]
            if "top_p" in sampling_opts and sampling_opts["top_p"] is not None:
                sampling_params["top_p"] = sampling_opts["top_p"]
            if "top_k" in sampling_opts and sampling_opts["top_k"] is not None:
                sampling_params["top_k"] = sampling_opts["top_k"]
            if "max_tokens" in stop_conditions and stop_conditions["max_tokens"] is not None:
                sampling_params["max_new_tokens"] = stop_conditions["max_tokens"]
            
            # Add stop strings if present
            if "stop_strings" in stop_conditions:
                sampling_params["stop"] = stop_conditions["stop_strings"]
                
            return sampling_params
        else:
            # Text-based request format (SGLang handles tokenization)
            return request.get("sampling_params", {})

    async def _process_token_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process token-based stream output.
        
        With stream_output=True (enforced by Dynamo), SGLang sends disjoint segments
        containing only new tokens since the last output. We pass these through directly.
        
        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.
            
        Yields:
            Dict with token_ids and optional finish_reason.
        """
        # Use Future pattern for request ID - will be set when first response arrives
        request_id_future = asyncio.Future()
        async with self._cancellation_monitor(request_id_future, context):
            async for res in stream_source:
                # Extract SGLang request ID from the first response and set the future
                if not request_id_future.done():
                    meta_info = res.get("meta_info", {})
                    sglang_request_id = meta_info.get("id")
                    if sglang_request_id:
                        request_id_future.set_result(sglang_request_id)
                        logger.debug(f"New SGLang Request ID: {sglang_request_id}")

                # Check cancellation before yielding
                out = {}
                finish_reason = res["meta_info"]["finish_reason"]
                if finish_reason:
                    out["finish_reason"] = finish_reason["type"]

                # With stream_output=True, output_ids contains only new tokens (disjoint)
                output_ids = res.get("output_ids", [])
                # If request is not finished yet, but there are no outputs, return an error.
                if not output_ids and not finish_reason:
                    if not context.is_stopped():
                        yield {"finish_reason": "error", "token_ids": []}
                    break

                # Pass through disjoint token segments directly
                out["token_ids"] = output_ids
                if finish_reason:
                    input_tokens = res["meta_info"]["prompt_tokens"]
                    completion_tokens = res["meta_info"]["completion_tokens"]
                    cached_tokens = res["meta_info"]["cached_tokens"]
                    prefill_prompt_tokens_details = None
                    if cached_tokens is not None and cached_tokens > 0:
                        prefill_prompt_tokens_details = {"cached_tokens": cached_tokens}
                    out["completion_usage"] = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": input_tokens + completion_tokens,
                        "prompt_tokens_details": prefill_prompt_tokens_details,
                    }
                if not context.is_stopped():
                    yield out

    def cleanup(self) -> None:
        """Cleanup resources."""
        # No special cleanup needed for diffusion worker
        pass
