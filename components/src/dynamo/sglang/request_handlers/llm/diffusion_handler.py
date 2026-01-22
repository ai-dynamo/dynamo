# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, AsyncGenerator, Dict

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler


class DiffusionWorkerHandler(DecodeWorkerHandler):
    """
    Handler for diffusion language model workers.
    """

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
        if (
            not hasattr(engine.tokenizer_manager.server_args, "dllm_algorithm")
            or engine.tokenizer_manager.server_args.dllm_algorithm is None
        ):
            logging.error(
                "SGLang engine does not have dllm_algorithm configured. "
                "Diffusion LM behavior may not be active."
                "Please check the SGLang engine configuration."
            )
        else:
            logging.info(
                f"Diffusion worker initialized with algorithm: "
                f"{engine.tokenizer_manager.server_args.dllm_algorithm}"
            )

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response using diffusion LM.

        Args:
            request: Request dict with input and sampling parameters.
            context: Context object for cancellation handling.

        Yields:
            Response dicts with token_ids or OpenAI-formatted chunks.
        """
        logging.debug(
            f"Starting diffusion generation for request {context.id()}, "
            f"input_tokens={len(request.get('token_ids', []))}"
        )

        # Get input parameters (tokens or text)
        input_param = self._get_input_param(request)

        # Build sampling parameters
        sampling_params = self._build_sampling_params(request)

        # Generate trace info if tracing is enabled
        trace_header = self._get_trace_header(context) if self.enable_trace else None
        trace_id = context.id() if trace_header else None

        async_gen = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,  # Always stream for Dynamo
            external_trace_header=trace_header,
            rid=trace_id,
        )

        # Process stream output (token-based or text-based)
        if self.skip_tokenizer_init:
            async for out in self._process_token_stream(async_gen, context):
                yield out
        else:
            async for out in self._process_text_stream(async_gen, context):
                yield out

    def cleanup(self) -> None:
        super().cleanup()
