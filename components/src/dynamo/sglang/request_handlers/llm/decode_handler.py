# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

import sglang as sgl

from dynamo._core import Client, Component, Context
from dynamo.sglang.args import Config, DisaggregationMode
from dynamo.sglang.protocol import DisaggPreprocessedRequest
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class DecodeWorkerHandler(BaseWorkerHandler):
    """Handler for decode workers in both aggregated and disaggregated serving modes."""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        prefill_client: Optional[Client] = None,
    ) -> None:
        """Initialize decode worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Metrics publisher for the worker.
            prefill_client: Optional client for prefill worker in disaggregated mode.
        """
        super().__init__(
            component,
            engine,
            config,
            publisher,
            prefill_client,
        )
        if self.serving_mode == DisaggregationMode.DECODE:
            logging.info(
                "Decode worker handler initialized (disaggregated decode mode)"
            )
        else:
            logging.info("Decode worker handler initialized (aggregated mode)")

    def cleanup(self) -> None:
        """Shutdown the engine and cleanup resources."""
        self.engine.shutdown()
        logging.info("Engine shutdown")
        super().cleanup()

    def _build_sampling_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build sampling params from request format.

        Args:
            request: Request dict in either token-based or OpenAI format.

        Returns:
            Dict of sampling parameters for SGLang engine.
        """
        if self.skip_tokenizer_init:
            # Token-based request format
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})

            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
            }
        else:
            # OpenAI request format
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "max_new_tokens": request.get("max_tokens"),
            }

        return {k: v for k, v in param_mapping.items() if v is not None}

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response in aggregated or disaggregated mode.

        Args:
            request: Request dict with input and sampling parameters.
            context: Context object for cancellation handling.

        Yields:
            Response dicts with token_ids or OpenAI-formatted chunks.

        Raises:
            RuntimeError: If no bootstrap info received from prefill worker.
        """
        logging.debug(f"New Request ID: {context.id()}")
        trace_id = context.trace_id
        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)

        if self.serving_mode == DisaggregationMode.DECODE:
            # Check if bootstrap_info is pre-computed in the request (from frontend with --router-mode kv)
            bootstrap_info = request.get("bootstrap_info")

            if not bootstrap_info:
                # Fallback: fetch bootstrap_info from prefill worker via round-robin routing
                if self.prefill_client is None:
                    raise RuntimeError(
                        "bootstrap_info is required for disaggregated decode but was not provided, "
                        "and no prefill_client is available for fallback."
                    )

                logging.debug(
                    "No bootstrap_info in request, fetching from prefill worker"
                )
                prefill_stream = await self.prefill_client.generate(
                    DisaggPreprocessedRequest(
                        request=request,
                        sampling_params=sampling_params,
                    ).model_dump(),
                    context=context,
                )

                prefill_response = None
                async for info in prefill_stream:
                    prefill_response = info.data()
                    break

                if not prefill_response:
                    raise RuntimeError("No response received from prefill worker")

                # Extract bootstrap_info from disaggregated_params (PrefillWorkerHandler format)
                bootstrap_info = prefill_response.get("disaggregated_params")
                if not bootstrap_info:
                    raise RuntimeError(
                        "No bootstrap info (disaggregated_params) received from prefill worker"
                    )

                logging.debug(
                    f"Received bootstrap_info from prefill worker: "
                    f"host={bootstrap_info['bootstrap_host']}, "
                    f"port={bootstrap_info['bootstrap_port']}, "
                    f"room={bootstrap_info['bootstrap_room']}"
                )
            else:
                logging.debug(
                    f"Using pre-computed bootstrap_info: "
                    f"host={bootstrap_info['bootstrap_host']}, "
                    f"port={bootstrap_info['bootstrap_port']}, "
                    f"room={bootstrap_info['bootstrap_room']}"
                )

            if self.enable_trace:
                self._propagate_trace_context_to_sglang(
                    context, bootstrap_info["bootstrap_room"]
                )

            decode = await self.engine.async_generate(
                **input_param,
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=bootstrap_info["bootstrap_host"],
                bootstrap_port=bootstrap_info["bootstrap_port"],
                bootstrap_room=bootstrap_info["bootstrap_room"],
                rid=trace_id,
            )

            if self.skip_tokenizer_init:
                async for out in self._process_token_stream(decode, context):
                    yield out
            else:
                async for out in self._process_text_stream(decode, context):
                    yield out
        else:
            if self.enable_trace:
                self._propagate_trace_context_to_sglang(context)

            agg = await self.engine.async_generate(
                **input_param,
                sampling_params=sampling_params,
                stream=True,
                rid=trace_id,
            )
            if self.skip_tokenizer_init:
                async for out in self._process_token_stream(agg, context):
                    yield out
            else:
                async for out in self._process_text_stream(agg, context):
                    yield out

    async def _process_token_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process token-based stream output.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            Dict with token_ids and optional finish_reason.
        """
        num_output_tokens_so_far = 0

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
                        logging.debug(f"New SGLang Request ID: {sglang_request_id}")

                # Check cancellation before yielding to allow proper cleanup.
                # This lets SGLang proceed to the second token generation, which will
                # async context switch and allow the abort monitor to signal cancellation.
                # The loop should exit by itself when context.is_stopped() returns True.
                out = {}
                finish_reason = res["meta_info"]["finish_reason"]
                if finish_reason:
                    out["finish_reason"] = finish_reason["type"]

                output_ids = res.get("output_ids", [])
                # If request is not finished yet, but there are no outputs, return an error.
                if not output_ids and not finish_reason:
                    if not context.is_stopped():
                        yield {"finish_reason": "error", "token_ids": []}
                    break

                next_total_toks = len(output_ids)
                out["token_ids"] = output_ids[num_output_tokens_so_far:]
                num_output_tokens_so_far = next_total_toks
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

    async def _process_text_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process text-based stream output in OpenAI format.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.

        Yields:
            OpenAI-formatted chat completion chunk dicts.
        """
        count = 0

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
                        logging.debug(f"New SGLang Request ID: {sglang_request_id}")

                # Check cancellation before yielding to allow proper cleanup.
                # This lets SGLang proceed to the second token generation, which will
                # async context switch and allow the abort monitor to signal cancellation.
                # The loop should exit by itself when context.is_stopped() returns True.

                index = res.get("index", 0)
                text = res.get("text", "")

                finish_reason = res["meta_info"]["finish_reason"]
                finish_reason_type = finish_reason["type"] if finish_reason else None
                next_count = len(text)
                delta = text[count:]

                choice_data = {
                    "index": index,
                    "delta": {"role": "assistant", "content": delta},
                    "finish_reason": finish_reason_type,
                }

                response = {
                    "id": res["meta_info"]["id"],
                    "created": int(time.time()),
                    "choices": [choice_data],
                    "model": self.config.server_args.served_model_name,
                    "object": "chat.completion.chunk",
                }
                if not context.is_stopped():
                    yield response
                count = next_count
