# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config, DisaggregationMode
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
    ) -> None:
        """Initialize decode worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Metrics publisher for the worker.
        """
        super().__init__(
            component,
            engine,
            config,
            publisher,
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
            output_options = request.get("output_options", {})

            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
            }

            # Handle logprobs from output_options
            logprobs_value = output_options.get("logprobs")
            if logprobs_value is not None and logprobs_value != "":
                try:
                    parsed_logprobs = int(logprobs_value)
                    if parsed_logprobs < 0:
                        logging.warning(
                            f"Invalid logprobs value: {logprobs_value} (must be non-negative), ignoring"
                        )
                    else:
                        param_mapping["return_logprob"] = True
                        param_mapping["top_logprobs_num"] = parsed_logprobs
                except (ValueError, TypeError):
                    logging.warning(
                        f"Invalid logprobs value: {logprobs_value} (must be integer), ignoring"
                    )
        else:
            # OpenAI request format
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "max_new_tokens": request.get("max_tokens"),
            }

            # Handle logprobs from OpenAI format
            logprobs = request.get("logprobs")
            top_logprobs = request.get("top_logprobs")
            if logprobs:
                param_mapping["return_logprob"] = True
                if top_logprobs is not None:
                    param_mapping["top_logprobs_num"] = top_logprobs

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

        # Extract logprobs params (they go to async_generate, not SamplingParams)
        return_logprob = sampling_params.pop("return_logprob", False)
        top_logprobs_num = sampling_params.pop("top_logprobs_num", None)

        if self.serving_mode == DisaggregationMode.DECODE:
            # Check if bootstrap_info is pre-computed in the request (from frontend)
            bootstrap_info = request.get("bootstrap_info")

            if not bootstrap_info:
                raise RuntimeError(
                    "bootstrap_info is required for disaggregated decode but was not provided"
                )

            logging.debug(
                f"Using bootstrap_info: "
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
                return_logprob=return_logprob,
                top_logprobs_num=top_logprobs_num,
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
                return_logprob=return_logprob,
                top_logprobs_num=top_logprobs_num,
                rid=trace_id,
            )
            if self.skip_tokenizer_init:
                async for out in self._process_token_stream(agg, context):
                    yield out
            else:
                async for out in self._process_text_stream(agg, context):
                    yield out

    @staticmethod
    def _extract_logprobs(
        res: Dict[str, Any], num_output_tokens_so_far: int
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        """
        Extract logprobs from SGLang response for new tokens.

        Args:
            res: SGLang response dict
            num_output_tokens_so_far: Number of tokens already processed

        Returns:
            Tuple of (log_probs, top_logprobs) in Dynamo's expected format:
            - log_probs: List of log probabilities for each new token
            - top_logprobs: List of top logprobs dicts for each new token
        """
        meta_info = res.get("meta_info", {})

        # SGLang uses "output_token_logprobs" for selected token logprobs
        # Format: [(logprob, token_id, decoded_text), ...] - one tuple per token
        output_token_logprobs = meta_info.get("output_token_logprobs")

        # SGLang uses "output_top_logprobs" for top-k alternatives
        # Format: [[(logprob, token_id, text), ...], ...] - list of lists
        output_top_logprobs = meta_info.get("output_top_logprobs")

        if not output_token_logprobs:
            return None, None

        # Get logprobs for new tokens only
        new_token_logprobs = output_token_logprobs[num_output_tokens_so_far:]
        if not new_token_logprobs:
            return None, None

        log_probs = []
        top_logprobs = []

        # Get top logprobs slice if available
        new_top_logprobs = None
        if output_top_logprobs:
            new_top_logprobs = output_top_logprobs[num_output_tokens_so_far:]

        # Extract logprobs for each token, maintaining 1:1 alignment
        for idx, token_data in enumerate(new_token_logprobs):
            # Skip if token_data is None or logprob_val is None
            if token_data is None:
                continue
            # SGLang format: (logprob, token_id, decoded_text)
            logprob_val = token_data[0]
            if logprob_val is None:
                continue

            log_probs.append(float(logprob_val))

            # Extract corresponding top logprobs for this token position
            if new_top_logprobs and idx < len(new_top_logprobs):
                token_top_list = new_top_logprobs[idx]
                if not token_top_list:
                    top_logprobs.append([])
                else:
                    # Filter out None entries and sort by logprob descending
                    # SGLang doesn't guarantee order, so we sort to assign proper ranks
                    valid_entries = [
                        alt_data
                        for alt_data in token_top_list
                        if alt_data is not None and alt_data[0] is not None
                    ]
                    # Sort by logprob descending (highest probability first)
                    valid_entries.sort(key=lambda x: x[0], reverse=True)

                    token_top_logprobs = []
                    for rank, alt_data in enumerate(valid_entries):
                        # SGLang format: (logprob, token_id, decoded_text)
                        alt_logprob_val = alt_data[0]
                        token_id = alt_data[1]
                        decoded_text = alt_data[2] if len(alt_data) > 2 else None
                        token_top_logprobs.append(
                            {
                                "rank": rank,
                                "token_id": token_id,
                                "token": decoded_text,
                                "logprob": float(alt_logprob_val),
                            }
                        )
                    top_logprobs.append(token_top_logprobs)

        return log_probs if log_probs else None, top_logprobs if top_logprobs else None

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

                # Extract logprobs for new tokens
                log_probs, top_logprobs = self._extract_logprobs(
                    res, num_output_tokens_so_far
                )
                if log_probs is not None:
                    out["log_probs"] = log_probs
                if top_logprobs is not None:
                    out["top_logprobs"] = top_logprobs

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
