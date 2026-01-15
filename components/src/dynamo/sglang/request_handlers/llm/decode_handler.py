# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

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
        reasoning_parser_type: Optional[str] = None,
        tool_call_parser_type: Optional[str] = None,
    ) -> None:
        """Initialize decode worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Metrics publisher for the worker.
            reasoning_parser_type: Type of reasoning parser (e.g., "deepseek-r1", "qwen3").
            tool_call_parser_type: Type of tool call parser (e.g., "qwen", "llama3").
        """
        super().__init__(
            component,
            engine,
            config,
            publisher,
        )
        self.reasoning_parser_type = reasoning_parser_type
        self.tool_call_parser_type = tool_call_parser_type
        if reasoning_parser_type:
            logging.info(f"Reasoning parser enabled: {reasoning_parser_type}")
        if tool_call_parser_type:
            logging.info(f"Tool call parser enabled: {tool_call_parser_type}")

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

            trace_header = (
                self._get_trace_header(context) if self.enable_trace else None
            )

            decode = await self.engine.async_generate(
                **input_param,
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=bootstrap_info["bootstrap_host"],
                bootstrap_port=bootstrap_info["bootstrap_port"],
                bootstrap_room=bootstrap_info["bootstrap_room"],
                external_trace_header=trace_header,
                rid=trace_id,
            )

            # Use token stream if input was token_ids, otherwise use text stream
            use_token_stream = self.skip_tokenizer_init or "token_ids" in request
            if use_token_stream:
                async for out in self._process_token_stream(decode, context):
                    yield out
            else:
                async for out in self._process_text_stream(decode, context, request):
                    yield out
        else:
            trace_header = (
                self._get_trace_header(context) if self.enable_trace else None
            )

            agg = await self.engine.async_generate(
                **input_param,
                sampling_params=sampling_params,
                stream=True,
                external_trace_header=trace_header,
                rid=trace_id,
            )
            # Use token stream if input was token_ids, otherwise use text stream
            use_token_stream = self.skip_tokenizer_init or "token_ids" in request
            if use_token_stream:
                async for out in self._process_token_stream(agg, context):
                    yield out
            else:
                async for out in self._process_text_stream(agg, context, request):
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

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        import uuid

        return f"call_{uuid.uuid4().hex[:24]}"

    async def _process_text_stream(
        self,
        stream_source: AsyncGenerator[Dict[str, Any], None],
        context: Context,
        request: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process text-based stream output in OpenAI format.

        Args:
            stream_source: Async generator from engine.async_generate.
            context: Context object for cancellation handling.
            request: Optional request dict for reasoning/tool call parser settings.

        Yields:
            OpenAI-formatted chat completion chunk dicts.
        """
        count = 0

        # Initialize reasoning parser if configured and requested
        reasoning_parser = None
        separate_reasoning = (
            request.get("separate_reasoning", True) if request else True
        )
        stream_reasoning = request.get("stream_reasoning", True) if request else True

        if self.reasoning_parser_type and separate_reasoning:
            try:
                from sglang.srt.parser.reasoning_parser import ReasoningParser

                reasoning_parser = ReasoningParser(
                    model_type=self.reasoning_parser_type,
                    stream_reasoning=stream_reasoning,
                )
                logging.debug(
                    f"Initialized streaming reasoning parser: {self.reasoning_parser_type}"
                )
            except Exception as e:
                logging.warning(f"Failed to initialize reasoning parser: {e}")

        # Initialize tool call parser if configured and tools are provided
        tool_call_parser = None
        tools = request.get("tools") if request else None
        tool_choice = request.get("tool_choice", "auto") if request else "auto"
        has_tool_calls = False

        if self.tool_call_parser_type and tools and tool_choice != "none":
            try:
                from sglang.srt.function_call.function_call_parser import (
                    FunctionCallParser,
                )

                tool_call_parser = FunctionCallParser(tools, self.tool_call_parser_type)
                logging.debug(
                    f"Initialized streaming tool call parser: {self.tool_call_parser_type}"
                )
            except Exception as e:
                logging.warning(f"Failed to initialize tool call parser: {e}")

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

                # Apply reasoning parser if enabled
                reasoning_content = None
                content_delta = delta
                if reasoning_parser and delta:
                    try:
                        (
                            reasoning_content,
                            content_delta,
                        ) = reasoning_parser.parse_stream_chunk(delta)
                    except Exception as e:
                        logging.warning(f"Reasoning parser error: {e}")
                        content_delta = delta

                # Apply tool call parser if enabled
                tool_calls_delta = None
                if tool_call_parser and content_delta:
                    try:
                        # parse_stream_chunk returns (normal_text, tool_calls_list)
                        (
                            normal_text,
                            tool_calls_list,
                        ) = tool_call_parser.parse_stream_chunk(content_delta)
                        content_delta = normal_text

                        if tool_calls_list:
                            has_tool_calls = True
                            tool_calls_delta = []
                            for tc in tool_calls_list:
                                # ToolCallItem has: tool_index, name (Optional), parameters (str)
                                tool_call_data = {
                                    "index": tc.tool_index,
                                    "type": "function",
                                }
                                # Add id only for new tool calls (when name is present)
                                if tc.name:
                                    tool_call_data["id"] = self._generate_tool_call_id()
                                    tool_call_data["function"] = {
                                        "name": tc.name,
                                        "arguments": tc.parameters or "",
                                    }
                                else:
                                    # Incremental arguments update
                                    tool_call_data["function"] = {
                                        "arguments": tc.parameters or "",
                                    }
                                tool_calls_delta.append(tool_call_data)
                    except Exception as e:
                        logging.warning(f"Tool call parser error: {e}")

                # Build delta message with optional reasoning_content and tool_calls
                delta_message = {"role": "assistant"}
                if reasoning_content:
                    delta_message["reasoning_content"] = reasoning_content
                if content_delta:
                    delta_message["content"] = content_delta
                if tool_calls_delta:
                    delta_message["tool_calls"] = tool_calls_delta

                # Update finish_reason to "tool_calls" if tool calls were detected
                if finish_reason_type == "stop" and has_tool_calls:
                    finish_reason_type = "tool_calls"

                choice_data = {
                    "index": index,
                    "delta": delta_message,
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
