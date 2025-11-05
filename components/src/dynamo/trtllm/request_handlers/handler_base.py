# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import AsyncGenerator, Optional, Union

import torch
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi.llm import SamplingParams

from dynamo._core import Context
from dynamo.logits_processing.examples import HelloWorldLogitsProcessor
from dynamo.nixl_connect import Connector
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.engine import TensorRTLLMEngine
from dynamo.trtllm.logits_processing.adapter import create_trtllm_adapters
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.publisher import Publisher
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)
from dynamo.trtllm.constants import DisaggregationMode, DisaggregationStrategy

configure_dynamo_logging()


@dataclass
class RequestHandlerConfig:
    """
    Configuration for the request handler
    """

    component: object
    engine: TensorRTLLMEngine
    default_sampling_params: SamplingParams
    publisher: Publisher
    disaggregation_mode: DisaggregationMode
    disaggregation_strategy: DisaggregationStrategy
    next_client: object
    next_router_client: Optional[object] = None
    encode_client: Optional[object] = None
    multimodal_processor: Optional[
        MultimodalRequestProcessor
    ] = None  # for multimodal support
    connector: Optional[Connector] = None


class HandlerBase:
    """
    Base class for request handlers.
    """

    def __init__(self, config: RequestHandlerConfig):
        self.engine = config.engine
        self.component = config.component
        self.default_sampling_params = config.default_sampling_params
        self.publisher = config.publisher
        self.disaggregation_mode = config.disaggregation_mode
        self.disaggregation_strategy = config.disaggregation_strategy
        self.next_client = config.next_client
        self.next_router_client = config.next_router_client
        self.encode_client = config.encode_client
        self.multimodal_processor = config.multimodal_processor
        self.first_generation = True
        self.connector = config.connector

    def check_error(self, result: dict):
        """
        Check if there is an error in the result.
        """
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return result["finish_reason"] == "error"
        else:
            return (
                result["finish_reason"] == "stop" or result["finish_reason"] == "error"
            )

    async def _handle_cancellation(
        self, generation_result: GenerationResult, context: Context
    ):
        """Background task to handle cancellation by monitoring context state."""
        try:
            # Wait asynchronously for cancellation signal instead of polling
            await context.async_killed_or_stopped()
            # Abort the generation
            generation_result.abort()
            logging.debug(f"Aborted Request ID: {context.id()}")
        except asyncio.CancelledError:
            # Task was cancelled, which is expected when generation completes
            pass

    @asynccontextmanager
    async def _cancellation_monitor(
        self, generation_result: GenerationResult, context: Context
    ) -> AsyncGenerator[asyncio.Task, None]:
        """
        Context manager for monitoring request cancellation.

        Automatically creates a background task to monitor for cancellation and
        cleans it up when the context exits.

        Yields:
            asyncio.Task: The cancellation monitoring task
        """
        cancellation_task = asyncio.create_task(
            self._handle_cancellation(generation_result, context)
        )

        try:
            yield cancellation_task
        finally:
            # Clean up the background cancellation task
            if not cancellation_task.done():
                cancellation_task.cancel()
                try:
                    await cancellation_task
                except asyncio.CancelledError:
                    pass

    async def generate_locally(
        self,
        request: dict,
        context: Context,
        embeddings: Optional[Union[torch.Tensor, dict]] = None,
        ep_disaggregated_params: Optional[DisaggregatedParams] = None,
    ):
        """
        Generate responses based on the disaggregation mode in the request.

        Args:
            request: The request dictionary containing generation parameters
            context: Context object for cancellation handling
            embeddings: Optional tensor or dict containing embeddings for multimodal processing
        """
        logging.debug(f"Request: {request}")

        # Decode the disaggregated params from the request FIRST
        # This must happen before multimodal processing so that ep_disaggregated_params
        # is available for full EPD flow
        disaggregated_params = None

        # Normalize OpenAI request format BEFORE processing
        # This ensures max_tokens is in stop_conditions when we need to save it
        if "stop_conditions" not in request:
            request["stop_conditions"] = {}
        if "max_tokens" in request and "max_tokens" not in request["stop_conditions"]:
            request["stop_conditions"]["max_tokens"] = request.pop("max_tokens")
            logging.info(f"Normalized OpenAI max_tokens to stop_conditions: {request['stop_conditions']['max_tokens']}")
        
        if "sampling_options" not in request:
            request["sampling_options"] = {}
        if "temperature" in request and "temperature" not in request["sampling_options"]:
            request["sampling_options"]["temperature"] = request.pop("temperature")

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Save original max_tokens before modifying for prefill
            # Store original max_tokens so decode worker can restore it
            if "max_tokens" in request["stop_conditions"]:
                request["_original_max_tokens"] = request["stop_conditions"]["max_tokens"]
                logging.info(f"PREFILL: Saved original max_tokens: {request['_original_max_tokens']}")
            else:
                logging.info(f"PREFILL: No max_tokens in request stop_conditions")
            request["stop_conditions"]["max_tokens"] = 1
            if ep_disaggregated_params:
                ep_disaggregated_params.request_type = "context_only"
                disaggregated_params = ep_disaggregated_params
            else:
                disaggregated_params = LlmDisaggregatedParams(request_type="context_only")
            
        if "disaggregated_params" in request:
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                raise ValueError("Cannot provide disaggregated_params in prefill mode")
            if ep_disaggregated_params:
                # ep_disaggregated_params was passed directly as parameter (shouldn't happen in decode)
                disaggregated_params = ep_disaggregated_params
            else:
                # Decode from request dict (normal decode flow)
                disaggregated_params = DisaggregatedParamsCodec.decode(
                    DisaggregatedParams(**request["disaggregated_params"])
                )
                # For full EPD flow, make decoded params available to multimodal processor
                ep_disaggregated_params = disaggregated_params
            disaggregated_params.request_type = "generation_only"

        # Default to text-based input. This will be overwritten if multimodal
        # content is found and processed.
        processed_input = None

        # Check for multimodal request and process it
        # Now ep_disaggregated_params is properly set for both prefill and decode modes
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            # Restore original max_tokens for decode phase
            if "_original_max_tokens" in request:
                if "stop_conditions" not in request:
                    request["stop_conditions"] = {}
                request["stop_conditions"]["max_tokens"] = request["_original_max_tokens"]
                logging.info(f"DECODE: Restored original max_tokens: {request['_original_max_tokens']}")
            else:
                logging.info(f"DECODE: No _original_max_tokens in request. Current max_tokens: {request.get('stop_conditions', {}).get('max_tokens', 'NOT SET')}")
            
            # Decode worker with generation_only mode
            # Pass the same inputs format as prefill
            if "_epd_processed_prompt" in request:
                processed_prompt = request["_epd_processed_prompt"]
                # Use pre-computed token IDs from encoder for consistency
                if "_epd_prompt_token_ids" in request and request["_epd_prompt_token_ids"]:
                    prompt_token_ids = request["_epd_prompt_token_ids"]
                else:
                    # Fallback: tokenize if token IDs not provided
                    prompt_token_ids = self.engine.llm.tokenizer.encode(processed_prompt, add_special_tokens=False)
                
                processed_input = {
                    "prompt": processed_prompt,
                    "prompt_token_ids": prompt_token_ids
                }
            else:
                # Fallback for text-only requests
                processed_input = request.get("token_ids")
        elif self.multimodal_processor:
            # Encode/Prefill worker: Process multimodal content normally
            processed_input = await self.multimodal_processor.process_openai_request(
                request, embeddings, ep_disaggregated_params
            )
        else:
            # text-only flow
            processed_input = request.get("token_ids")

        # Check if there is an error in the publisher error queue
        publishers_error = (
            self.publisher.check_error_queue() if self.publisher else None
        )
        if publishers_error:
            raise publishers_error

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and disaggregated_params is None
        ):
            raise ValueError("Disaggregated params are required for decode mode")

        num_output_tokens_so_far = 0

        sampling_params = copy.deepcopy(self.default_sampling_params)

        # Only process sampling_options if present (may not exist for decode worker)
        if "sampling_options" in request:
            for key, value in request["sampling_options"].items():
                if not value:
                    continue
                if hasattr(sampling_params, key):
                    setattr(sampling_params, key, value)

        # Only process stop_conditions if present
        if "stop_conditions" in request:
            max_tokens = request["stop_conditions"].get("max_tokens")
            if max_tokens:
                sampling_params.max_tokens = max_tokens

            ignore_eos = request["stop_conditions"].get("ignore_eos")
            if ignore_eos:
                sampling_params.ignore_eos = ignore_eos

            min_tokens = request["stop_conditions"].get("min_tokens")
            if min_tokens:
                sampling_params.min_tokens = min_tokens

        # TODO: Instead of True, we should use streaming from the request.
        # However, currently dynamo run does not send streaming in the request.
        streaming = (
            False if self.disaggregation_mode == DisaggregationMode.PREFILL else True
        )

        request_id = request.get("id") or request.get("request_id", "unknown-id")
        model_name = request.get("model", "unknown_model")

        # Optional test-only logits processing (enable with DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR=1)
        if os.getenv("DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR") == "1":
            processors = [HelloWorldLogitsProcessor(self.engine.llm.tokenizer)]
            adapters = create_trtllm_adapters(processors)
            sampling_params.logits_processor = adapters
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            logging.info(f"Generate Called for DECODE mode")
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            logging.info(f"Generate Called for PREFILL mode")
        else:
            logging.info(f"Generate Called for ENCODE mode")
        logging.info(f"Generate locally: processed_input: {processed_input}")
        logging.info(f"Generate locally: sampling_params: {sampling_params}")
        logging.info(f"Generate locally: disaggregated_params: {disaggregated_params}")
        logging.info(f"Generate locally: streaming: {streaming}")
        generation_result = self.engine.llm.generate_async(
            inputs=processed_input,
            sampling_params=sampling_params,
            disaggregated_params=disaggregated_params,
            streaming=streaming,
        )
        logging.info(f"Generate locally: generation_result: {generation_result}")

        # Use the context manager to handle cancellation monitoring
        async with self._cancellation_monitor(generation_result, context):
            async for res in generation_result:
                # TRTLLM engine needs to start generating tokens first before stats
                # can be retrieved.
                if self.first_generation and self.publisher:
                    self.publisher.start()
                    self.first_generation = False

                # Upon completion, send a final chunk with "stop" as the finish reason.
                # This signals to the client that the stream has ended.
                if (
                    res.finished
                    and self.disaggregation_mode != DisaggregationMode.PREFILL
                ):
                    if self.multimodal_processor:
                        final_out = self.multimodal_processor.get_stop_response(
                            request_id, model_name
                        )
                        yield final_out

                # If we are not done generating, but there are no outputs, return an error
                if not res.outputs and not res.finished:
                    yield {"finish_reason": "error", "token_ids": []}
                    break

                output = res.outputs[0]
                # The engine returns all tokens generated so far. We must calculate the new
                # tokens generated in this iteration to create the "delta".
                next_total_toks = len(output.token_ids)
                if self.multimodal_processor:
                    out = self.multimodal_processor.create_response_chunk(
                        output, num_output_tokens_so_far, request_id, model_name
                    )
                else:
                    out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
                if output.finish_reason:
                    out["finish_reason"] = output.finish_reason
                if output.stop_reason:
                    out["stop_reason"] = output.stop_reason
                if self.disaggregation_mode == DisaggregationMode.PREFILL:
                    # Return the disaggregated params only when operating in prefill mode
                    encoded_params = DisaggregatedParamsCodec.encode(output.disaggregated_params)
                    out["disaggregated_params"] = asdict(encoded_params)
                    
                    # Pass the processed prompt and token IDs for decode worker
                    # Use the actual prompt and token IDs from the RequestOutput (res)
                    # which includes all the image placeholder tokens processed by TRTLLM
                    if "_epd_processed_prompt" in request and res.prompt:
                        out["_epd_processed_prompt"] = res.prompt
                    if "_epd_prompt_token_ids" in request and res.prompt_token_ids:
                        out["_epd_prompt_token_ids"] = res.prompt_token_ids
                    
                    # Pass the original max_tokens to decode worker
                    if "_original_max_tokens" in request:
                        out["_original_max_tokens"] = request["_original_max_tokens"]

                if res.finished and not out.get("finish_reason"):
                    out["finish_reason"] = "unknown"
                    logging.warning(
                        "Request finished with no finish reason set - this indicates a possible bug"
                    )

                # Yield the chunk to the client and update the token count for the next iteration.
                yield out
                num_output_tokens_so_far = next_total_toks
