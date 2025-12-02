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
from typing import Any, AsyncGenerator, Optional, Union

import torch
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi.llm import SamplingParams

from dynamo._core import Context
from dynamo.logits_processing.examples import HelloWorldLogitsProcessor
from dynamo.nixl_connect import Connector
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.engine import TensorRTLLMEngine
from dynamo.trtllm.logits_processing.adapter import create_trtllm_adapters
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.publisher import Publisher
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)

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
    encode_client: Optional[object] = None
    multimodal_processor: Optional[
        MultimodalRequestProcessor
    ] = None  # for multimodal support
    connector: Optional[Connector] = None
    runtime: Optional[
        DistributedRuntime
    ] = None  # DistributedRuntime reference for graceful shutdown
    metrics_collector: Optional[Any] = None  # TensorRT-LLM MetricsCollector
    kv_block_size: int = 32


class HandlerBase:
    """
    Base class for request handlers.
    """

    def __init__(self, config: RequestHandlerConfig):
        self.engine = config.engine
        self.component = config.component
        self.default_sampling_params = config.default_sampling_params
        self.publisher = config.publisher
        self.metrics_collector = config.metrics_collector
        self.disaggregation_mode = config.disaggregation_mode
        self.encode_client = config.encode_client
        self.multimodal_processor = config.multimodal_processor
        self.first_generation = True
        self.connector = config.connector
        # Store runtime reference for graceful shutdown
        self.runtime = config.runtime
        self.kv_block_size: int = config.kv_block_size

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

    def _decode_disaggregated_params_from_prefill(
        self, prefill_result: dict
    ) -> tuple[Any, dict]:
        """
        Extract and decode disaggregated params from prefill_result.
        
        Args:
            prefill_result: Result from prefill worker containing encoded disaggregated params
            
        Returns:
            Tuple of (disaggregated_params, epd_metadata) where:
            - disaggregated_params: Decoded LlmDisaggregatedParams object
            - epd_metadata: Dictionary containing EPD-specific metadata (_epd_processed_prompt, etc.)
        """
        params_dict = prefill_result["disaggregated_params"]
        
        # Extract EPD metadata that was packed by prefill worker
        epd_metadata = {}
        if "_epd_metadata" in params_dict:
            epd_metadata = params_dict.pop("_epd_metadata")
            logging.debug(
                f"DECODE: Extracted _epd_metadata with {len(epd_metadata)} fields"
            )
        
        # Decode the disaggregated params
        disaggregated_params = DisaggregatedParamsCodec.decode(
            DisaggregatedParams(**params_dict)
        )
        # Set to generation_only mode for decode phase
        disaggregated_params.request_type = "generation_only"
        
        # In generation-only mode, multimodal embeddings are already processed and in KV cache
        # Remove multimodal_embedding_handles to avoid TRT-LLM validation error
        if (
            hasattr(disaggregated_params, "multimodal_embedding_handles")
            and disaggregated_params.multimodal_embedding_handles
        ):
            disaggregated_params.multimodal_embedding_handles = None
        
        logging.debug("DECODE: Set request_type to generation_only")
        
        return disaggregated_params, epd_metadata

    async def _prepare_decode_input(
        self,
        request: dict,
        epd_metadata: dict,
        prefill_result: Optional[dict],
        embeddings: Any,
        ep_disaggregated_params: Any,
    ) -> Optional[Any]:
        """
        Prepare input for DECODE mode processing.
        
        Handles EPD flow (with encoder) by extracting prompt and token IDs,
        or falls back to multimodal processor for other flows.
        
        Args:
            request: The request dictionary
            epd_metadata: EPD metadata extracted from prefill result
            prefill_result: Result from prefill worker
            embeddings: Multimodal embeddings (if any)
            ep_disaggregated_params: Disaggregated params from encoder/prefill
            
        Returns:
            Processed input ready for the engine, or None if not available
        """
        # Decode worker with generation_only mode
        # Pass the same inputs format as prefill
        # Check epd_metadata (packed by prefill), then prefill_result, then direct request
        epd_prompt = epd_metadata.get("_epd_processed_prompt")
        epd_token_ids = epd_metadata.get("_epd_prompt_token_ids")

        if epd_prompt:
            # In EPD generation-only mode (decode), pass the SAME input format as prefill
            # This matches TRT-LLM's test: llm_decode.generate(inputs, disaggregated_params=...)
            # The inputs dict provides prompt structure, disaggregated_params provide multimodal embeddings
            if epd_token_ids:
                prompt_token_ids = epd_token_ids

            processed_input = {
                "prompt": epd_prompt,
                "prompt_token_ids": prompt_token_ids,
            }

            # Remove ALL multimodal data from request to avoid TRT-LLM validation error
            # In generation-only mode, ALL multimodal data must be in disaggregated_params only
            mm_keys_to_remove = ["multi_modal_data", "image_data", "mm_data"]
            for key in mm_keys_to_remove:
                if key in request:
                    removed_data = request.pop(key)
                    logging.debug(
                        f"DECODE: Removed {key} from request (already in disaggregated_params)"
                    )            
            return processed_input
        elif self.multimodal_processor:
            # Encode/Prefill worker: Process multimodal content normally
            # In EPD flow, multimodal_processor should be called in PREFILL/ENCODE modes only
            # DECODE mode should skip this and use EPD metadata from prefill
            processed_input = (
                await self.multimodal_processor.process_openai_request(
                    request, embeddings, ep_disaggregated_params
                )
            )
            return processed_input
        else:
            logging.debug(
                "DECODE: No multimodal_processor found, using request token_ids"
            )
            return None

    def _encode_and_pack_disaggregated_params(
        self,
        output: GenerationResult,
        disaggregated_params: Any,
        request: dict,
        res: Any,
    ) -> Optional[dict]:
        """
        Encode and pack disaggregated params for PREFILL mode response.
        
        Handles:
        - Choosing between output and input disaggregated params
        - Preserving multimodal_embedding_handles in EPD flow
        - Encoding params for transmission
        - Packing EPD metadata (_epd_processed_prompt, _epd_prompt_token_ids)
        
        Args:
            output: GenerationResult from the engine
            disaggregated_params: Input disaggregated params
            request: Original request dict
            res: RequestOutput object with prompt and prompt_token_ids attributes
            
        Returns:
            Dictionary with encoded disaggregated params, or None if encoding failed
        """
        # In EPD flow, output.disaggregated_params might be None, use the input params
        params_to_encode = (
            output.disaggregated_params
            if output.disaggregated_params is not None
            else disaggregated_params
        )

        # In EPD flow, manually preserve multimodal_embedding_handles from input
        # because TRT-LLM engine may not propagate them through prefill
        if params_to_encode is not None and disaggregated_params is not None:
            input_handles = getattr(
                disaggregated_params,
                "multimodal_embedding_handles",
                None,
            )
            output_handles = getattr(
                params_to_encode, "multimodal_embedding_handles", None
            )

            if input_handles is not None and output_handles is None:
                params_to_encode.multimodal_embedding_handles = input_handles
                # Also preserve hashes if they exist
                input_hashes = getattr(
                    disaggregated_params, "multimodal_hashes", None
                )
                if input_hashes is not None:
                    params_to_encode.multimodal_hashes = input_hashes

        encoded_params = DisaggregatedParamsCodec.encode(params_to_encode)

        if encoded_params is None:
            logging.error(
                "PREFILL: encoded_params is None - decode worker will fail!"
            )
            return None

        logging.debug("PREFILL: Successfully encoded disaggregated params")
        params_dict = asdict(encoded_params)

        # Pack EPD-specific fields into disaggregated_params to ensure they're forwarded
        # The frontend only forwards disaggregated_params and prompt_tokens_details from prefill response
        # Note: max_tokens is already handled by Rust frontend's PrefillRouter
        epd_metadata = {}

        if "_epd_processed_prompt" in request and res.prompt:
            epd_metadata["_epd_processed_prompt"] = res.prompt

        if "_epd_prompt_token_ids" in request and res.prompt_token_ids:
            epd_metadata["_epd_prompt_token_ids"] = res.prompt_token_ids

        # Add EPD metadata to the disaggregated_params dict
        if epd_metadata:
            params_dict["_epd_metadata"] = epd_metadata

        return params_dict

    async def _initiate_shutdown(self, error: Exception):
        """Initiate graceful shutdown after fatal error"""
        logging.warning(f"Initiating graceful shutdown due to: {error}")

        try:
            if self.runtime:
                logging.info("Shutting down Dynamo runtime...")
                self.runtime.shutdown()

            if self.engine:
                logging.info("Shutting down TensorRT-LLM engine...")
                await self.engine.cleanup()
        except Exception as cleanup_error:
            logging.error(f"Error during graceful shutdown: {cleanup_error}")
        finally:
            logging.critical("Forcing process exit for restart")
            os._exit(1)

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

        # Normalize OpenAI request format
        # Note: The Rust frontend's PrefillRouter already handles max_tokens:
        #   - Saves original max_tokens before prefill
        #   - Sends max_tokens=1 to prefill worker
        #   - Restores original max_tokens for decode worker
        # So we don't need to save/modify max_tokens here!
        if "stop_conditions" not in request:
            request["stop_conditions"] = {}
        if "max_tokens" in request and "max_tokens" not in request["stop_conditions"]:
            request["stop_conditions"]["max_tokens"] = request.pop("max_tokens")

        if "sampling_options" not in request:
            request["sampling_options"] = {}
        if (
            "temperature" in request
            and "temperature" not in request["sampling_options"]
        ):
            request["sampling_options"]["temperature"] = request.pop("temperature")

        # Setup disaggregated_params for PREFILL mode
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if ep_disaggregated_params:
                ep_disaggregated_params.request_type = "context_only"
                disaggregated_params = ep_disaggregated_params
            else:
                disaggregated_params = LlmDisaggregatedParams(
                    request_type="context_only"
                )
        # Check for disaggregated_params in prefill_result (new frontend routing)
        # or directly in request (legacy/direct routing)
        prefill_result = request.get("prefill_result")
        epd_metadata = {}

        if prefill_result and "disaggregated_params" in prefill_result:
            # Decode from prefill_result (new frontend disaggregated routing)
            disaggregated_params, epd_metadata = (
                self._decode_disaggregated_params_from_prefill(prefill_result)
            )
            # For full EPD flow, make decoded params available to multimodal processor
            ep_disaggregated_params = disaggregated_params

        # Default to text-based input. This will be overwritten if multimodal
        # content is found and processed.
        processed_input = None
        # Now ep_disaggregated_params is properly set for both prefill and decode modes
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            processed_input = await self._prepare_decode_input(
                request, epd_metadata, prefill_result, embeddings, ep_disaggregated_params
            )
        else:
            # PREFILL/ENCODE mode
            if self.multimodal_processor:
                # Process multimodal content in PREFILL mode
                processed_input = (
                    await self.multimodal_processor.process_openai_request(
                        request, embeddings, ep_disaggregated_params
                    )
                )
                if processed_input:
                    logging.debug(
                        "PREFILL: Multimodal content processed by multimodal_processor"
                    )
                else:
                    # Fallback to text-only if no multimodal content
                    processed_input = request.get("token_ids")
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
            logging.error("DECODE: disaggregated_params is None but required!")
            logging.error(f"DECODE: Request keys: {list(request.keys())}")
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

        # Optional test-only logits processing (enable with DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR=1)
        if os.getenv("DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR") == "1":
            processors = [HelloWorldLogitsProcessor(self.engine.llm.tokenizer)]
            adapters = create_trtllm_adapters(processors)
            sampling_params.logits_processor = adapters

        prefill_result = request.get("prefill_result")
        prefill_prompt_tokens_details = (
            prefill_result.get("prompt_tokens_details") if prefill_result else None
        )

        try:
            # NEW: Updated engine call to include multimodal data
            generation_result = self.engine.llm.generate_async(
                inputs=processed_input,  # Use the correctly extracted inputs
                sampling_params=sampling_params,
                disaggregated_params=disaggregated_params,
                streaming=streaming,
            )

            # Use the context manager to handle cancellation monitoring
            async with self._cancellation_monitor(generation_result, context):
                async for res in generation_result:
                    # TRTLLM engine needs to start generating tokens first before stats
                    # can be retrieved.
                    if self.first_generation and self.publisher:
                        self.publisher.start()
                        self.first_generation = False

                    # If we are not done generating, but there are no outputs, return an error
                    if not res.outputs and not res.finished:
                        yield {"finish_reason": "error", "token_ids": []}
                        break

                    output = res.outputs[0]
                    # The engine returns all tokens generated so far. We must calculate the new
                    # tokens generated in this iteration to create the "delta".
                    next_total_toks = len(output.token_ids)

                    out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

                    if output.finish_reason:
                        out["finish_reason"] = output.finish_reason
                    if output.stop_reason:
                        out["stop_reason"] = output.stop_reason
                    if self.disaggregation_mode == DisaggregationMode.PREFILL:
                        # Return the disaggregated params only when operating in prefill mode.
                        params_dict = self._encode_and_pack_disaggregated_params(
                            output, disaggregated_params, request, res
                        )
                        if params_dict is not None:
                            out["disaggregated_params"] = params_dict

                    if out.get("finish_reason"):
                        num_input_tokens = len(request.get("token_ids", []))

                        prompt_tokens_details = None
                        if prefill_prompt_tokens_details:
                            prompt_tokens_details = prefill_prompt_tokens_details
                        else:
                            if output.request_perf_metrics is not None:
                                kv_cache_metrics = (
                                    output.request_perf_metrics.kv_cache_metrics
                                )
                                cached_tokens = min(
                                    num_input_tokens,
                                    kv_cache_metrics.num_reused_blocks
                                    * self.kv_block_size,
                                )
                                if cached_tokens > 0:
                                    prompt_tokens_details = {
                                        "cached_tokens": int(cached_tokens),
                                    }

                        out["completion_usage"] = {
                            "prompt_tokens": int(num_input_tokens),
                            "completion_tokens": int(next_total_toks),
                            "total_tokens": int(num_input_tokens + next_total_toks),
                            "prompt_tokens_details": prompt_tokens_details,
                        }

                    if res.finished and not out.get("finish_reason"):
                        out["finish_reason"] = "unknown"
                        logging.warning(
                            "Request finished with no finish reason set - this indicates a possible bug"
                        )

                    # Log metrics to TensorRT-LLM MetricsCollector when request finishes
                    if (
                        res.finished
                        and self.metrics_collector
                        and hasattr(res, "metrics_dict")
                    ):
                        try:
                            self.metrics_collector.log_metrics_dict(res.metrics_dict)
                        except Exception as e:
                            logging.warning(f"Failed to log TensorRT-LLM metrics: {e}")

                    # Yield the chunk to the client and update the token count for the next iteration.
                    yield out
                    num_output_tokens_so_far = next_total_toks

        # 1. Client cancellation - don't shutdown
        except asyncio.CancelledError:
            logging.debug(f"Request {request_id}: Client cancelled")
            # _cancellation_monitor already called abort_request
            return  # Just stop, no error response

        # 2. Per-request errors - send to client, don't shutdown
        except RequestError as e:
            logging.warning(f"Request {request_id} error: {e}")
            yield {"finish_reason": "error", "token_ids": []}

        # 3. ALL OTHER ERRORS - graceful shutdown
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logging.error(
                f"Fatal {error_type} in request {request_id}: {error_msg}",
                exc_info=True,
            )

            # Try to send error to client before shutdown
            try:
                yield {
                    "finish_reason": "error",
                    "token_ids": [],
                }
            except Exception:
                pass  # Best effort

            # Initiate graceful shutdown
            await self._initiate_shutdown(e)
