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
            logging.info(
                f"Normalized OpenAI max_tokens to stop_conditions: {request['stop_conditions']['max_tokens']}"
            )

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
                logging.info(
                    "PREFILL: Using ep_disaggregated_params from encoder (EPD flow)"
                )
                logging.info(
                    f"PREFILL: ep_disaggregated_params has multimodal_embedding_handles: {hasattr(ep_disaggregated_params, 'multimodal_embedding_handles') and ep_disaggregated_params.multimodal_embedding_handles is not None}"
                )
                ep_disaggregated_params.request_type = "context_only"
                disaggregated_params = ep_disaggregated_params
            else:
                logging.info(
                    "PREFILL: Creating new LlmDisaggregatedParams (standard flow)"
                )
                disaggregated_params = LlmDisaggregatedParams(
                    request_type="context_only"
                )
            if ep_disaggregated_params:
                logging.info(
                    "PREFILL: Using ep_disaggregated_params from encoder (EPD flow)"
                )
                logging.info(
                    f"PREFILL: ep_disaggregated_params has multimodal_embedding_handles: {hasattr(ep_disaggregated_params, 'multimodal_embedding_handles') and ep_disaggregated_params.multimodal_embedding_handles is not None}"
                )
                ep_disaggregated_params.request_type = "context_only"
                disaggregated_params = ep_disaggregated_params
            else:
                logging.info(
                    "PREFILL: Creating new LlmDisaggregatedParams (standard flow)"
                )
                disaggregated_params = LlmDisaggregatedParams(
                    request_type="context_only"
                )

        # Check for disaggregated_params in prefill_result (new frontend routing)
        # or directly in request (legacy/direct routing)
        prefill_result = request.get("prefill_result")
        epd_metadata = {}

        if prefill_result and "disaggregated_params" in prefill_result:
            # Decode from prefill_result (new frontend disaggregated routing)
            logging.info("DECODE: Received disaggregated_params from prefill_result")
            params_dict = prefill_result["disaggregated_params"]

            # Extract EPD metadata that was packed by prefill worker
            if "_epd_metadata" in params_dict:
                epd_metadata = params_dict.pop("_epd_metadata")
                logging.info(
                    f"DECODE: Extracted _epd_metadata with {len(epd_metadata)} fields"
                )
                for key in ["_epd_processed_prompt", "_epd_prompt_token_ids"]:
                    if key in epd_metadata:
                        value = epd_metadata[key]
                        if isinstance(value, list):
                            logging.info(f"DECODE: {key} length={len(value)}")
                        else:
                            logging.info(
                                f"DECODE: {key}={value[:50]}..."
                                if len(str(value)) > 50
                                else f"DECODE: {key}={value}"
                            )

            disaggregated_params = DisaggregatedParamsCodec.decode(
                DisaggregatedParams(**params_dict)
            )
            logging.info(
                "DECODE: disaggregated_params decoded successfully from prefill_result"
            )
            logging.info(
                f"DECODE: Has multimodal_embedding_handles: {hasattr(disaggregated_params, 'multimodal_embedding_handles') and disaggregated_params.multimodal_embedding_handles is not None}"
            )
            # For full EPD flow, make decoded params available to multimodal processor
            ep_disaggregated_params = disaggregated_params
            disaggregated_params.request_type = "generation_only"

            # In generation-only mode, multimodal embeddings are already processed and in KV cache
            # Remove multimodal_embedding_handles to avoid TRT-LLM validation error
            if (
                hasattr(disaggregated_params, "multimodal_embedding_handles")
                and disaggregated_params.multimodal_embedding_handles
            ):
                logging.info(
                    "DECODE: Removing multimodal_embedding_handles from disaggregated_params (already processed in prefill)"
                )
                disaggregated_params.multimodal_embedding_handles = None

            logging.info("DECODE: Set request_type to generation_only")
        elif "disaggregated_params" in request:
            # Decode from request dict (legacy/direct decode flow)
            logging.info("DECODE: Received disaggregated_params directly in request")
            disaggregated_params = DisaggregatedParamsCodec.decode(
                DisaggregatedParams(**request["disaggregated_params"])
            )
            logging.info("DECODE: disaggregated_params decoded successfully")
            logging.info(
                f"DECODE: Has multimodal_embedding_handles: {hasattr(disaggregated_params, 'multimodal_embedding_handles') and disaggregated_params.multimodal_embedding_handles is not None}"
            )
            # For full EPD flow, make decoded params available to multimodal processor
            ep_disaggregated_params = disaggregated_params
            disaggregated_params.request_type = "generation_only"

            # In generation-only mode, multimodal embeddings are already processed and in KV cache
            # Remove multimodal_embedding_handles to avoid TRT-LLM validation error
            if (
                hasattr(disaggregated_params, "multimodal_embedding_handles")
                and disaggregated_params.multimodal_embedding_handles
            ):
                logging.info(
                    "DECODE: Removing multimodal_embedding_handles from disaggregated_params (already processed in prefill)"
                )
                disaggregated_params.multimodal_embedding_handles = None

            logging.info("DECODE: Set request_type to generation_only")

        # Default to text-based input. This will be overwritten if multimodal
        # content is found and processed.
        processed_input = None
        # Now ep_disaggregated_params is properly set for both prefill and decode modes
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            # max_tokens is already correctly set by the Rust frontend's PrefillRouter
            # No need to restore it here
            logging.info(
                f"DECODE: max_tokens from frontend: {request.get('stop_conditions', {}).get('max_tokens', 'NOT SET')}"
            )
            # Decode worker with generation_only mode
            # Pass the same inputs format as prefill
            # Check epd_metadata (packed by prefill), then prefill_result, then direct request
            epd_prompt = epd_metadata.get("_epd_processed_prompt")
            epd_token_ids = epd_metadata.get("_epd_prompt_token_ids")

            if epd_prompt:
                logging.info("DECODE: Found _epd_processed_prompt in epd_metadata")
            elif prefill_result:
                epd_prompt = prefill_result.get("_epd_processed_prompt")
                epd_token_ids = prefill_result.get("_epd_prompt_token_ids")
                if epd_prompt:
                    logging.info(
                        "DECODE: Found _epd_processed_prompt in prefill_result"
                    )

            if not epd_prompt and "_epd_processed_prompt" in request:
                epd_prompt = request["_epd_processed_prompt"]
                epd_token_ids = request.get("_epd_prompt_token_ids")
                logging.info("DECODE: Found _epd_processed_prompt in request")

            if epd_prompt:
                # In EPD generation-only mode (decode), pass the SAME input format as prefill
                # This matches TRT-LLM's test: llm_decode.generate(inputs, disaggregated_params=...)
                # The inputs dict provides prompt structure, disaggregated_params provide multimodal embeddings
                if epd_token_ids:
                    prompt_token_ids = epd_token_ids
                    logging.info(
                        f"DECODE: Using pre-computed token IDs (length={len(epd_token_ids)})"
                    )
                else:
                    # Fallback: tokenize if token IDs not provided
                    prompt_token_ids = self.engine.llm.tokenizer.encode(
                        epd_prompt, add_special_tokens=False
                    )
                    logging.info(
                        f"DECODE: Tokenized prompt (length={len(prompt_token_ids)})"
                    )

                # In generation-only mode, multimodal data is in disaggregated_params
                # BUT the LLaVA model REQUIRES the prompt string (modeling_llava_next.py:185)
                # Pass the same dict format as prefill: {"prompt": ..., "prompt_token_ids": ...}
                # This matches TRT-LLM test: llm_decode.generate(inputs, disaggregated_params=...)
                processed_input = {
                    "prompt": epd_prompt,
                    "prompt_token_ids": prompt_token_ids,
                }
                logging.info(
                    "DECODE: Using EPD input dict (prompt + token_ids) for generation-only mode"
                )
                logging.info(
                    f"DECODE: Prompt length: {len(epd_prompt)} chars, Token IDs length: {len(prompt_token_ids)}"
                )
                logging.info(
                    "DECODE: Multimodal embedding handles are in disaggregated_params"
                )

                # Remove ALL multimodal data from request to avoid TRT-LLM validation error
                # In generation-only mode, ALL multimodal data must be in disaggregated_params only
                mm_keys_to_remove = ["multi_modal_data", "image_data", "mm_data"]
                for key in mm_keys_to_remove:
                    if key in request:
                        removed_data = request.pop(key)
                        logging.info(
                            f"DECODE: Removed {key} from request (already in disaggregated_params)"
                        )
                        if isinstance(removed_data, dict):
                            logging.info(
                                f"DECODE: {key} had keys: {list(removed_data.keys())}"
                            )
            elif self.multimodal_processor:
                # Encode/Prefill worker: Process multimodal content normally
                # In EPD flow, multimodal_processor should be called in PREFILL/ENCODE modes only
                # DECODE mode should skip this and use EPD metadata from prefill
                processed_input = (
                    await self.multimodal_processor.process_openai_request(
                        request, embeddings, ep_disaggregated_params
                    )
                )
            else:
                logging.info(
                    "DECODE: No multimodal_processor found, using request token_ids"
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
                    logging.info(
                        "PREFILL: Multimodal content processed by multimodal_processor"
                    )
                else:
                    # Fallback to text-only if no multimodal content
                    processed_input = request.get("token_ids")
                    logging.info("PREFILL: No multimodal content, using token_ids")
            else:
                # text-only flow
                processed_input = request.get("token_ids")
                logging.info("PREFILL: No multimodal_processor, using token_ids")

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

        # Log what we're passing to the engine
        logging.info(
            f"{self.disaggregation_mode.value.upper()}: ========== ENGINE CALL INPUT =========="
        )
        logging.info(f"  - processed_input type: {type(processed_input)}")
        if isinstance(processed_input, dict):
            logging.info(f"  - processed_input keys: {list(processed_input.keys())}")
            # Check for any multimodal-related keys that shouldn't be there
            mm_keys = [
                k
                for k in processed_input.keys()
                if "multi" in k.lower() or "image" in k.lower() or "mm_" in k
            ]
            if mm_keys:
                logging.warning(f"  - FOUND MULTIMODAL KEYS IN INPUT: {mm_keys}")
        elif isinstance(processed_input, list):
            logging.info(f"  - processed_input list length: {len(processed_input)}")
        logging.info(
            f"  - disaggregated_params: {'None' if disaggregated_params is None else 'present'}"
        )
        if disaggregated_params:
            logging.info(
                f"  - disaggregated_params.request_type: {getattr(disaggregated_params, 'request_type', 'NOT SET')}"
            )
            has_mm_handles = (
                hasattr(disaggregated_params, "multimodal_embedding_handles")
                and disaggregated_params.multimodal_embedding_handles
            )
            logging.info(
                f"  - disaggregated_params.multimodal_embedding_handles: {'present (' + str(len(disaggregated_params.multimodal_embedding_handles)) + ' handle(s))' if has_mm_handles else 'None'}"
            )
        logging.info(
            f"{self.disaggregation_mode.value.upper()}: ========================================"
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

        # Final validation before calling engine
        logging.info(
            f"{self.disaggregation_mode.value.upper()}: ========== FINAL VALIDATION =========="
        )
        logging.info(f"  - inputs type: {type(processed_input)}")
        logging.info(f"  - inputs is dict: {isinstance(processed_input, dict)}")
        if isinstance(processed_input, dict):
            all_keys = list(processed_input.keys())
            logging.info(f"  - ALL input dict keys: {all_keys}")
            for k in all_keys:
                if isinstance(processed_input[k], (list, torch.Tensor)):
                    val_type = type(processed_input[k]).__name__
                    val_len = len(processed_input[k])
                    logging.info(f"    - {k}: {val_type} (len={val_len})")
                else:
                    logging.info(f"    - {k}: {type(processed_input[k]).__name__}")
        logging.info(
            f"  - disaggregated_params is None: {disaggregated_params is None}"
        )
        if disaggregated_params:
            logging.info(
                f"  - disaggregated_params.request_type: {disaggregated_params.request_type}"
            )
        logging.info("=========================================")

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
                        # In EPD flow, output.disaggregated_params might be None, use the input params
                        logging.info(
                            f"PREFILL: output.disaggregated_params is {'None' if output.disaggregated_params is None else 'present'}"
                        )
                        logging.info(
                            f"PREFILL: input disaggregated_params is {'None' if disaggregated_params is None else 'present'}"
                        )

                        params_to_encode = (
                            output.disaggregated_params
                            if output.disaggregated_params is not None
                            else disaggregated_params
                        )

                        # In EPD flow, manually preserve multimodal_embedding_handles from input
                        # because TRT-LLM engine may not propagate them through prefill
                        if (
                            params_to_encode is not None
                            and disaggregated_params is not None
                        ):
                            input_handles = getattr(
                                disaggregated_params,
                                "multimodal_embedding_handles",
                                None,
                            )
                            output_handles = getattr(
                                params_to_encode, "multimodal_embedding_handles", None
                            )

                            if input_handles is not None and output_handles is None:
                                logging.info(
                                    "PREFILL: Preserving multimodal_embedding_handles from input (EPD flow)"
                                )
                                params_to_encode.multimodal_embedding_handles = (
                                    input_handles
                                )
                                # Also preserve hashes if they exist
                                input_hashes = getattr(
                                    disaggregated_params, "multimodal_hashes", None
                                )
                                if input_hashes is not None:
                                    params_to_encode.multimodal_hashes = input_hashes

                        if params_to_encode is not None:
                            logging.info(
                                f"PREFILL: params_to_encode has multimodal_embedding_handles: {hasattr(params_to_encode, 'multimodal_embedding_handles') and params_to_encode.multimodal_embedding_handles is not None}"
                            )
                            logging.info(
                                f"PREFILL: params_to_encode.request_type: {getattr(params_to_encode, 'request_type', 'NOT SET')}"
                            )
                        else:
                            logging.warning(
                                "PREFILL: params_to_encode is None - no disaggregated params to send!"
                            )

                        encoded_params = DisaggregatedParamsCodec.encode(
                            params_to_encode
                        )

                        if encoded_params is not None:
                            logging.info(
                                "PREFILL: Successfully encoded disaggregated params"
                            )
                            params_dict = asdict(encoded_params)

                            # Pack EPD-specific fields into disaggregated_params to ensure they're forwarded
                            # The frontend only forwards disaggregated_params and prompt_tokens_details from prefill response
                            # Note: max_tokens is already handled by Rust frontend's PrefillRouter
                            epd_metadata = {}

                            if "_epd_processed_prompt" in request and res.prompt:
                                epd_metadata["_epd_processed_prompt"] = res.prompt
                                logging.info(
                                    f"PREFILL: Packing _epd_processed_prompt into disaggregated_params: {res.prompt[:50]}..."
                                )

                            if (
                                "_epd_prompt_token_ids" in request
                                and res.prompt_token_ids
                            ):
                                epd_metadata[
                                    "_epd_prompt_token_ids"
                                ] = res.prompt_token_ids
                                logging.info(
                                    f"PREFILL: Packing _epd_prompt_token_ids into disaggregated_params (length={len(res.prompt_token_ids)})"
                                )

                            # Add EPD metadata to the disaggregated_params dict
                            if epd_metadata:
                                params_dict["_epd_metadata"] = epd_metadata
                                logging.info(
                                    f"PREFILL: Added _epd_metadata with {len(epd_metadata)} fields to disaggregated_params"
                                )

                            out["disaggregated_params"] = params_dict
                            logging.info(
                                "PREFILL: Added disaggregated_params to response"
                            )
                        else:
                            logging.error(
                                "PREFILL: encoded_params is None - decode worker will fail!"
                            )

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
