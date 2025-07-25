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

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams

from dynamo.llm.tensorrtllm.engine import TensorRTLLMEngine
from dynamo.llm.tensorrtllm.publisher import Publisher
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.utils.disagg_utils import (
    DisaggregatedParams,
    DisaggregatedParamsCodec,
)
from dynamo.trtllm.utils.multimodal_processor import MultimodalRequestProcessor

configure_dynamo_logging()

# Configure detailed logging for disaggregation
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DisaggregationMode(Enum):
    AGGREGATED = "prefill_and_decode"
    PREFILL = "prefill"
    DECODE = "decode"


class DisaggregationStrategy(Enum):
    PREFILL_FIRST = "prefill_first"
    DECODE_FIRST = "decode_first"


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
    multimodal_processor: Optional[
        MultimodalRequestProcessor
    ] = None  # NEW: for multimodal support
    tokenizer: Optional[object] = None  # NEW: for decoding tokens


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
        self.multimodal_processor = config.multimodal_processor
        self.tokenizer = config.tokenizer  # NEW: store tokenizer
        self.first_generation = True

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

    async def generate_locally(self, request: dict):
        """
        Generate responses based on the disaggregation mode in the request.
        """

        self.first_generation = True

        # FIX: Normalize the request to handle OpenAI format
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

        logger.debug("=" * 80)
        logger.debug("🚀 DISAGGREGATION FLOW STARTED")
        logger.debug("=" * 80)
        logger.debug("📋 Request Overview:")
        logger.debug(f"   Mode: {self.disaggregation_mode.value}")
        logger.debug(f"   Strategy: {self.disaggregation_strategy.value}")

        # NEW: Check for multimodal request and process it
        if "messages" in request and self.multimodal_processor:
            logger.debug("🖼️  MULTIMODAL REQUEST DETECTED")
            logger.debug("🔄 Processing with MultimodalRequestProcessor...")
            processed_inputs = await self.multimodal_processor.process_openai_request(
                request
            )

            if "processed_inputs" in processed_inputs:
                logger.debug("✅ Multimodal processing completed")
                processed_input = processed_inputs["processed_inputs"][0]
                logger.debug("processed_input: ", processed_input)

            else:
                logger.debug("📝 No multimodal content, using original request")
        else:
            # Original text-only flow
            logger.debug(f"   Input tokens count: {len(request.get('token_ids', []))}")

        logger.debug(
            f"   Max tokens requested: {request.get('stop_conditions', {}).get('max_tokens', 'Not specified')}"
        )

        # Log the full request for debugging (truncated for readability)
        request_summary = {
            k: v
            for k, v in request.items()
            if k not in ["token_ids", "processed_inputs"]
        }
        if "token_ids" in request:
            request_summary["token_ids_count"] = len(request.get("token_ids", []))
        logger.debug(f"📦 Full request details: {request_summary}")

        # Check if there is an error in the publisher error queue
        publishers_error = (
            self.publisher.check_error_queue() if self.publisher else None
        )
        if publishers_error:
            raise publishers_error

        # Decode the disaggregated params from the request
        disaggregated_params = None

        logger.debug("=" * 60)
        logger.debug("🔧 DISAGGREGATED PARAMETERS SETUP")
        logger.debug("=" * 60)

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            request["stop_conditions"]["max_tokens"] = 1
            disaggregated_params = LlmDisaggregatedParams(request_type="context_only")
            logger.debug(f"Prefill disaggregated_params: {disaggregated_params}")

        if "disaggregated_params" in request:
            logger.debug(
                "📥 DECODE MODE: Received disaggregated parameters from prefill worker"
            )
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                logger.error(
                    "❌ ERROR: Cannot provide disaggregated_params in prefill mode"
                )
                raise ValueError("Cannot provide disaggregated_params in prefill mode")

            logger.debug("   ➜ Decoding received disaggregated parameters...")
            received_params = DisaggregatedParams(**request["disaggregated_params"])
            logger.debug(f"   ➜ Raw received params: {received_params}")

            disaggregated_params = DisaggregatedParamsCodec.decode(received_params)
            disaggregated_params.request_type = "generation_only"
            logger.debug(
                f"   ➜ Context request ID: {disaggregated_params.ctx_request_id}"
            )
            logger.debug(
                f"   ➜ First generation tokens: {disaggregated_params.first_gen_tokens}"
            )
            logger.debug(
                f"   ➜ Opaque state size: {len(disaggregated_params.opaque_state) if disaggregated_params.opaque_state else 0} bytes"
            )

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and disaggregated_params is None
        ):
            logger.error(
                "❌ ERROR: Disaggregated params are required for decode mode but not provided"
            )
            raise ValueError("Disaggregated params are required for decode mode")

        if disaggregated_params is None:
            logger.debug(
                "🔄 AGGREGATED MODE: No disaggregated parameters (normal inference)"
            )

        num_output_tokens_so_far = 0

        sampling_params = self.default_sampling_params
        logger.info(f"📊 Default sampling params: {sampling_params}")

        for key, value in request["sampling_options"].items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                logger.debug(
                    f"   ➜ Updating {key}: {getattr(sampling_params, key)} → {value}"
                )
                setattr(sampling_params, key, value)

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            logger.debug(f"   ➜ Setting max_tokens: {max_tokens}")
            sampling_params.max_tokens = max_tokens

        # TODO: Instead of True, we should use streaming from the request.
        # However, currently dynamo run does not send streaming in the request.
        streaming = (
            False if self.disaggregation_mode == DisaggregationMode.PREFILL else True
        )
        logger.debug(
            f"🌊 Streaming mode: {streaming} (prefill=False, decode/aggregated=True)"
        )

        logger.debug("=" * 60)
        logger.debug("🏭 TENSORRT-LLM ENGINE GENERATION")
        logger.debug("=" * 60)
        logger.debug("🔄 Starting generation with:")
        logger.debug(f"   ➜ Sampling params: {sampling_params}")
        logger.debug(
            f"   ➜ Disaggregated params: {'Present' if disaggregated_params else 'None'}"
        )
        logger.debug(f"   ➜ Streaming: {streaming}")

        request_id = request.get("id") or request.get("request_id", "unknown-id")
        model_name = request.get("model", "unknown_model")

        # NEW: Updated engine call to include multimodal data
        async for res in self.engine.llm.generate_async(
            inputs=processed_input,  # Use the correctly extracted inputs
            sampling_params=sampling_params,
            disaggregated_params=disaggregated_params,
            streaming=streaming,
        ):
            logger.debug("=" * 40)
            logger.debug("📤 GENERATION ITERATION")
            logger.debug("=" * 40)

            # TRTLLM engine needs to start generating tokens first before stats
            # can be retrieved.
            if self.first_generation and self.publisher:
                logger.debug("📊 Starting publisher for metrics collection")
                self.publisher.start()

            logger.debug(
                f"🔍 Response status: finished={res.finished}, outputs_count={len(res.outputs) if res.outputs else 0}"
            )

            if res.finished and self.disaggregation_mode != DisaggregationMode.PREFILL:
                logger.debug("✅ GENERATION COMPLETE")
                logger.debug("   ➜ Response finished, yielding final stop token")
                final_choice = {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
                final_out = {
                    "id": request_id,
                    "model": model_name,
                    "created": int(time.time()),
                    "object": "chat.completion.chunk",
                    "choices": [final_choice],
                    "finish_reason": "stop",
                }
                yield final_out
                break

            if not res.outputs:
                logger.error("❌ ERROR: No outputs received from engine")
                yield {"finish_reason": "error", "token_ids": []}
                break

            logger.info(f"res.outputs: {res.outputs}")
            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            new_tokens = output.token_ids[num_output_tokens_so_far:]
            delta_text = self.tokenizer.decode(new_tokens)
            logger.debug(f"delta_text: {delta_text}")

            delta = {"content": delta_text if delta_text else ""}
            if self.first_generation:
                delta["role"] = "assistant"
                self.first_generation = False

            choice = {
                "index": 0,
                "delta": delta,
                "finish_reason": output.finish_reason,
            }

            out = {
                "id": request_id,
                "model": model_name,
                "created": int(time.time()),
                "object": "chat.completion.chunk",
                "choices": [choice],
            }
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # Return the disaggregated params only when operating in prefill mode.
                out["disaggregated_params"] = asdict(
                    DisaggregatedParamsCodec.encode(output.disaggregated_params)
                )
            yield out
            num_output_tokens_so_far = next_total_toks
