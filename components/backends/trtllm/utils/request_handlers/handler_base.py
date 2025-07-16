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
from dataclasses import asdict, dataclass
from enum import Enum

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from utils.disagg_utils import DisaggregatedParams, DisaggregatedParamsCodec

from dynamo.llm.tensorrtllm.engine import TensorRTLLMEngine
from dynamo.llm.tensorrtllm.publisher import Publisher
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


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

        logging.debug(f"Request: {request}")

        # Check if there is an error in the publisher error queue
        publishers_error = (
            self.publisher.check_error_queue() if self.publisher else None
        )
        if publishers_error:
            raise publishers_error

        inputs = request["token_ids"]

        # Decode the disaggregated params from the request
        disaggregated_params = None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            request["stop_conditions"]["max_tokens"] = 1
            disaggregated_params = LlmDisaggregatedParams(request_type="context_only")

        if "disaggregated_params" in request:
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                raise ValueError("Cannot provide disaggregated_params in prefill mode")
            disaggregated_params = DisaggregatedParamsCodec.decode(
                DisaggregatedParams(**request["disaggregated_params"])
            )
            disaggregated_params.request_type = "generation_only"

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and disaggregated_params is None
        ):
            raise ValueError("Disaggregated params are required for decode mode")

        num_output_tokens_so_far = 0

        sampling_params = self.default_sampling_params
        for key, value in request["sampling_options"].items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        # TODO: Instead of True, we should use streaming from the request.
        # However, currently dynamo run does not send streaming in the request.
        streaming = (
            False if self.disaggregation_mode == DisaggregationMode.PREFILL else True
        )

        async for res in self.engine.llm.generate_async(
            inputs=inputs,
            sampling_params=sampling_params,
            disaggregated_params=disaggregated_params,
            streaming=streaming,
        ):
            # TRTLLM engine needs to start generating tokens first before stats
            # can be retrieved.
            if self.first_generation and self.publisher:
                self.publisher.start()
                self.first_generation = False

            if res.finished and self.disaggregation_mode != DisaggregationMode.PREFILL:
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # Return the disaggregated params only when operating in prefill mode.
                out["disaggregated_params"] = asdict(
                    DisaggregatedParamsCodec.encode(output.disaggregated_params)
                )
            yield out
            num_output_tokens_so_far = next_total_toks
