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

import uvloop
from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.generators import chat_generator, completion_generator
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import AdaptedChatCompletionRequest, AdaptedCompletionRequest
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionStreamResponse,
    CompletionStreamResponse,
)

from dynamo.llm import KvMetricsPublisher
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("info")


class TensorrtLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, trt_llm_engine_config: TensorrtLLMEngineConfig):
        super().__init__(trt_llm_engine_config)

    @dynamo_endpoint(AdaptedChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate_chat(self, request):
        if request.max_completion_tokens is not None:
            request.max_tokens = request.max_completion_tokens
        async for response in chat_generator(self, request):
            yield response
        self._start_threads()

    @dynamo_endpoint(AdaptedCompletionRequest, CompletionStreamResponse)
    async def generate_completion(self, request):
        if request.max_completion_tokens is not None:
            request.max_tokens = request.max_completion_tokens
        async for response in completion_generator(self, request):
            yield response
        self._start_threads()


@dynamo_worker()
async def trtllm_worker(runtime: DistributedRuntime, engine_config: LLMAPIConfig, args):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    namespace_str = "dynamo"
    component_str = "tensorrt-llm"

    component = runtime.namespace(namespace_str).component(component_str)
    await component.create_service()

    completions_endpoint = component.endpoint("completions")
    chat_completions_endpoint = component.endpoint("chat/completions")

    trt_llm_engine_config = TensorrtLLMEngineConfig(
        namespace_str=namespace_str,
        component_str=component_str,
        engine_config=engine_config,
        publish_stats=args.publish_stats,
        publish_kv_cache_events=args.publish_kv_cache_events,
        kv_block_size=args.kv_block_size,
    )

    if args.publish_stats:
        trt_llm_engine_config.kv_metrics_publisher = KvMetricsPublisher()

    # TODO: fix
    trt_llm_engine_config.worker_id = completions_endpoint.lease_id()

    engine = TensorrtLLMEngine(trt_llm_engine_config)

    coros = [
        completions_endpoint.serve_endpoint(engine.generate_completion),
        chat_completions_endpoint.serve_endpoint(engine.generate_chat),
    ]
    if args.publish_stats:
        coros.append(
            trt_llm_engine_config.kv_metrics_publisher.create_endpoint(component)
        )

    await asyncio.gather(*coros)


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()

    asyncio.run(trtllm_worker(engine_config, args))
