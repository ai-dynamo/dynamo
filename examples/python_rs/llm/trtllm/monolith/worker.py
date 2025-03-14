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

from common.base_engine import BaseTensorrtLLMEngine, TensorrtLLMEngineConfig
from common.generators import chat_generator, completion_generator
from common.parser import LLMAPIConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionStreamResponse,
)

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("debug")


class TensorrtLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, trt_llm_engine_config: TensorrtLLMEngineConfig):
        super().__init__(trt_llm_engine_config)

    @dynamo_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate_chat(self, request):
        async for response in chat_generator(self, request):
            yield response

    @dynamo_endpoint(CompletionRequest, CompletionStreamResponse)
    async def generate_completion(self, request):
        async for response in completion_generator(self, request):
            yield response


@dynamo_worker()
async def trtllm_worker(runtime: DistributedRuntime, engine_config: LLMAPIConfig):
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
    )
    engine = TensorrtLLMEngine(trt_llm_engine_config)

    await asyncio.gather(
        completions_endpoint.serve_endpoint(engine.generate_completion),
        chat_completions_endpoint.serve_endpoint(engine.generate_chat),
    )
