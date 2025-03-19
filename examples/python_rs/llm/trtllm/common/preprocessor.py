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
import json
from enum import Enum

import uvloop
from common.base_engine import ChatProcessorMixin
from common.generators import (
    ServerType,
    chat_postprocessor,
    chat_preprocessor,
    completion_postprocessor,
    completion_preprocessor,
)
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import (
    AdaptedChatCompletionRequest,
    AdaptedCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionStreamResponse,
)
from tensorrt_llm.logger import logger

from dynamo.runtime import Client, DistributedRuntime, dynamo_endpoint, dynamo_worker


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class Processor(ChatProcessorMixin):
    def __init__(self, args, engine_config: LLMAPIConfig, workers_client: Client):
        self.args = args
        self.engine_config = engine_config
        self.workers_client = workers_client
        super().__init__(self.engine_config)

    async def _generate(self, raw_request, request_type: RequestType):
        logger.debug(f"[preprocessor] Received request: {raw_request}")

        # worker_id = ""
        if request_type == RequestType.CHAT:
            preprocessed_request = await chat_preprocessor(raw_request, self._tokenizer)
        else:
            preprocessed_request = await completion_preprocessor(raw_request)

        engine_generator = await self.workers_client.round_robin(
            preprocessed_request.model_dump_json()
        )

        if request_type == RequestType.CHAT:
            async for response in chat_postprocessor(
                engine_generator,
                raw_request,
                preprocessed_request.conversation,
                ServerType.GEN,
                self.chat_processor,
            ):
                yield json.loads(response)
        else:
            async for response in completion_postprocessor(
                engine_generator, raw_request, self.completions_processor
            ):
                yield json.loads(response)

    @dynamo_endpoint(AdaptedChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate_chat(self, raw_request):
        async for response in self._generate(raw_request, RequestType.CHAT):
            yield response

    @dynamo_endpoint(AdaptedCompletionRequest, CompletionStreamResponse)
    async def generate_completions(self, raw_request):
        async for response in self._generate(raw_request, RequestType.COMPLETION):
            yield response


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args, engine_config: LLMAPIConfig):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    preprocess_component = runtime.namespace("dynamo").component("preprocess")
    await preprocess_component.create_service()

    workers_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm")
        .endpoint("generate")
        .client()
    )
    # router_client = (
    #     await runtime.namespace("dynamo")
    #     .component("router")
    #     .endpoint("generate")
    #     .client()
    # )

    chat_endpoint = preprocess_component.endpoint("chat/completions")
    completions_endpoint = preprocess_component.endpoint("completions")

    processor = Processor(args, engine_config, workers_client)

    await asyncio.gather(
        chat_endpoint.serve_endpoint(processor.generate_chat),
        completions_endpoint.serve_endpoint(processor.generate_completions),
    )


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()

    asyncio.run(worker(args, engine_config))
