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

import uvloop
from common.base_engine import ChatProcessorMixin
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import (
    DynamoTRTLLMChatCompletionRequest,
    DynamoTRTLLMChatCompletionStreamResponse,
    DynamoTRTLLMCompletionRequest,
    DynamoTRTLLMCompletionStreamResponse,
    Tokens,
)
from common.utils import (
    RequestType,
    ServerType,
    wait_for_workers,
    RoutingStrategy,
)
from tensorrt_llm.logger import logger

from dynamo.runtime import Client, DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("debug")


async def get_worker_id(kv_router_client: Client, tokens: Tokens) -> str:
    worker_id_generator: AsyncIterator = await kv_router_client.generate(tokens.model_dump_json())

    response = await worker_id_generator.__anext__()  # only one worker id is returned
    print(response.data())
    worker_id, prefix_hit_rate = response.data().split("_")
    prefix_hit_rate = float(prefix_hit_rate)

    logger.debug(
        f"Scheduling to worker_id: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
    )
    return worker_id


class Processor(ChatProcessorMixin):
    def __init__(
        self,
        engine_config: LLMAPIConfig,
        workers_client: Client,
        kv_router_client: Client,
        routing_strategy: RoutingStrategy,
    ):
        self.engine_config = engine_config
        self.workers_client = workers_client
        self.kv_router_client = kv_router_client
        self.routing_strategy = routing_strategy
        super().__init__(self.engine_config)

    async def _generate(self, raw_request, request_type: RequestType):
        raw_request.skip_special_tokens = False
        raw_request.add_special_tokens = False
        raw_request.spaces_between_special_tokens = False
        logger.debug(f"[preprocessor] Received request: {raw_request}")

        if request_type == RequestType.CHAT:
            preprocessed_request = await self.chat_processor.preprocess(raw_request)
        else:
            preprocessed_request = await self.completions_processor.preprocess(
                raw_request
            )

        worker_id = ""
        if self.routing_strategy == RoutingStrategy.PREFIX:
            worker_id = await get_worker_id(
                self.kv_router_client, preprocessed_request.tokens
            )

        if worker_id == "":
            if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                engine_generator = await self.workers_client.round_robin(
                    preprocessed_request.model_dump_json()
                )
            else:
                # fallback to random
                engine_generator = await self.workers_client.random(
                    preprocessed_request.model_dump_json()
                )
        else:
            engine_generator = await self.workers_client.direct(
                preprocessed_request.model_dump_json(), int(worker_id)
            )

        if request_type == RequestType.CHAT:
            async for response in self.chat_processor.postprocess(
                engine_generator,
                raw_request,
                preprocessed_request.conversation,
                ServerType.GEN,
            ):
                logger.debug(f"[preprocessor] Response: {response}")
                yield json.loads(response)
        else:
            async for response in self.completions_processor.postprocess(
                engine_generator, raw_request
            ):
                logger.debug(f"[preprocessor] Response: {response}")
                yield json.loads(response)

    @dynamo_endpoint(
        DynamoTRTLLMChatCompletionRequest, DynamoTRTLLMChatCompletionStreamResponse
    )
    async def generate_chat(self, raw_request):
        async for response in self._generate(raw_request, RequestType.CHAT):
            yield response

    @dynamo_endpoint(
        DynamoTRTLLMCompletionRequest, DynamoTRTLLMCompletionStreamResponse
    )
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
    await wait_for_workers(workers_client, args.min_workers)

    if args.routing_strategy == RoutingStrategy.PREFIX:
        kv_router_client = (
            await runtime.namespace("dynamo")
            .component("router")
            .endpoint("generate")
            .client()
        )
        logger.info(f"Initialized KV router client for prefix routing.")
    else:
        kv_router_client = None

    chat_endpoint = preprocess_component.endpoint("chat/completions")
    completions_endpoint = preprocess_component.endpoint("completions")

    processor = Processor(
        engine_config, workers_client, kv_router_client, args.routing_strategy
    )

    await asyncio.gather(
        chat_endpoint.serve_endpoint(processor.generate_chat),
        completions_endpoint.serve_endpoint(processor.generate_completions),
    )


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()

    asyncio.run(worker(args, engine_config))
