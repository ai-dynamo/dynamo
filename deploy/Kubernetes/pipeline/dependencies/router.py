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
from common.base_engine import ChatProcessorMixin
from common.kv_router import KVRouter, RoutingStrategy, get_worker_id
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.utils import wait_for_workers
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionStreamResponse,
)

from dynamo.llm import KvIndexer, KvMetricsAggregator
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("info")


class Router(ChatProcessorMixin):
    def __init__(
        self,
        completion_client,
        chat_client,
        kv_router: KVRouter,
        engine_config: LLMAPIConfig,
        routing_strategy: RoutingStrategy,
    ):
        self.completion_client = completion_client
        self.chat_client = chat_client
        self.kv_router = kv_router
        self.routing_strategy = routing_strategy
        # allows to use tokenizer
        super().__init__(engine_config)

        logger.info(
            f"INITIALIZED ROUTER with routing strategy: {self.routing_strategy}"
        )

    async def _generate(self, request, client):
        request.skip_special_tokens = False
        request.add_special_tokens = False
        request.spaces_between_special_tokens = False

        logger.debug(f"[router] Received request {request}")

        worker_id = ""
        if self.routing_strategy == RoutingStrategy.PREFIX:
            worker_id = await get_worker_id(self.kv_router, request, self._tokenizer)

        if worker_id == "":
            if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                async for resp in await client.round_robin(request.model_dump_json()):
                    yield resp.data()
            else:
                # fallback to random
                async for resp in await client.random(request.model_dump_json()):
                    yield resp.data()
        else:
            async for resp in await client.direct(
                request.model_dump_json(), int(worker_id)
            ):
                yield resp.data()

    @dynamo_endpoint(CompletionRequest, CompletionStreamResponse)
    async def generate_completion(self, request):
        async for response in self._generate(request, self.completion_client):
            yield response

    @dynamo_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate_chat(self, request):
        async for response in self._generate(request, self.chat_client):
            yield response


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args, engine_config):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("dynamo").component("router")
    await component.create_service()

    completion_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm")
        .endpoint("completions")
        .client()
    )
    chat_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm")
        .endpoint("chat/completions")
        .client()
    )

    await wait_for_workers(completion_client, args.min_workers)
    await wait_for_workers(chat_client, args.min_workers)

    if args.routing_strategy == RoutingStrategy.PREFIX:
        kv_listener = runtime.namespace("dynamo").component("tensorrt-llm")
        await kv_listener.create_service()

        logger.info(
            f"Intializing KV indexer with tokens per block: {args.kv_block_size}"
        )
        indexer = KvIndexer(kv_listener, args.kv_block_size)
        metrics_aggregator = KvMetricsAggregator(kv_listener)
    else:
        indexer = None
        metrics_aggregator = None

    completions_endpoint = component.endpoint("completions")
    chat_endpoint = component.endpoint("chat/completions")

    # FIXME: only using completion_client for now
    # need 1 method for both completion and chat
    kv_router = KVRouter(indexer, metrics_aggregator, completion_client)
    router = Router(
        completion_client,
        chat_client,
        kv_router,
        engine_config,
        args.routing_strategy,
    )

    await asyncio.gather(
        completions_endpoint.serve_endpoint(router.generate_completion),
        chat_endpoint.serve_endpoint(router.generate_chat),
    )


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()

    asyncio.run(worker(args, engine_config))
