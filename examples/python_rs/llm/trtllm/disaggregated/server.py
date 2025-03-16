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
import json

import uvloop
from common.base_engine import ChatProcessorMixin
from common.kv_router import KVRouter, RoutingStrategy, get_worker_id
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import (
    DisaggChatCompletionRequest,
    DisaggChatCompletionStreamResponse,
    DisaggCompletionStreamResponse,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams

from dynamo.llm import KvIndexer, KvMetricsAggregator
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("info")


class DisaggServer(ChatProcessorMixin):
    def __init__(
        self,
        ctx_chat_client,
        gen_chat_client,
        ctx_completion_client,
        gen_completion_client,
        engine_config: LLMAPIConfig,
        kv_router: KVRouter,
        routing_strategy: RoutingStrategy,
    ):
        self.ctx_chat_client = ctx_chat_client
        self.gen_chat_client = gen_chat_client
        self.ctx_completion_client = ctx_completion_client
        self.gen_completion_client = gen_completion_client
        self.kv_router = kv_router

        if self.kv_router is None:
            if routing_strategy == RoutingStrategy.PREFIX:
                logger.warning(
                    "Prefix routing is not supported without a kv router. Falling back to random."
                )
                routing_strategy = RoutingStrategy.RANDOM

        self.routing_strategy = routing_strategy

        # allows to use tokenizer
        super().__init__(engine_config)

        logger.info(
            f"Initialized Disaggregated Server with routing strategy: {self.routing_strategy}"
        )

    async def _get_ctx_resp(self, request, ctx_client):
        logger.debug(f"Received request {request}")

        request.disaggregated_params = DisaggregatedParams(request_type="context_only")
        logger.debug(f"[router] Sending request to context server: {request}")

        worker_id = ""
        if self.routing_strategy == RoutingStrategy.PREFIX:
            worker_id = await get_worker_id(self.kv_router, request, self._tokenizer)

        if worker_id == "":
            if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                ctx_resp = [
                    resp
                    async for resp in await ctx_client.round_robin(
                        request.model_dump_json()
                    )
                ]
            else:
                # fallback to random
                ctx_resp = [
                    resp
                    async for resp in await ctx_client.random(request.model_dump_json())
                ]
        else:
            ctx_resp = [
                resp
                async for resp in await ctx_client.direct(
                    request.model_dump_json(), int(worker_id)
                )
            ]

        if len(ctx_resp) > 1:
            raise ValueError(
                "Context server returned more than one response. This is currently not supported in disaggregated server."
            )
        logger.debug(
            f"[router] received response from context server: {ctx_resp[0].data()}"
        )
        return ctx_resp[0].data()

    # TODO (shreyasm): The only reason we cant further combine the two methods below is
    # because the disagg params are in different locations.
    # Disagg params should be in under the choices field in the response object.
    # This is the case for completions but not for chat.

    @dynamo_endpoint(CompletionRequest, DisaggCompletionStreamResponse)
    async def generate_completion(self, request):
        # These settings are needed to satisfy request checks.
        request.skip_special_tokens = False
        request.add_special_tokens = False
        request.spaces_between_special_tokens = False

        gen_req = copy.deepcopy(request)

        request.max_tokens = 1
        ctx_resp = await self._get_ctx_resp(request, self.ctx_completion_client)
        ctx_resp_obj = DisaggCompletionStreamResponse.model_validate(ctx_resp)

        gen_req.disaggregated_params = DisaggregatedParams.model_validate(
            ctx_resp_obj.choices[0].disaggregated_params
        )
        gen_req.disaggregated_params.request_type = "generation_only"

        if request.stream:
            yield json.loads(
                ctx_resp_obj.model_dump_json(
                    exclude_unset=True, exclude={"disaggregated_params"}
                )
            )

        logger.debug(f"[router] Sending request to generation server: {gen_req}")
        async for response in await self.gen_completion_client.round_robin(
            gen_req.model_dump_json()
        ):
            logger.debug(
                f"[router] Received response from generation server: {response.data()}"
            )
            gen_resp_obj = DisaggCompletionStreamResponse.model_validate(
                response.data()
            )
            yield json.loads(gen_resp_obj.model_dump_json(exclude_unset=True))

    @dynamo_endpoint(DisaggChatCompletionRequest, DisaggChatCompletionStreamResponse)
    async def generate_chat(self, request):
        # These settings are needed to satisfy request checks.
        request.skip_special_tokens = False
        request.add_special_tokens = False
        request.spaces_between_special_tokens = False

        gen_req = copy.deepcopy(request)

        request.max_tokens = 1
        ctx_resp = await self._get_ctx_resp(request, self.ctx_chat_client)
        ctx_resp_obj = DisaggChatCompletionStreamResponse.model_validate_json(ctx_resp)

        gen_req.disaggregated_params = DisaggregatedParams.model_validate(
            ctx_resp_obj.disaggregated_params
        )
        gen_req.disaggregated_params.request_type = "generation_only"

        if request.stream:
            yield json.loads(
                ctx_resp_obj.model_dump_json(
                    exclude_unset=True, exclude={"disaggregated_params"}
                )
            )

        logger.debug(f"[router] Sending request to generation server: {gen_req}")
        async for response in await self.gen_chat_client.round_robin(
            gen_req.model_dump_json()
        ):
            logger.debug(
                f"[router] Received response from generation server: {response.data()}"
            )
            gen_resp_obj = DisaggChatCompletionStreamResponse.model_validate_json(
                response.data()
            )
            yield json.loads(gen_resp_obj.model_dump_json(exclude_unset=True))


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args, engine_config: LLMAPIConfig):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("dynamo").component("disaggregated_server")
    await component.create_service()

    ctx_completion_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm-ctx")
        .endpoint("completions")
        .client()
    )
    gen_completion_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm-gen")
        .endpoint("completions")
        .client()
    )
    ctx_chat_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm-ctx")
        .endpoint("chat/completions")
        .client()
    )
    gen_chat_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm-gen")
        .endpoint("chat/completions")
        .client()
    )

    if args.routing_strategy == RoutingStrategy.PREFIX:
        # Only listen to context server for now
        kv_listener = runtime.namespace("dynamo").component("tensorrt-llm-ctx")
        await kv_listener.create_service()

        logger.info(
            f"Intializing KV indexer with tokens per block: {args.kv_block_size}"
        )
        indexer = KvIndexer(kv_listener, args.kv_block_size)
        metrics_aggregator = KvMetricsAggregator(kv_listener)
        # FIXME: only using completion_client for now
        # need 1 method for both completion and chat
        kv_router = KVRouter(indexer, metrics_aggregator, ctx_completion_client)
    else:
        kv_router = None

    completions_endpoint = component.endpoint("completions")
    chat_endpoint = component.endpoint("chat/completions")

    disaggregated_server = DisaggServer(
        ctx_chat_client,
        gen_chat_client,
        ctx_completion_client,
        gen_completion_client,
        engine_config,
        kv_router,
        args.routing_strategy,
    )
    await asyncio.gather(
        completions_endpoint.serve_endpoint(disaggregated_server.generate_completion),
        chat_endpoint.serve_endpoint(disaggregated_server.generate_chat),
    )


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()

    asyncio.run(worker(args, engine_config))
