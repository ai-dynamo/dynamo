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
from common.generators import (
    chat_postprocessor,
    chat_preprocessor,
    completion_postprocessor,
    completion_preprocessor,
)
from common.kv_router import KVRouter, RoutingStrategy, get_worker_id
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import (
    AdaptedCompletionRequest,
    DisaggChatCompletionRequest,
    DisaggChatCompletionStreamResponse,
    DisaggCompletionStreamResponse,
)
from common.utils import RequestType, ServerType
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import DisaggregatedParams

from dynamo.llm import KvIndexer, KvMetricsAggregator
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("debug")


class DisaggServer(ChatProcessorMixin):
    def __init__(
        self,
        ctx_client,
        gen_client,
        engine_config: LLMAPIConfig,
        kv_router: KVRouter,
        routing_strategy: RoutingStrategy,
    ):
        self.ctx_client = ctx_client
        self.gen_client = gen_client
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

    async def _generate(self, request, request_type: RequestType):
        request.skip_special_tokens = False
        request.add_special_tokens = False
        request.spaces_between_special_tokens = False
        gen_req = copy.deepcopy(request)

        request.max_tokens = 1
        request.disaggregated_params = DisaggregatedParams(request_type="context_only")
        logger.debug(f"[disagg-server] Sending request to context server: {request}")

        if request_type == RequestType.CHAT:
            preprocessed_request = await chat_preprocessor(request, self._tokenizer)
            response_cls = DisaggChatCompletionStreamResponse
        else:
            preprocessed_request = await completion_preprocessor(request)
            response_cls = DisaggCompletionStreamResponse

        logger.debug(f"[disagg-server] Preprocessed request: {preprocessed_request}")

        # TODO: move this somewhere. can be reused by monolith
        worker_id = ""
        if self.routing_strategy == RoutingStrategy.PREFIX:
            worker_id = await get_worker_id(self.kv_router, request, self._tokenizer)

        if worker_id == "":
            if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                ctx_generator = await self.ctx_client.round_robin(
                    preprocessed_request.model_dump_json()
                )
            else:
                # fallback to random
                ctx_generator = await self.ctx_client.random(
                    preprocessed_request.model_dump_json()
                )
        else:
            ctx_generator = await self.ctx_client.direct(
                preprocessed_request.model_dump_json(), int(worker_id)
            )

        if request_type == RequestType.CHAT:
            ctx_resp = [
                resp
                async for resp in chat_postprocessor(
                    ctx_generator,
                    request,
                    preprocessed_request.conversation,
                    ServerType.CTX,
                    self.chat_processor,
                )
            ]
        else:
            ctx_resp = [
                resp
                async for resp in completion_postprocessor(
                    ctx_generator, request, self.completions_processor
                )
            ]

        if len(ctx_resp) > 1:
            raise ValueError(
                "Context server returned more than one response. This is currently not supported in disaggregated server."
            )
        logger.debug(
            f"[disagg-server] received response from context server: {ctx_resp[0].data()}"
        )
        ctx_resp_obj = response_cls.model_validate(ctx_resp[0].data())

        gen_req.disaggregated_params = DisaggregatedParams.model_validate(
            ctx_resp_obj.choices[0].disaggregated_params
        )
        gen_req.disaggregated_params.request_type = "generation_only"

        yield json.loads(
            ctx_resp_obj.model_dump_json(
                exclude_unset=True, exclude={"disaggregated_params"}
            )
        )

        if request_type == RequestType.CHAT:
            preprocessed_gen_request = await chat_preprocessor(gen_req, self._tokenizer)
        else:
            preprocessed_gen_request = await completion_preprocessor(gen_req)

        logger.debug(
            f"[disagg-server] Sending request to generation server: {preprocessed_gen_request}"
        )
        gen_generator = await self.gen_client.round_robin(
            preprocessed_gen_request.model_dump_json()
        )

        if request_type == RequestType.CHAT:
            async for resp in chat_postprocessor(
                gen_generator,
                gen_req,
                preprocessed_gen_request.conversation,
                ServerType.GEN,
                self.chat_processor,
            ):
                gen_resp_obj = response_cls.model_validate(resp.data())
                yield json.loads(gen_resp_obj.model_dump_json(exclude_unset=True))
        else:
            async for resp in completion_postprocessor(
                gen_generator, gen_req, self.completions_processor
            ):
                gen_resp_obj = response_cls.model_validate(resp.data())
                yield json.loads(gen_resp_obj.model_dump_json(exclude_unset=True))

    @dynamo_endpoint(AdaptedCompletionRequest, DisaggCompletionStreamResponse)
    async def generate_completion(self, request):
        await self._generate(
            request,
            RequestType.COMPLETION,
        )

    @dynamo_endpoint(DisaggChatCompletionRequest, DisaggChatCompletionStreamResponse)
    async def generate_chat(self, request):
        await self._generate(
            request,
            RequestType.CHAT,
        )


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args, engine_config: LLMAPIConfig):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("dynamo").component("disaggregated_server")
    await component.create_service()

    ctx_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm-ctx")
        .endpoint("generate")
        .client()
    )
    gen_client = (
        await runtime.namespace("dynamo")
        .component("tensorrt-llm-gen")
        .endpoint("generate")
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
        kv_router = KVRouter(indexer, metrics_aggregator, ctx_client)
    else:
        kv_router = None

    completions_endpoint = component.endpoint("completions")
    chat_endpoint = component.endpoint("chat/completions")

    disaggregated_server = DisaggServer(
        ctx_client,
        gen_client,
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
