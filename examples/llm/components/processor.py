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
import uuid
from enum import Enum
from typing import AsyncIterator, Tuple, Union

from components.kv_router import Router
from components.worker import VllmWorker
from transformers import AutoTokenizer
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.logging import check_required_workers
from utils.protocol import MyRequestOutput, Tokens, vLLMGenerateRequest
from utils.vllm import RouterType, parse_vllm_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

from dynamo.llm import KvMetricsAggregator
from dynamo.runtime import EtcdKvCache
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    worker = depends(VllmWorker)
    router = depends(Router)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.model_config = self.engine_args.create_model_config()
        self.tokenizer = self._create_tokenizer(self.engine_args)
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        self.completions_processor = CompletionsProcessor(
            self.tokenizer, self.model_config
        )
        self.min_workers = 1
        print(f"Processor init: {self.engine_args.router}")

    def _create_tokenizer(self, engine_args: AsyncEngineArgs) -> AnyTokenizer:
        """Create a TokenizerGroup using engine arguments similar to VLLM's approach"""
        model_path = engine_args.model

        # Create the base tokenizer with VLLM's typical settings
        base_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
            use_fast=True,  # VLLM might use the fast tokenizer for efficiency
        )
        return base_tokenizer

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = VllmWorker.dynamo_address()  # type: ignore
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )

        if self.engine_args.router == RouterType.KV:
            router_ns, router_name = Router.dynamo_address()  # type: ignore
            self.router_client = (
                await runtime.namespace(router_ns)
                .component(router_name)
                .endpoint("generate")
                .client()
            )

        await check_required_workers(self.worker_client, self.min_workers)

        kv_listener = runtime.namespace("dynamo").component("VllmWorker")
        await kv_listener.create_service()
        self.metrics_aggregator = KvMetricsAggregator(kv_listener)

        self.etcd_kv_cache = await EtcdKvCache.create(
            runtime.etcd_client(),
            "/dynamo/processor/",
            {"router": self.engine_args.router},
        )

    async def _get_kv_load(self):
        metrics = await self.metrics_aggregator.get_metrics()
        kv_load = {}
        for endpoint in metrics.endpoints:
            worker_id = endpoint.worker_id
            kv_load[worker_id] = getattr(endpoint, "gpu_cache_usage_perc", 0.0)
        return kv_load

    async def _get_pending_requests(self):
        metrics = await self.metrics_aggregator.get_metrics()
        pending_requests = {}
        for endpoint in metrics.endpoints:
            worker_id = endpoint.worker_id
            pending_requests[worker_id] = getattr(endpoint, "num_requests_waiting", 0)
        return pending_requests

    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4())
        logger.debug(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)
        # TODO: queue request at processor when engines are full
        router_mode = (await self.etcd_kv_cache.get("router")).decode()
        if router_mode == RouterType.KV:
            router_generator = await self.router_client.generate(
                Tokens(tokens=engine_prompt["prompt_token_ids"]).model_dump_json()
            )
            decision = await router_generator.__anext__()
            decision = decision.data()
            worker_id, prefix_hit_rate = decision.split("_")
            prefix_hit_rate = float(prefix_hit_rate)
            logger.info(
                f"Worker ID: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
            )

            if worker_id == "":
                engine_generator = await self.worker_client.generate(
                    vLLMGenerateRequest(
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        prefix_hit_rate=prefix_hit_rate,
                    ).model_dump_json()
                )
            else:
                engine_generator = await self.worker_client.direct(
                    vLLMGenerateRequest(
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        prefix_hit_rate=prefix_hit_rate,
                    ).model_dump_json(),
                    int(worker_id),
                )
        elif router_mode == RouterType.RANDOM:
            engine_generator = await self.worker_client.generate(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json()
            )
        elif router_mode == RouterType.ROUND_ROBIN:
            engine_generator = await self.worker_client.round_robin(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json()
            )
        elif router_mode == RouterType.KV_LOAD:
            # route to worker with least kv load
            # TODO: move the router to a separate file and clean up processor.py
            try:
                kv_load = await self._get_kv_load()
                best_worker_id = min(kv_load, key=kv_load.get)
                logger.info(f"Routing to worker {best_worker_id} (kv load: {kv_load})")
                engine_generator = await self.worker_client.direct(
                    vLLMGenerateRequest(
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                    ).model_dump_json(),
                    int(best_worker_id),
                )
            except Exception as e:
                logger.info(
                    f"Error finding worker with least kv load: {e}, fallback to random"
                )
                engine_generator = await self.worker_client.generate(
                    vLLMGenerateRequest(
                        engine_prompt=engine_prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                    ).model_dump_json()
                )
        output = self._generate_responses(engine_generator, request_type)

        async for response in await self._stream_response(
            request, output, request_id, conversation
        ):
            yield response

    async def _generate_responses(
        self, engine_generator: AsyncIterator[RequestOutput], request_type: RequestType
    ) -> AsyncIterator[Union[RequestOutput, Tuple[int, RequestOutput]]]:
        prompt_idx = 0
        async for resp in engine_generator:
            # Deserialize the response from the engine
            # Creates correct vLLM objects for each field
            output = MyRequestOutput.model_validate_json(resp.data())

            # OpenAIServingChat.chat_completion_stream_generator() method expects a RequestOutput object
            request_output = RequestOutput(
                request_id=output.request_id,
                prompt=output.prompt,
                prompt_token_ids=output.prompt_token_ids,
                prompt_logprobs=output.prompt_logprobs,
                outputs=output.outputs,
                finished=output.finished,
                metrics=output.metrics,
            )

            if request_type == RequestType.CHAT:
                # For chat requests, yield the request_output directly.
                yield request_output
            elif request_type == RequestType.COMPLETION:
                # Completion requests can have multiple prompts and stream generator requires the prompt index
                yield (prompt_idx, request_output)
            else:
                raise NotImplementedError(
                    f"Request type {request_type} not implemented"
                )

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        async for response in self._generate(raw_request, RequestType.CHAT):
            yield response

    # @dynamo_endpoint()
    # async def completions(self, raw_request: CompletionRequest):
    #     async for response in self._generate(raw_request, RequestType.COMPLETION):
    #         yield response
