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
import logging

from common.chat_processor import ChatProcessorMixin
from common.parser import parse_tensorrt_llm_args
from common.protocol import (
    DynamoTRTLLMChatCompletionRequest,
    DynamoTRTLLMCompletionRequest,
)
from common.utils import RequestType
from components.kv_router import Router
from components.worker import TensorRTLLMWorker

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(ChatProcessorMixin):
    worker = depends(TensorRTLLMWorker)
    router = depends(Router)

    def __init__(
        self,
    ):
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        args, engine_config = parse_tensorrt_llm_args(config_args)
        self.remote_prefill = args.remote_prefill
        self.router_mode = args.router
        self.min_workers = 1
        self.args = args

        super().__init__(engine_config)

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = TensorRTLLMWorker.dynamo_address()  # type: ignore
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )

        if self.args.router == "kv":
            router_ns, router_name = Router.dynamo_address()  # type: ignore
            self.router_client = (
                await runtime.namespace(router_ns)
                .component(router_name)
                .endpoint("generate")
                .client()
            )

        while len(self.worker_client.endpoint_ids()) < self.min_workers:
            logger.info(
                f"Waiting for workers to be ready.\n"
                f" Current: {len(self.worker_client.endpoint_ids())},"
                f" Required: {self.min_workers}"
            )
            await asyncio.sleep(30)

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
        if self.router_mode == "kv":
            router_generator = await self.router_client.generate(
                preprocessed_request.tokens.model_dump_json()
            )
            decision = await router_generator.__anext__()
            decision = decision.data()
            worker_id, prefix_hit_rate = decision.split("_")
            prefix_hit_rate = float(prefix_hit_rate)
            logger.info(
                f"Worker ID: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
            )

        if worker_id == "":
            if self.router_mode == "round-robin":
                self._send_request = self.worker_client.round_robin
            else:
                # fallback to random
                self._send_request = self.worker_client.random

            engine_generator = await self._send_request(
                preprocessed_request.model_dump_json()
            )

        else:
            engine_generator = await self.worker_client.direct(
                preprocessed_request.model_dump_json(), int(worker_id)
            )

        if request_type == RequestType.CHAT:
            async for response in self.chat_processor.postprocess(
                engine_generator,
                raw_request,
                preprocessed_request.conversation,
            ):
                logger.debug(f"[preprocessor] Response: {response}")
                yield json.loads(response)
        else:
            async for response in self.completions_processor.postprocess(
                engine_generator, raw_request
            ):
                logger.debug(f"[preprocessor] Response: {response}")
                yield json.loads(response)

    @dynamo_endpoint(name="chat/completions")
    async def generate_chat(self, raw_request: DynamoTRTLLMChatCompletionRequest):
        # max_tokens is deprecated, however if the max_tokens is provided instead
        # of max_completion_tokens, we will use the value as max_completion_tokens.
        if raw_request.max_tokens is not None:
            if raw_request.max_completion_tokens is None:
                raw_request.max_completion_tokens = raw_request.max_tokens
            else:
                if raw_request.max_tokens != raw_request.max_completion_tokens:
                    raise ValueError(
                        "max_tokens and max_completion_tokens must be the same"
                    )

        # min_tokens isn't currently propagated through the Rust OpenAI HTTP frontend,
        # and ignore_eos is passed through the 'nvext' field, so set both when found.
        if raw_request.nvext:
            ignore_eos = raw_request.nvext.get("ignore_eos")
            raw_request.ignore_eos = ignore_eos
            # If ignore_eos is True, set min_tokens to max_tokens to guarantee
            # the full expected OSL for consistent benchmarking purposes.
            if ignore_eos:
                logger.debug(
                    f"[preprocessor] `ignore_eos` detected, setting `min_tokens` to `max_completion_tokens`: {raw_request.max_completion_tokens}"
                )
                raw_request.min_tokens = raw_request.max_completion_tokens

        async for response in self._generate(raw_request, RequestType.CHAT):
            yield response

    @dynamo_endpoint(name="completions")
    async def completions(self, raw_request: DynamoTRTLLMCompletionRequest):
        async for response in self._generate(raw_request, RequestType.COMPLETION):
            yield response
