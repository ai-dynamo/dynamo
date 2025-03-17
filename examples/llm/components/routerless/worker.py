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


import json

import msgspec
from components.routerless.prefill_worker import PrefillWorkerRouterLess
from utils.nixl import NixlMetadataStore
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from dynamo.sdk import (
    async_on_shutdown,
    async_on_start,
    depends,
    dynamo_context,
    dynamo_endpoint,
    service,
)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmWorkerRouterLess:
    prefill_client = depends(PrefillWorkerRouterLess)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.client = None
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        if self.engine_args.remote_prefill:
            if self.engine_args.enable_chunked_prefill is not False:
                print("Chunked prefill is not supported yet, setting to False")
                self.engine_args.enable_chunked_prefill = False

            if self.engine_args.preemption_mode != "swap":
                print("Preemption mode is not supported yet, setting to swap")
                self.engine_args.preemption_mode = "swap"

            if self.engine_args.pipeline_parallel_size != 1:
                print("Pipeline parallel size is not supported yet, setting to 1")
                self.engine_args.pipeline_parallel_size = 1
        self.openai_serving_chat = None
        self.initialized = False
        print("VllmWorkerRouterLess initialized")

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")
        runtime = dynamo_context["runtime"]
        if self.engine_args.remote_prefill:
            metadata = self.engine_client.nixl_metadata
            metadata_store = NixlMetadataStore("dynamo", runtime)
            await metadata_store.put(metadata.engine_id, metadata)

        models = OpenAIServingModels(
            engine_client=self.engine_client,
            model_config=await self.engine_client.get_model_config(),
            base_model_paths=[
                BaseModelPath(
                    name=self.model_name,
                    model_path=self.model_name,
                )
            ],
        )
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine_client,
            model_config=await self.engine_client.get_model_config(),
            models=models,
            request_logger=None,
            response_role="assistant",
            chat_template=None,
            chat_template_content_format="auto",
        )
        self.initialized = True

    @async_on_shutdown
    async def async_shutdown(self):
        if self._engine_context is not None:
            await self._engine_context.__aexit__(None, None, None)
        print("VllmWorkerRouterLess shutting down")

    def get_remote_prefill_request_callback(self):
        async def callback(request: RemotePrefillRequest):
            json_request = msgspec.json.encode(request).decode("utf-8")
            async for _ in self.prefill_client.generate(json_request):
                pass

        return callback

    @dynamo_endpoint()
    async def generate(self, request: ChatCompletionRequest):
        assert self.openai_serving_chat is not None
        request.model = "vllm"
        if self.do_remote_prefill:
            remote_prefill_params = RemotePrefillParams(
                is_remote_prefill=True,
                remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
            )
        else:
            remote_prefill_params = None

        async for raw_response in await self.openai_serving_chat.create_chat_completion(
            request,
            remote_prefill_params=remote_prefill_params,
        ):
            if raw_response.startswith("data: [DONE]"):
                break
            response = json.loads(raw_response.lstrip("data: "))
            yield response
