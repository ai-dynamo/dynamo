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

import msgspec
import uvloop
from common import parse_vllm_args, temp_metadata_file
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import EngineClient
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)


class RequestHandler:
    def __init__(
        self,
        model_name: str,
        engine_client: EngineClient,
        prefill_client,
        do_remote_prefill: bool,
    ):
        self.model_name = model_name
        self.engine_client = engine_client
        self.prefill_client = prefill_client
        self.openai_serving_chat = None
        self.initialized = False
        self.do_remote_prefill = (
            do_remote_prefill  # TODO: this should be decided by the algorithm
        )
        print("RequestHandler initialized")

    async def init(self):
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

    def get_remote_prefill_request_callback(self):
        async def callback(request: RemotePrefillRequest):
            json_request = msgspec.json.encode(request).decode("utf-8")
            self.prefill_client.generate(json_request)

        return callback

    @triton_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate(self, request):
        if not self.initialized:
            await self.init()
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


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    component = runtime.namespace("test-nixl").component("vllm")
    await component.create_service()

    endpoint = component.endpoint("generate")

    prefill_client = (
        await runtime.namespace("test-nixl")
        .component("prefill")
        .endpoint("generate")
        .client()
    )

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
        # This should be replaced with etcd

        if engine_args.remote_prefill:
            metadata = engine_client.nixl_metadata
            with temp_metadata_file(metadata.engine_id, metadata):
                await endpoint.serve_endpoint(
                    RequestHandler(
                        model_name="vllm",
                        engine_client=engine_client,
                        prefill_client=prefill_client,
                        do_remote_prefill=True,
                    ).generate
                )
        else:
            await endpoint.serve_endpoint(
                RequestHandler(
                    model_name="vllm",
                    engine_client=engine_client,
                    prefill_client=prefill_client,
                    do_remote_prefill=False,
                ).generate
            )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()

    if engine_args.remote_prefill:
        if engine_args.enable_chunked_prefill is not False:
            print("Chunked prefill is not supported yet, setting to False")
            engine_args.enable_chunked_prefill = False

        if engine_args.preemption_mode != "swap":
            print("Preemption mode is not supported yet, setting to swap")
            engine_args.preemption_mode = "swap"

        if engine_args.pipeline_parallel_size != 1:
            print("Pipeline parallel size is not supported yet, setting to 1")
            engine_args.pipeline_parallel_size = 1

    asyncio.run(worker(engine_args))
