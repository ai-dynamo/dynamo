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
from typing import AsyncGenerator, AsyncIterator

import uvloop
from common.parser import parse_vllm_args
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker


class VllmEngine:
    def __init__(
        self, engine_client: AsyncIterator[EngineClient], model_config: ModelConfig
    ):
        self.engine_client = engine_client
        self.model_config = model_config

        # Ensure served_model_name matches the openai model name
        # Use --served-model-name to explicitly set this or it will fallback to --model
        models = OpenAIServingModels(
            engine_client=engine_client,
            model_config=model_config,
            base_model_paths=[
                BaseModelPath(
                    name=model_config.served_model_name,
                    model_path=model_config.model,
                )
            ],
        )

        self.chat_serving = OpenAIServingChat(
            engine_client=self.engine_client,
            model_config=self.model_config,
            models=models,
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        self.completion_serving = OpenAIServingCompletion(
            engine_client=self.engine_client,
            model_config=self.model_config,
            models=models,
            request_logger=None,
        )

    @dynamo_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate_chat(self, request):
        result = await self.chat_serving.create_chat_completion(request)

        if isinstance(result, AsyncGenerator):
            async for raw_response in result:
                if raw_response.startswith("data: [DONE]"):
                    break
                response = json.loads(raw_response.lstrip("data: "))
                yield response

        # We should always be streaming so should never get here
        elif isinstance(result, ChatCompletionResponse):
            raise RuntimeError("ChatCompletionResponse support not implemented")

        elif isinstance(result, ErrorResponse):
            error = result.dict()
            raise RuntimeError(
                f"Error {error['code']}: {error['message']} "
                f"(type: {error['type']}, param: {error['param']})"
            )

        else:
            raise TypeError(f"Unexpected response type: {type(result)}")

    @dynamo_endpoint(CompletionRequest, CompletionStreamResponse)
    async def generate_completions(self, request):
        result = await self.completion_serving.create_completion(request)

        if isinstance(result, AsyncGenerator):
            async for raw_response in result:
                if raw_response.startswith("data: [DONE]"):
                    break
                response = json.loads(raw_response.lstrip("data: "))
                yield response

        # We should always be streaming so should never get here
        elif isinstance(result, CompletionResponse):
            raise RuntimeError("CompletionResponse support not implemented")

        elif isinstance(result, ErrorResponse):
            error = result.dict()
            raise RuntimeError(
                f"Error {error['code']}: {error['message']} "
                f"(type: {error['type']}, param: {error['param']})"
            )

        else:
            raise TypeError(f"Unexpected response type: {type(result)}")


@dynamo_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("dynamo").component("vllm")
    await component.create_service()

    chat_endpoint = component.endpoint("chat/completions")
    completions_endpoint = component.endpoint("completions")

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
        model_config = await engine_client.get_model_config()
        engine = VllmEngine(engine_client, model_config)

        await asyncio.gather(
            chat_endpoint.serve_endpoint(engine.generate_chat),
            completions_endpoint.serve_endpoint(engine.generate_completions),
        )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
