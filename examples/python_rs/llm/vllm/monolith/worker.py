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
import uuid

import uvloop
from common.base_engine import BaseVllmEngine
from common.chat_processor import ProcessMixIn
from common.parser import parse_vllm_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from vllm.logger import logger as vllm_logger

from dynemo.runtime import DistributedRuntime, triton_endpoint, triton_worker


class VllmEngine(BaseVllmEngine, ProcessMixIn):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        super().__init__(engine_args)

    @triton_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate(self, raw_request):
        if self.engine_client is None:
            await self.initialize()

        vllm_logger.debug(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            _,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)
        request_id = str(uuid.uuid4())

        vllm_logger.debug(
            f"Running generate with engine_prompt: {engine_prompt}, sampling_params: {sampling_params}, request_id: {request_id}"
        )
        if self.engine_client is None:
            raise RuntimeError("Engine client not initialized")
        else:
            generator = self.engine_client.generate(
                engine_prompt, sampling_params, request_id
            )

        async for response in await self._stream_response(
            request, generator, request_id, conversation
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield response


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("vllm")
    await component.create_service()

    endpoint = component.endpoint("generate")

    async with VllmEngine(engine_args) as engine:
        await endpoint.serve_endpoint(engine.generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
