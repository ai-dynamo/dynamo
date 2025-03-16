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
from common.protocol import (
    DisaggChatCompletionRequest,
    DisaggChatCompletionStreamResponse,
    DisaggCompletionStreamResponse,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

logger.set_level("debug")


class Router:
    def __init__(
        self,
        ctx_chat_client,
        gen_chat_client,
        ctx_completion_client,
        gen_completion_client,
    ):
        self.ctx_chat_client = ctx_chat_client
        self.gen_chat_client = gen_chat_client
        self.ctx_completion_client = ctx_completion_client
        self.gen_completion_client = gen_completion_client
        logger.info("INITIALIZED ROUTER")

    async def _get_ctx_resp(self, request, ctx_client):
        logger.debug(f"Received request {request}")

        request.max_completion_tokens = 1
        request.disaggregated_params = DisaggregatedParams(request_type="context_only")
        logger.debug(f"[router] Sending request to context server: {request}")
        ctx_resp = [
            resp
            async for resp in await ctx_client.round_robin(request.model_dump_json())
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
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("dynamo").component("router")
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

    completions_endpoint = component.endpoint("completions")
    chat_endpoint = component.endpoint("chat/completions")
    router = Router(
        ctx_chat_client, gen_chat_client, ctx_completion_client, gen_completion_client
    )
    await asyncio.gather(
        completions_endpoint.serve_endpoint(router.generate_completion),
        chat_endpoint.serve_endpoint(router.generate_chat),
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
