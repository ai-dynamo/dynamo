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
import inspect
import uuid
from contextlib import AsyncContextDecorator
from typing import Any

import uvloop
from processor_backend.common import NvAsyncEngineArgs, parse_vllm_args
from vllm import SamplingParams
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.outputs import CompletionOutput

from triton_distributed.runtime import (
    Backend,
    DistributedRuntime,
    ModelDeploymentCard,
    triton_endpoint,
    triton_worker,
)

finish_reason_map = {
    None: None,
    "stop": "stop",
    "abort": "cancelled",
    "length": "length",
    "error": "error",
}


class DeltaState:
    """
    The vLLM AsyncEngine returns the full internal state of each slot per forward pass.
    The OpenAI ChatCompletionResponseDelta object only requires the delta, so this object
    is used to track the state of the last forward pass to calculate the delta.
    """

    def __init__(self):
        self.token_ids = None
        self.last_token_count = 0

    def delta(self, choice):
        self.token_ids = choice.token_ids
        tokens_produced = len(choice.token_ids) - self.last_token_count
        self.last_token_count = len(choice.token_ids)
        return choice.token_ids[-tokens_produced:]


class VllmEngine(AsyncContextDecorator):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: NvAsyncEngineArgs, mdc: ModelDeploymentCard):
        self.mdc = mdc
        self.engine_args = engine_args
        self.engine_client = None
        print("vllm backend started")

    async def __aenter__(self):
        await self.async_init()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        print("vllm backend exited")

    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args, False
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

    def to_backend_output(self, response: CompletionOutput, delta_token_ids: list[int]):
        return {
            "token_ids": delta_token_ids,
            "tokens": [],
            "finish_reason": finish_reason_map.get(response.finish_reason, "stop"),
            "cum_log_probs": response.cumulative_logprob,
            "text": None,
        }

    def to_sampling_params(self, request) -> SamplingParams:
        sampling_params_names = inspect.signature(SamplingParams).parameters.keys()
        sampling_params = {
            k: v
            for k, v in request.get("sampling_options", {}).items()
            if k in sampling_params_names and v is not None
        }
        return SamplingParams(**sampling_params)

    @triton_endpoint(Any, CompletionOutput)
    async def generate(self, request):
        state = DeltaState()
        request_id = str(uuid.uuid4())
        sampling_params = self.to_sampling_params(request)
        inputs = {"prompt_token_ids": request["token_ids"]}
        stream = self.engine_client.generate(
            inputs, sampling_params, request_id=request_id
        )
        async for request_output in stream:
            for choice in request_output.outputs:
                delta_token_ids = state.delta(choice)
                yield self.to_backend_output(choice, delta_token_ids)


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: NvAsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")

    mdc = await ModelDeploymentCard.from_local_path(
        engine_args.model_path, engine_args.model
    )
    async with VllmEngine(engine_args, mdc) as engine:
        backend = Backend(mdc, endpoint)
        await backend.start(engine.generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
