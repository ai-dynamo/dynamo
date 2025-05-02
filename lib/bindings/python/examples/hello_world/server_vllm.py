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

#
# A very basic example of vllm worker handling pre-processed requests.
# Dynamo does the HTTP handling, prompt templating and tokenization, then forwards the
# request via NATS to this python script, which runs vllm.
#
# Setup a virtualenv with dynamo.llm, dynamo.runtime and vllm installed
#  in lib/bindings/python `maturin develop` and `pip install -e .` should do it
# Start nats and etcd:
#  - nats-server -js
#
# Window 1: `python server_vllm.py`. Wait for log "Starting endpoint".
# Window 2: `dynamo-run out=dyn://dynamo.backend.generate`

import asyncio
import logging

import uvloop
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs import TokensPrompt

from dynamo.llm import register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine):
        self.engine_client = engine

    async def generate(self, request):
        print(f"Received request: {request}")
        request_id = "1"
        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])
        sampling_params = SamplingParams(temperature=0.7)
        try:
            gen = self.engine_client.generate(prompt, sampling_params, request_id)
            async for res in gen:
                yield {"token_ids": [res.outputs[0].token_ids[-1]]}
        except Exception as e:
            logging.error(f"vllm generate failed: {e}")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    await init(runtime, "dynamo")


async def init(runtime: DistributedRuntime, ns: str):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace(ns).component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    print("Started server instance")

    await register_llm(endpoint, MODEL, 3)  # 3 is ModelType::Backend

    engine_args = AsyncEngineArgs(
        model=MODEL,
        task="generate",
        skip_tokenizer_init=True,
    )

    engine_context = build_async_engine_client_from_engine_args(engine_args)
    engine_client = await engine_context.__aenter__()

    # the server will gracefully shutdown (i.e., keep opened TCP streams finishes)
    # after the lease is revoked
    await endpoint.serve_endpoint(RequestHandler(engine_client).generate, None)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
