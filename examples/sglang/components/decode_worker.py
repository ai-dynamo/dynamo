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

from __future__ import annotations

import asyncio
import logging
import os
import sys

import msgspec
import sglang as sgl
import uvloop
from sglang.srt.server_args import ServerArgs
from utils.sgl_utils import parse_sglang_args_inc

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


class DecodeRequestHandler:
    def __init__(self, engine: sgl.Engine):
        self.engine = engine
        logging.info("Decode request handler initialized")

    async def generate(self, request: str):
        req = msgspec.json.decode(request, type=dict)

        results = await self.engine.async_generate(
            input_ids=req["request"]["token_ids"]
            if req["request"]["batch_token_ids"] is None
            else req["request"]["batch_token_ids"],
            sampling_params=req["sampling_params"],
            stream=True,
            bootstrap_host=req["bootstrap_host"],
            bootstrap_port=req["bootstrap_port"],
            bootstrap_room=req["bootstrap_room"],
        )

        async for result in results:
            yield result

    async def flush_cache(self, request: dict):
        _ = request  # Suppress mypy unused variable warning
        await self.engine.tokenizer_manager.flush_cache()
        logging.info("Prefill worker cache flushed")
        yield {"status": "success", "message": "Cache flushed"}


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    server_args = parse_sglang_args_inc(sys.argv[1:])
    await init(runtime, server_args)


async def init(runtime: DistributedRuntime, server_args: ServerArgs):
    """Initialize decode worker"""

    engine = sgl.Engine(server_args=server_args)

    handler = DecodeRequestHandler(engine)

    component = runtime.namespace("dynamo").component("decode")
    await component.create_service()

    gen_endpoint = component.endpoint("generate")
    flush_endpoint = component.endpoint("flush_cache")

    tasks = [gen_endpoint.serve_endpoint(handler.generate)]
    
    if os.environ.get("DYN_SGL_HTTP_SERVER"):
        tasks.append(flush_endpoint.serve_endpoint(handler.flush_cache))
        logging.info("SGL engine HTTP server enabled. Adding flush_cache endpoint.")
    
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
