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
import sys

import sglang as sgl
import msgspec
import uvloop
from sglang.srt.server_args import ServerArgs
from utils.protocol import DisaggPreprocessedRequest
from utils.sgl_utils import parse_sglang_args_inc

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


class DecodeRequestHandler:
    def __init__(self, engine: sgl.Engine):
        self.engine = engine
        logging.info("Decode request handler initialized")

    async def generate(self, req: str):
        req = msgspec.json.decode(req, type=dict)

        g = await self.engine.async_generate(
            input_ids=req["request"]["token_ids"]
            if req["request"]["batch_token_ids"] is None
            else req["request"]["batch_token_ids"],
            sampling_params=req["sampling_params"],
            stream=True,
            bootstrap_host=req["bootstrap_host"],
            bootstrap_port=req["bootstrap_port"],
            bootstrap_room=req["bootstrap_room"],
        )

        async for result in g:
            yield result


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

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(handler.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
