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

import logging

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from utils.protocol import DisaggPreprocessedRequest
from utils.args import parse_sglang_args

from dynamo.sdk import dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1},
    workers=1,
)
class SGLangDecodeWorker:
    def __init__(self, engine_args: ServerArgs):
        self.engine_args = engine_args
        self.engine = sgl.Engine(server_args=self.engine_args)

        logger.info("Decode worker initialized")

    @dynamo_endpoint()
    async def generate(self, req: DisaggPreprocessedRequest):
        g = await self.engine.async_generate(
            input_ids=req.request.token_ids,
            sampling_params=req.sampling_params,
            stream=True,
            bootstrap_host=req.bootstrap_host,
            bootstrap_port=req.bootstrap_port,
            bootstrap_room=req.bootstrap_room,
        )

        async for result in g:
            yield result

if __name__ == "__main__":
    import asyncio
    import uvloop
    from dynamo.runtime import DistributedRuntime, dynamo_worker

    engine_args: ServerArgs = parse_sglang_args()

    @dynamo_worker()
    async def worker(runtime: DistributedRuntime, engine_args):
        worker = SGLangDecodeWorker(engine_args)

        component = runtime.namespace("dynamo").component(SGLangDecodeWorker.__name__)
        await component.create_service()

        # serve endpoint
        endpoint = component.endpoint("generate")
        await endpoint.serve_endpoint(worker.generate)


    uvloop.install()
    asyncio.run(worker(engine_args))