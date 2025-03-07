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

import uvloop
import vllm
from common.base_engine import BaseVllmEngine
from common.parser import parse_vllm_args
from common.protocol import PrefillRequest, PrefillResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker


class VllmPrefillEngine(BaseVllmEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        assert (
            engine_args.kv_transfer_config.is_kv_producer
        ), "Prefill worker must be a KV producer"
        if engine_args.enable_chunked_prefill is not False:
            vllm_logger.info(
                "Chunked prefill is not supported in disaggregated mode, disabling it"
            )
            engine_args.enable_chunked_prefill = False
        super().__init__(engine_args)
        self.kv_transfer_config = engine_args.create_engine_config().kv_transfer_config
        self.kv_rank = self.kv_transfer_config.kv_rank

    @dynamo_endpoint(PrefillRequest, PrefillResponse)
    async def generate(self, request):
        if self.engine_client is None:
            await self.initialize()

        vllm_logger.debug(f"Received prefill request: {request}")
        sampling_params = vllm.sampling_params.SamplingParams(**request.sampling_params)
        if self.engine_client is None:
            raise RuntimeError("Engine client not initialized")
        else:
            async for response in self.engine_client.generate(
                request.prompt, sampling_params, request.request_id
            ):
                vllm_logger.debug(f"Generated response: {response}")
                yield True


@dynamo_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("dynamo").component("prefill")
    await component.create_service()

    async with VllmPrefillEngine(engine_args) as prefill_engine:
        endpoint = component.endpoint("generate")
        await endpoint.serve_endpoint(prefill_engine.generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
