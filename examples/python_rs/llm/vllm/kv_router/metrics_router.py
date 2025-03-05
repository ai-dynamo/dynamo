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
from common.protocol import Request, Response
from vllm.logger import logger as vllm_logger

from dynemo.llm import KvRouter
from dynemo.runtime import DistributedRuntime, triton_endpoint, triton_worker


class Router:
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        router,
        workers_client,
    ):
        self.router = router
        self.workers_client = workers_client

    @triton_endpoint(Request, Response)
    async def generate(self, request):
        lora_id = 0
        worker_id = None
        tokens = [3] * 64
        try:
            worker_id = await self.router.schedule(tokens, lora_id)
        # [NOTE][TODO] Now that the scheduler may return more error messages,
        # now we are catching all exceptions and logging them. Should have
        # catch specific router exceptions once we have dedicated types.
        except Exception as e:
            vllm_logger.info(f"got exception of type {type(e)}: {e}")
            worker_id = None
            vllm_logger.exception(f"Error during worker selection: {e}")

        vllm_logger.info(f"Scheduling to worker_id: {worker_id}")

        if worker_id is None:
            vllm_logger.info("randomly select worker")
            engine_generator = await self.workers_client.random(
                request.model_dump_json()
            )
        else:
            vllm_logger.info(f"directly select worker: {worker_id}")
            engine_generator = await self.workers_client.direct(
                request.model_dump_json(), worker_id
            )

        async for resp in engine_generator:
            resp = resp.data() if hasattr(resp, "data") else resp
            yield resp

    @triton_endpoint(Request, Response)
    async def mock_generate(self, request):
        print(f"Received request: {request}")
        yield "Hello, World!"


ROUTE_SELF = True


@triton_worker()
async def worker(runtime: DistributedRuntime):
    workers_client = (
        await runtime.namespace("dynemo")
        .component("vllm")
        .endpoint("generate")
        .client()
    )

    vllm_logger.info(
        f"Have number of workers ({len(workers_client.endpoint_ids())}) are ready:\n"
        + "\n".join(f"id: {id}" for id in workers_client.endpoint_ids())
    )

    # [TODO] Collect endpoint implementation expects services to provide
    # ForwardPassMetrics as part of stats handling and it will panic if
    # otherwise. This needs to be fixed so that non-providing endpoints will
    # simply be ignored, but before that, we will make sure that the services
    # of the same namespace::component are created via KvMetricsPublisher,
    # if it is also used to create endpoints.
    kv_listener = runtime.namespace("dynemo").component("vllm")
    await kv_listener.create_service()
    router = KvRouter(runtime, kv_listener)
    # i.e. below will cause panic
    # endpoint = kv_listener.endpoint("generate")
    # await endpoint.serve_endpoint(
    #     Router(router, workers_client).mock_generate
    # )

    router_component = runtime.namespace("dynemo").component("frontend")
    await router_component.create_service()

    endpoint = router_component.endpoint("generate")
    await endpoint.serve_endpoint(Router(router, workers_client).generate)


if __name__ == "__main__":
    uvloop.install()

    asyncio.run(worker())
