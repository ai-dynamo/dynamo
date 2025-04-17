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

import logging
from typing import Protocol

from components.kv_router import Router
from components.utils import GeneralRequest, GeneralResponse, check_required_workers
from components.worker import DummyWorker

from dynamo._core import Client
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.dependency import DynamoClient

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo-demo",
    },
    workers=1,
)
class Processor(Protocol):
    """
    vLLM pre and post processing
    """

    router: DynamoClient = depends(Router)
    router_mode: str
    min_workers: int
    worker_client: Client

    def __init__(self):
        self.router_mode = "kv"
        self.min_workers = 2

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = DummyWorker.dynamo_address()  # type: ignore
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("worker_generate")
            .client()
        )

        await check_required_workers(
            self.worker_client, self.min_workers, tag="processor"
        )

    async def _generate(
        self,
        raw_request: GeneralRequest,
    ):
        if self.router_mode == "kv":
            async for route_response in self.router.check_hit_rate(raw_request.prompt):
                worker_id, prefix_hit_rate = route_response.split("_")
                prefix_hit_rate = float(prefix_hit_rate)
                print(
                    f"Worker ID: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
                )
                break
            if worker_id == "":
                engine_generator = await self.worker_client.random(
                    raw_request.model_dump_json()
                )
            else:
                engine_generator = await self.worker_client.direct(
                    raw_request.model_dump_json(),
                    int(worker_id),
                )
        elif self.router_mode == "random":
            engine_generator = await self.worker_client.random(
                raw_request.model_dump_json()
            )
        elif self.router_mode == "round-robin":
            engine_generator = await self.worker_client.round_robin(
                raw_request.model_dump_json()
            )

        async for resp in engine_generator:
            yield GeneralResponse.model_validate_json(resp.data())

    @dynamo_endpoint()
    async def processor_generate(self, raw_request: GeneralRequest):
        async for response in self._generate(raw_request):
            yield response.model_dump_json()
