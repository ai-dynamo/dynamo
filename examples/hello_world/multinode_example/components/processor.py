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
from components.worker import DummyWorker
from components.utils import GeneralRequest, GeneralResponse
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
import socket
from typing import Protocol

@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(Protocol):
    """
    Pre and Post Processing
    """

    worker = depends(DummyWorker)

    def __init__(self):
        config = ServiceConfig.get_instance()
        processor_config = config.get("Processor", {})
        self.hostname = socket.gethostname()
        self.min_workers = processor_config.get("min_worker", 1)
        self.router = processor_config.get("router", "round-robin")

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = DummyWorker.dynamo_address()  # type: ignore
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )
        while len(self.worker_client.endpoint_ids()) < self.min_workers:
            print(
                f"Waiting for workers to be ready.\n"
                f" Current: {len(self.worker_client.endpoint_ids())},"
                f" Required: {self.min_workers}"
            )
            await asyncio.sleep(5)
        print(f"----workers are all ready {self.worker_client.endpoint_ids()}")

    async def _generate(
        self,
        raw_request: GeneralRequest,
    ):
        raw_request.prompt = raw_request.prompt + \
            "_ProcessedBy_" + self.hostname
        if self.router == "random":
            engine_generator = await self.worker_client.random(
                raw_request.model_dump_json()
            )
        elif self.router == "round-robin":
            engine_generator = await self.worker_client.round_robin(
                raw_request.model_dump_json()
            )

        async for resp in engine_generator:
            yield GeneralResponse.model_validate_json(resp.data())

    @dynamo_endpoint()
    async def generate(self, request: GeneralRequest):
        """Forward requests to backend."""
        mid_request = request.model_dump_json()
        print(f"---Middle layer received {mid_request=}")
        async for response in self._generate(request):
            print(f"---Middle layer received response: {response.model_dump_json()}")
            yield response.model_dump_json()

  
