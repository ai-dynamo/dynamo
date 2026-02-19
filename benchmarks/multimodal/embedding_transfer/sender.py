# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import torch
import uvloop
from pydantic import BaseModel

from dynamo.common.multimodal.embedding_transfer import LocalEmbeddingSender
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging()


class SenderConfig(BaseModel):
    num_requests: int
    tensor_count_per_request: int


class Sender:
    def __init__(self, runtime: DistributedRuntime):
        self.runtime = runtime
        self.sender = LocalEmbeddingSender()
        self.tensor = torch.randn([256, 8 * 1024], dtype=torch.float16)
        self.config = SenderConfig(num_requests=100, tensor_count_per_request=30)

    async def async_init(self):
        self.receiver_endpoint = (
            self.runtime.namespace("embedding_transfer")
            .component("receiver")
            .endpoint("generate")
        )
        self.client = await self.receiver_endpoint.client()
        await self.client.wait_for_instances()

    async def send_request(self):
        request = []
        futures = []
        for _ in range(self.config.tensor_count_per_request):
            transfer_request, send_future = await self.sender.send_embeddings(
                self.tensor, stage_embeddings=True
            )
            request.append(transfer_request)
            futures.append(send_future)
        stream = await self.client.generate(request)
        async for response in stream:
            continue
        await asyncio.gather(*futures)

    async def generate(self, request: str):
        tasks = [
            asyncio.create_task(self.send_request())
            for _ in range(self.config.num_requests)
        ]
        await asyncio.gather(*tasks)
        yield "done"


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace_name = "embedding_transfer"
    component_name = "sender"
    endpoint_name = "generate"
    sender = Sender(runtime)
    await sender.async_init()

    component = runtime.namespace(namespace_name).component(component_name)

    logger.info(f"Created service {namespace_name}/{component_name}")

    endpoint = component.endpoint(endpoint_name)

    logger.info(f"Serving endpoint {endpoint_name}")
    await endpoint.serve_endpoint(sender.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
