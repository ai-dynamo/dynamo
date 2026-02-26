# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import torch
import uvloop
from pydantic import BaseModel

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingSender,
    NixlEmbeddingSender,
    NixlPersistentEmbeddingSender,
    TransferRequest,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging()


class SenderConfig(BaseModel):
    num_requests: int
    tensor_count_per_request: int


class TransferRequest(BaseModel):
    requests: list[TransferRequest]


class AgentRequest(BaseModel):
    agent_id: str
    agent_metadata: str


class Sender:
    def __init__(self, runtime: DistributedRuntime):
        self.runtime = runtime
        self.local_sender = LocalEmbeddingSender()
        self.read_sender = NixlPersistentEmbeddingSender()
        self.write_sender = NixlEmbeddingSender()
        # GPU tensor to mimic encoder output
        self.tensor = torch.randn([256, 8 * 1024], dtype=torch.float16)
        self.config = SenderConfig(num_requests=100, tensor_count_per_request=30)

    async def async_init(self):
        self.receiver_read_endpoint = self.runtime.endpoint(
            "embedding_transfer.receiver.read"
        )
        self.read_client = await self.receiver_read_endpoint.client()
        # await self.read_client.wait_for_instances()

    async def generate(self, request: str):
        # sender = self.local_sender
        sender = self.read_sender
        request = TransferRequest(requests=[])
        futures = []
        for _ in range(self.config.tensor_count_per_request):
            transfer_request, send_future = await sender.send_embeddings(
                self.tensor, stage_embeddings=True
            )
            request.requests.append(transfer_request)
            futures.append(send_future)
        stream = await self.read_client.round_robin(request.model_dump_json())
        async for response in stream:
            continue
        await asyncio.gather(*futures)
        yield "done"

    async def write(self, request: str):
        sender = self.write_sender

        request = AgentRequest.model_validate_json(request)
        await sender.add_agent(request.agent_id, request.agent_metadata)

        response = TransferRequest(requests=[])
        futures = []
        for _ in range(self.config.tensor_count_per_request):
            transfer_request, send_future = await sender.send_embeddings(
                self.tensor, stage_embeddings=True
            )
            response.requests.append(transfer_request)
            futures.append(send_future)
        yield response.model_dump_json()
        await asyncio.gather(*futures)


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace_name = "embedding_transfer"
    component_name = "sender"
    worker = Sender(runtime)
    await worker.async_init()

    logger.info(f"Created service {namespace_name}/{component_name}")
    logger.info(f"Serving endpoint {namespace_name}.{component_name}.generate")
    logger.info(f"Serving endpoint {namespace_name}.{component_name}.write")

    generate_endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.generate")
    write_endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.write")
    await asyncio.gather(
        *[
            generate_endpoint.serve_endpoint(worker.generate),
            write_endpoint.serve_endpoint(worker.write),
        ]
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
