# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import torch
import uvloop
from protocols import TransferConfig, TransferRequest

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingSender,
    NixlPersistentEmbeddingSender,
    NixlWriteEmbeddingSender,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging()


class Sender:
    def __init__(self, runtime: DistributedRuntime):
        self.runtime = runtime
        self.local_sender = LocalEmbeddingSender()
        self.read_sender = NixlPersistentEmbeddingSender()
        self.write_sender = NixlWriteEmbeddingSender()
        # GPU tensor to mimic encoder output
        self.cpu_tensor = torch.randn([256, 8 * 1024], dtype=torch.float16)
        self.gpu_tensor = torch.randn(
            [256, 8 * 1024], dtype=torch.float16, device="cuda"
        )
        self.config = TransferConfig(
            use_gpu=False, tensor_count_per_request=30, transfer_type="local"
        )

    def get_run_config(self):
        # Select the variant of sender/receiver based on config
        if self.config.transfer_type == "local":
            sender = self.local_sender
        elif self.config.transfer_type == "nixl_write":
            sender = self.write_sender
        elif self.config.transfer_type == "nixl_read":
            sender = self.read_sender
        tensor = self.gpu_tensor if self.config.use_gpu else self.cpu_tensor
        tensor_count = self.config.tensor_count_per_request
        return sender, tensor, tensor_count

    async def async_init(self):
        self.receiver_read_endpoint = self.runtime.endpoint(
            "embedding_transfer.receiver.read"
        )
        self.read_client = await self.receiver_read_endpoint.client()
        # await self.read_client.wait_for_instances()

    async def generate(self, request: str):
        # Select the variant of sender/receiver based on config
        sender, tensor, tensor_count = self.get_run_config()

        request = TransferRequest(requests=[])
        futures = []
        for _ in range(tensor_count):
            transfer_request, send_future = await sender.send_embeddings(
                tensor, stage_embeddings=True
            )
            request.requests.append(transfer_request)
            futures.append(send_future)
        stream = await self.read_client.round_robin(request.model_dump_json())
        async for response in stream:
            continue
        await asyncio.gather(*futures)
        yield "done"

    async def write(self, request: str):
        # Select the variant of sender/receiver based on config
        sender, tensor, tensor_count = self.get_run_config()

        response = TransferRequest(requests=[])
        futures = []
        for _ in range(tensor_count):
            transfer_request, send_future = await sender.send_embeddings(
                tensor, stage_embeddings=True
            )
            response.requests.append(transfer_request)
            futures.append(send_future)
        yield response.model_dump_json()
        await asyncio.gather(*futures)

    async def update_config(self, request: str):
        request = TransferConfig.model_validate_json(request)
        self.config = request
        yield "config updated"


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace_name = "embedding_transfer"
    component_name = "sender"
    worker = Sender(runtime)
    await worker.async_init()

    logger.info(f"Created service {namespace_name}/{component_name}")
    logger.info(f"Serving endpoint {namespace_name}.{component_name}.generate")
    logger.info(f"Serving endpoint {namespace_name}.{component_name}.write")
    logger.info(f"Serving endpoint {namespace_name}.{component_name}.update_config")
    generate_endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.generate")
    write_endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.write")
    update_config_endpoint = runtime.endpoint(
        f"{namespace_name}.{component_name}.update_config"
    )
    await asyncio.gather(
        *[
            generate_endpoint.serve_endpoint(worker.generate),
            write_endpoint.serve_endpoint(worker.write),
            update_config_endpoint.serve_endpoint(worker.update_config),
        ]
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
