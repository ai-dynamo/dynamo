# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import uvloop
from protocols import AgentRequest, TransferConfig, TransferRequest

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    NixlEmbeddingReceiver,
    NixlPersistentEmbeddingReceiver,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging()


class Receiver:
    def __init__(self, runtime: DistributedRuntime):
        self.runtime = runtime
        self.local_receiver = LocalEmbeddingReceiver()
        self.write_receiver = NixlEmbeddingReceiver()
        self.read_receiver = NixlPersistentEmbeddingReceiver(
            embedding_hidden_size=8 * 1024, max_item_mm_token=1024
        )
        self.config = TransferConfig(
            use_gpu=False, tensor_count_per_request=30, transmitter_type="local"
        )

    def get_run_config(self):
        # Select the variant of sender/receiver based on config
        if self.config.transmitter_type == "local":
            receiver = self.local_receiver
        elif self.config.transmitter_type == "nixl_write":
            receiver = self.write_receiver
        elif self.config.transmitter_type == "nixl_read":
            receiver = self.read_receiver
        # sender size config
        # tensor = self.gpu_tensor if self.config.use_gpu else self.cpu_tensor
        # tensor_count = self.config.tensor_count_per_request
        tensor = None
        tensor_count = None
        return receiver, tensor, tensor_count

    async def async_init(self):
        self.sender_write_endpoint = self.runtime.endpoint(
            "embedding_transfer.sender.write"
        )
        self.send_client = await self.sender_write_endpoint.client()
        # await self.send_client.wait_for_instances()

    async def generate(self, request):
        receiver, _, _ = self.get_run_config()
        # Need to handshake with sender first
        if isinstance(receiver, NixlEmbeddingReceiver):
            id, metadata = receiver.get_agent_metadata()
        else:
            id, metadata = "dummy_id", "dummy_metadata"
        request = AgentRequest(agent_id=id, agent_metadata=metadata)
        stream = await self.send_client.round_robin(request.model_dump_json())
        async for response in stream:
            response = TransferRequest.model_validate_json(response.data())

            tasks = [
                asyncio.create_task(receiver.receive_embeddings(tr))
                for tr in response.requests
            ]
            responses = await asyncio.gather(*tasks)
            for id, _ in responses:
                receiver.release_tensor(id)
            yield "done"

    async def read(self, request):
        receiver, _, _ = self.get_run_config()
        request = TransferRequest.model_validate_json(request)
        tasks = [
            asyncio.create_task(receiver.receive_embeddings(tr))
            for tr in request.requests
        ]
        responses = await asyncio.gather(*tasks)
        for id, _ in responses:
            receiver.release_tensor(id)
        yield "done"

    async def update_config(self, request):
        request = TransferConfig.model_validate_json(request)
        self.config = request
        yield "config updated"


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace_name = "embedding_transfer"
    component_name = "receiver"
    worker = Receiver(runtime)
    await worker.async_init()

    logger.info(f"Created service {namespace_name}/{component_name}")
    logger.info(f"Serving endpoint {namespace_name}.{component_name}.generate")
    logger.info(f"Serving endpoint {namespace_name}.{component_name}.read")
    logger.info(f"Serving endpoint {namespace_name}.{component_name}.update_config")

    generate_endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.generate")
    read_endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.read")
    update_config_endpoint = runtime.endpoint(
        f"{namespace_name}.{component_name}.update_config"
    )
    await asyncio.gather(
        *[
            generate_endpoint.serve_endpoint(worker.generate),
            read_endpoint.serve_endpoint(worker.read),
            update_config_endpoint.serve_endpoint(worker.update_config),
        ]
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
