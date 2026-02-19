# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import uvloop
from pydantic import BaseModel

from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    TransferRequest,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging()


class TransferRequest(BaseModel):
    requests: list[TransferRequest]


class Receiver:
    def __init__(self):
        self.receiver = LocalEmbeddingReceiver()

    async def generate(self, request):
        request = TransferRequest.model_validate_json(request)
        tasks = [
            asyncio.create_task(self.receiver.receive_embeddings(tr))
            for tr in request.requests
        ]
        responses = await asyncio.gather(*tasks)
        for id, _ in responses:
            self.receiver.release_tensor(id)
        yield "done"


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace_name = "embedding_transfer"
    component_name = "receiver"
    endpoint_name = "generate"
    worker = Receiver()

    component = runtime.namespace(namespace_name).component(component_name)

    logger.info(f"Created service {namespace_name}/{component_name}")

    endpoint = component.endpoint(endpoint_name)

    logger.info(f"Serving endpoint {endpoint_name}")
    await endpoint.serve_endpoint(worker.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
