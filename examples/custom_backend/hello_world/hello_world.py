# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="backend")


@dynamo_endpoint(str, str)
async def content_generator(request: str):
    logger.info(f"Received request: {request}")
    for word in request.split(","):
        await asyncio.sleep(1)
        yield f"Hello {word}!"


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace_name = "hello_world"
    component_name = "backend"
    endpoint_name = "generate"

    component = runtime.namespace(namespace_name).component(component_name)

    logger.info(f"Created service {namespace_name}/{component_name}")

    endpoint = component.endpoint(endpoint_name)

    logger.info(f"Serving endpoint {endpoint_name}")
    await endpoint.serve_endpoint(content_generator)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
