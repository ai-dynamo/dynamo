# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)
configure_dynamo_logging(service_name="backend")


def get_word_delay() -> float:
    """Return the per-word response delay, preserving the example default."""
    return float(os.environ.get("DYN_TEST_HELLO_WORLD_WORD_DELAY_SEC", "1.0"))


@dynamo_endpoint(str, str)
async def content_generator(request: str):
    logger.info(f"Received request: {request}")
    word_delay = get_word_delay()
    for word in request.split(","):
        await asyncio.sleep(word_delay)
        yield f"Hello {word}!"


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    namespace_name = os.environ.get("DYN_TEST_HELLO_WORLD_NAMESPACE", "hello_world")
    component_name = "backend"
    endpoint_name = "generate"

    endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.{endpoint_name}")

    logger.info(f"Serving endpoint {namespace_name}/{component_name}/{endpoint_name}")
    await endpoint.serve_endpoint(content_generator)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
