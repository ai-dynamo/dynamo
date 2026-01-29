# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

# Import shared memory monitoring
import sys

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from memory_monitor import create_monitor, setup_background_monitor

# Create monitor if profiling is enabled
monitor = create_monitor("BACKEND")

uvloop.install()


class RequestHandler:
    def __init__(self, proxy_client=None):
        print(
            f"Initialized backend request handler {'proxy' if proxy_client else 'worker'} with memory monitoring"
        )
        if monitor:
            monitor.log_memory("Initial:")
        self.proxy_client = proxy_client

    async def generate(self, request, context):
        if monitor:
            monitor.increment_request()

        max_tokens = request.get("max_tokens", 10)
        if self.proxy_client:
            print(f"Proxying request {context.id()}")
            stream = await self.proxy_client.random(request)
            i = 0
            async for chunk in stream:
                i += 1
                yield f"chunk{i}"
        else:
            print(f"Generating mock response {context.id()}")
            for i in range(max_tokens):
                await asyncio.sleep(1e-5)  # yield / switch context
                yield f"chunk{i}"


@dynamo_worker()
async def worker(runtime: DistributedRuntime, proxy_mode: bool = False):
    name = "worker" if not proxy_mode else "proxy_worker"
    component = runtime.namespace("openai/pipeline").component(name)
    # await component.create_service()

    endpoint = component.endpoint("generate")

    # Setup background memory monitoring
    monitor_task = setup_background_monitor(monitor)
    # Create pipeline client if in proxy mode
    if proxy_mode:
        proxy_client = (
            await runtime.namespace("openai/pipeline")
            .component("worker")
            .endpoint("generate")
            .client()
        )
    else:
        proxy_client = None

    try:
        await endpoint.serve_endpoint(RequestHandler(proxy_client).generate)
    finally:
        if monitor:
            print("\nShutdown - Final memory state:")
            monitor.log_memory("Final:")
        if monitor_task:
            monitor_task.cancel()
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-mode", action="store_true", help="Run in proxy mode")
    args = parser.parse_args()
    asyncio.run(worker(proxy_mode=args.proxy_mode))
