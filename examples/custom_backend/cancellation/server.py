# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple server demonstration of request cancellation using context.is_stopped()
"""

import asyncio

from dynamo._core import DistributedRuntime


class DemoServer:
    """Simple server that generates numbers and respects cancellation"""

    async def generate(self, request, context):
        """Generate numbers 0-999, checking for cancellation before each yield"""
        for i in range(1000):
            print(f"Server: Processing iteration {i}")

            # Check if client requested cancellation
            if context.is_stopped():
                print(f"Server: Cancelled at iteration {i}")
                raise asyncio.CancelledError

            await asyncio.sleep(0.1)  # Simulate some work
            print(f"Server: Sending iteration {i}")
            yield i


async def main():
    """Start the demo server"""
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "file", "nats")

    # Create server component
    component = runtime.namespace("demo").component("server")
    await component.create_service()

    endpoint = component.endpoint("generate")
    handler = DemoServer()

    print("Demo server started")
    print("Waiting for client connections...")

    # Serve the endpoint - this blocks until shutdown
    await endpoint.serve_endpoint(handler.generate)

    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
