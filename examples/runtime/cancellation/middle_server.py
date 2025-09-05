# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Middle server demonstration that proxies requests to backend servers
using round_robin() and passes context for cancellation support
"""

import asyncio

from dynamo._core import DistributedRuntime


class MiddleServer:
    """Middle server that forwards requests to backend servers"""

    def __init__(self, runtime):
        self.runtime = runtime
        self.backend_client = None

    async def initialize(self):
        """Initialize connection to backend servers"""
        # Connect to backend servers
        endpoint = (
            self.runtime.namespace("demo").component("server").endpoint("generate")
        )
        self.backend_client = await endpoint.client()
        await self.backend_client.wait_for_instances()
        print("Middle server: Connected to backend servers")

    async def generate(self, request, context):
        """Forward request to backend using round_robin and pass context"""
        print("Middle server: Received request, forwarding to backend")

        assert self.backend_client is not None, "Did you call initialize()?"

        # Forward request to backend using round_robin with the same context
        # This passes the cancellation context through to the backend
        stream = await self.backend_client.generate(request, context=context)

        # Stream responses back to client
        try:
            async for response in stream:
                data = response.data()
                print(f"Middle server: Forwarding response {data}")
                yield data

        except ValueError as e:
            if str(e) != "Stream ended before generation completed":
                raise
            print("Middle server: Backend stream ended early due to cancellation")


async def main():
    """Start the middle server"""
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, True)

    # Create middle server handler
    handler = MiddleServer(runtime)
    await handler.initialize()

    # Create middle server component
    component = runtime.namespace("demo").component("middle")
    await component.create_service()

    endpoint = component.endpoint("generate")

    print("Middle server started")
    print("Forwarding requests to backend servers...")

    # Serve the endpoint - this blocks until shutdown
    await endpoint.serve_endpoint(handler.generate)

    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
