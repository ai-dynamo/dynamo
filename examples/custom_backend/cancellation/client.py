# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple client demonstration of request cancellation using context.stop_generating()
"""

import asyncio
import sys

from dynamo._core import Context, DistributedRuntime


async def demo_cancellation(client):
    """Perform the generation request with cancellation demonstration"""
    # Create context for cancellation control
    context = Context()

    # Start streaming request
    print("Starting streaming request...")
    stream = await client.generate("dummy_request", context=context)

    iteration_count = 0
    async for response in stream:
        number = response.data()
        print(f"Client: Received {number}")

        # Cancel after receiving 3 responses
        if iteration_count >= 2:
            print("Client: Cancelling after 3 responses...")
            context.stop_generating()
            break

        iteration_count += 1

    print("Client: Stream stopped")


async def main():
    """Connect to server and demonstrate cancellation"""
    # Parse command line argument
    use_middle_server = False  # Default to direct connection
    if len(sys.argv) > 1:
        if sys.argv[1] == "--middle":
            use_middle_server = True
        else:
            print("Usage: python3 client.py [--middle]")
            print("  (no flag): Connect directly to backend server (default)")
            print("  --middle: Connect through middle server")
            return

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "file", "nats")

    # Connect to middle server or direct server based on argument
    if use_middle_server:
        endpoint = runtime.namespace("demo").component("middle").endpoint("generate")
        print("Client connecting to middle server...")
    else:
        endpoint = runtime.namespace("demo").component("server").endpoint("generate")
        print("Client connecting directly to backend server...")

    client = await endpoint.client()
    await client.wait_for_instances()

    print(
        f"Client connected to {'middle server' if use_middle_server else 'backend server'}"
    )

    # Perform the generation request with cancellation
    await demo_cancellation(client)

    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
