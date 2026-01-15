# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Client for the simple pipeline example - calls Stage 1 which flows through the pipeline
"""

import asyncio

from dynamo._core import Context, DistributedRuntime


async def main():
    """Connect to the pipeline and send a request"""
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "file", "nats")

    # Connect to Stage 1 (entry point of the pipeline)
    endpoint = runtime.namespace("pipeline").component("stage1").endpoint("process")
    client = await endpoint.client()
    await client.wait_for_instances()

    print("[Client] Connected to pipeline")

    # Create context and send request
    context = Context()
    input_text = "hello"

    print(f"[Client] Sending: {input_text}")
    stream = await client.process(input_text, context=context)

    # Receive results
    async for response in stream:
        result = response.data()
        print(f"[Client] Received: {result}")

    print("[Client] Done")
    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
