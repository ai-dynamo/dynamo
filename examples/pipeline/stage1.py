# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 1: Entry stage - receives client request, calls Stage 2, yields result
"""

import asyncio

from dynamo._core import DistributedRuntime


class Stage1:
    """Entry stage that receives requests, calls Stage 2, and yields result"""

    def __init__(self, runtime):
        self.runtime = runtime
        self.stage2_client = None

    async def initialize(self):
        """Connect to Stage 2"""
        endpoint = (
            self.runtime.namespace("pipeline").component("stage2").endpoint("process")
        )
        self.stage2_client = await endpoint.client()
        await self.stage2_client.wait_for_instances()
        print("[Stage1] Connected to Stage2")

    async def process(self, request, context):
        """Receive client input, transform, call Stage 2, yield combined result"""
        input_text = request
        print(f"[Stage1] Input: {input_text}")

        # Transform: add our marker
        transformed = f"{input_text} -> stage1"
        print(f"[Stage1] Transformed: {transformed}")

        # Call Stage 2
        stream = await self.stage2_client.process(transformed, context=context)
        async for response in stream:
            output_text = response.data()
            print(f"[Stage1] Output: {output_text}")
            yield output_text


async def main():
    """Start Stage 1 server"""
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "file", "nats")

    handler = Stage1(runtime)
    await handler.initialize()

    component = runtime.namespace("pipeline").component("stage1")
    endpoint = component.endpoint("process")

    print("[Stage1] Server started, waiting for requests...")
    await endpoint.serve_endpoint(handler.process)

    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
