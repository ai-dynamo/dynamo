# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2: Middle stage - transforms input, calls Stage 3, yields result
"""

import asyncio

from dynamo._core import DistributedRuntime


class Stage2:
    """Middle stage that processes input, calls Stage 3, and yields result"""

    def __init__(self, runtime):
        self.runtime = runtime
        self.stage3_client = None

    async def initialize(self):
        """Connect to Stage 3"""
        endpoint = (
            self.runtime.namespace("pipeline").component("stage3").endpoint("process")
        )
        self.stage3_client = await endpoint.client()
        await self.stage3_client.wait_for_instances()
        print("[Stage2] Connected to Stage3")

    async def process(self, request, context):
        """Receive input, transform, call Stage 3, yield combined result"""
        print("[Stage2] start")
        input_text = request
        transformed = f"{input_text} -> stage2"

        stream = await self.stage3_client.generate(transformed, context=context)
        async for response in stream:
            output_text = response.data()
            yield output_text
        print("[Stage2] end")


async def main():
    """Start Stage 2 server"""
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "file", "tcp")

    handler = Stage2(runtime)
    await handler.initialize()

    component = runtime.namespace("pipeline").component("stage2")
    endpoint = component.endpoint("process")

    print("[Stage2] Server started, waiting for requests...")
    await endpoint.serve_endpoint(handler.process)

    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
