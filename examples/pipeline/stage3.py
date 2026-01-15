# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 3: Final stage of the pipeline - adds suffix and yields result
"""

import asyncio

from dynamo._core import DistributedRuntime


class Stage3:
    """Final stage that processes input and yields the result"""

    async def process(self, request, context):
        """Receive input, transform it, yield result"""
        print("[Stage3] start")
        input_text = request
        output_text = f"{input_text} -> stage3_done"
        yield output_text
        print("[Stage3] end")


async def main():
    """Start Stage 3 server"""
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "file", "tcp")

    component = runtime.namespace("pipeline").component("stage3")
    endpoint = component.endpoint("process")
    handler = Stage3()

    print("[Stage3] Server started, waiting for requests...")
    await endpoint.serve_endpoint(handler.process)

    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
