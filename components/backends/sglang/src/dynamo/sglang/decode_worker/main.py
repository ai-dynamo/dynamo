# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import msgspec
import sglang as sgl
import uvloop
from sglang.srt.server_args import ServerArgs

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.common import (
    BaseWorkerHandler,
    graceful_shutdown,
    parse_sglang_args_inc,
    setup_native_endpoints,
)

configure_dynamo_logging()


class DecodeRequestHandler(BaseWorkerHandler):
    def __init__(self, engine: sgl.Engine, server_args: ServerArgs, component):
        super().__init__(engine, server_args, component)
        logging.info("Decode request handler initialized")

    async def generate(self, request: str):
        req = msgspec.json.decode(request, type=dict)

        results = await self.engine.async_generate(
            input_ids=req["request"]["token_ids"]
            if req["request"]["batch_token_ids"] is None
            else req["request"]["batch_token_ids"],
            sampling_params=req["sampling_params"],
            stream=True,
            bootstrap_host=req["bootstrap_host"],
            bootstrap_port=req["bootstrap_port"],
            bootstrap_room=req["bootstrap_room"],
        )
        try:
            async for result in results:
                yield result
        except asyncio.CancelledError:
            raise GeneratorExit(
                "SGLang decode engine was shut down during token generation."
            ) from None


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        # Schedule the shutdown coroutine instead of calling it directly
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    server_args = parse_sglang_args_inc(sys.argv[1:])
    await init(runtime, server_args)


async def init(runtime: DistributedRuntime, server_args: ServerArgs):
    """Initialize decode worker"""

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace("dynamo").component("decode")
    await component.create_service()

    handler = DecodeRequestHandler(engine, server_args, component)

    gen_endpoint = component.endpoint("generate")

    tasks = [gen_endpoint.serve_endpoint(handler.generate, graceful_shutdown=False)]

    tasks.extend(setup_native_endpoints(server_args, component, handler))

    await asyncio.gather(*tasks)


def main():
    uvloop.install()
    asyncio.run(worker())


if __name__ == "__main__":
    main()
