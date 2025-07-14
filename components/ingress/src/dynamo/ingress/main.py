#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.ingress [args]`
#
# Start a frontend node. This runs:
# - OpenAI HTTP server.
# - Auto-discovery: Watches etcd for engine/worker registration (via `register_llm`).
# - Pre-processor: Prompt templating and tokenization.
# - Router, defaulting to round-robin (TODO: Add flags to enable KV routing).

import argparse
import asyncio

import uvloop

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamo Frontend: HTTP+Pre-processor+Router",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve multi-line help formatting
    )
    parser.add_argument(
        "--kv-cache-block-size", type=int, help="KV cache block size (u32)."
    )
    parser.add_argument(
        "--http-port", type=int, default=8080, help="HTTP port for the engine (u16)."
    )
    flags = parser.parse_args()

    kwargs = {"http_port": flags.http_port}
    if flags.kv_cache_block_size is not None:
        kwargs["kv_cache_block_size"] = flags.kv_cache_block_size

    return kwargs


async def async_main():
    runtime = DistributedRuntime(asyncio.get_running_loop(), False)
    flags = parse_args()

    # out=dyn
    e = EntrypointArgs(EngineType.Dynamic, **flags)
    engine = await make_engine(runtime, e)

    # in=http
    try:
        await run_input(runtime, "http", engine)
    except asyncio.exceptions.CancelledError:
        pass


def main():
    uvloop.run(async_main())


if __name__ == "__main__":
    main()
