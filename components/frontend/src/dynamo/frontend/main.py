#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.frontend [args]`
#
# Start a frontend node. This runs:
# - OpenAI HTTP server.
# - Auto-discovery: Watches etcd for engine/worker registration (via `register_llm`).
# - Pre-processor: Prompt templating and tokenization.
# - Router, defaulting to round-robin (TODO: Add flags to enable KV routing).

import argparse
import asyncio

import uvloop

from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    KvRouterConfig,
    RouterConfig,
    RouterMode,
    make_engine,
    run_input,
)
from dynamo.runtime import DistributedRuntime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamo Frontend: HTTP+Pre-processor+Router",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve multi-line help formatting
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive text chat"
    )
    parser.add_argument(
        "--kv-cache-block-size", type=int, help="KV cache block size (u32)."
    )
    parser.add_argument(
        "--http-port", type=int, default=8080, help="HTTP port for the engine (u16)."
    )
    parser.add_argument(
        "--router-mode",
        type=str,
        choices=["round-robin", "random", "kv"],
        default="round-robin",
        help="How to route the request",
    )
    parser.add_argument(
        "--kv-overlap-score-weight",
        type=float,
        default=1.0,
        help="KV Router: Weight for overlap score in worker selection. Higher values prioritize KV cache reuse.",
    )
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=0.0,
        help="KV Router: Temperature for worker sampling via softmax. Higher values promote more randomness, and 0 fallbacks to deterministic.",
    )
    parser.add_argument(
        "--kv-events",
        action="store_true",
        dest="use_kv_events",
        help=" KV Router: Whether to use KV events to maintain the view of cached blocks. If false, would use ApproxKvRouter for predicting block creation / deletion based only on incoming requests at a timer.",
    )
    parser.add_argument(
        "--no-kv-events",
        action="store_false",
        dest="use_kv_events",
        help=" KV Router. Disable KV events.",
    )
    parser.set_defaults(use_kv_events=True)

    return parser.parse_args()


async def async_main():
    runtime = DistributedRuntime(asyncio.get_running_loop(), False)
    flags = parse_args()

    if flags.router_mode == "kv":
        router_mode = RouterMode.KV
        kv_router_config = KvRouterConfig(
            overlap_score_weight=flags.kv_overlap_score_weight,
            router_temperature=flags.router_temperature,
            use_kv_events=flags.use_kv_events,
        )
    elif flags.router_mode == "random":
        router_mode = RouterMode.Random
        kv_router_config = None
    else:
        router_mode = RouterMode.RoundRobin
        kv_router_config = None

    kwargs = {
        "http_port": flags.http_port,
        "kv_cache_block_size": flags.kv_cache_block_size,
        "router_config": RouterConfig(router_mode, kv_router_config),
    }

    # out=dyn
    e = EntrypointArgs(EngineType.Dynamic, **kwargs)
    engine = await make_engine(runtime, e)

    try:
        if flags.interactive:
            await run_input(runtime, "text", engine)
        else:
            await run_input(runtime, "http", engine)
    except asyncio.exceptions.CancelledError:
        pass


def main():
    uvloop.run(async_main())


if __name__ == "__main__":
    main()
