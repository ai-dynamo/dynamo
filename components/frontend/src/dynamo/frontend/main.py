#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.frontend [args]`
#
# Start a frontend node. This runs:
# - OpenAI HTTP server.
# - Auto-discovery: Watches etcd for engine/worker registration (via `register_llm`).
# - Pre-processor: Prompt templating and tokenization.
# - Router, defaulting to round-robin (TODO: Add flags to enable KV routing).
#
# Pass `--interactive` or `-i` for text chat instead of HTTP server.
#
# For static mode (no etcd auto-discovery):
# - python -m dynamo.frontend --model-name Qwen3-0.6B-Q8_0.gguf --model-path ~/llms/Qwen3-0.6B --static-endpoint dynamo.backend.generate
# Worker example:
# - cd lib/bindings/python/examples/hello_world
# - python server_sglang_static.py

import argparse
import asyncio
import os
import re

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


def validate_static_endpoint(value):
    """Validate that static-endpoint is three words separated by dots."""
    if not re.match(
        r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$",
        value,
    ):
        raise argparse.ArgumentTypeError(
            f"static-endpoint must be three words separated by dots, got: {value}"
        )
    return value


def validate_model_name(value):
    """Validate that model-name is a non-empty string."""
    if not value or not isinstance(value, str) or len(value.strip()) == 0:
        raise argparse.ArgumentTypeError(
            f"model-name must be a non-empty string, got: {value}"
        )
    return value.strip()


def validate_model_path(value):
    """Validate that model-path is a valid directory on disk."""
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError(
            f"model-path must be a valid directory on disk, got: {value}"
        )
    return value


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
    parser.add_argument(
        "--static-endpoint",
        type=validate_static_endpoint,
        help="Static endpoint in format: word.word.word (e.g., dynamo.backend.generate)",
    )
    parser.add_argument(
        "--model-name",
        type=validate_model_name,
        help="Model name as a string (e.g., 'Llama-3.2-1B-Instruct')",
    )
    parser.add_argument(
        "--model-path",
        type=validate_model_path,
        help="Path to model directory on disk (e.g., /tmp/model_cache/lama3.2_1B/)",
    )

    flags = parser.parse_args()

    if flags.static_endpoint and (not flags.model_name or not flags.model_path):
        parser.error("--static-endpoint requires both --model-name and --model-path")

    return flags


async def async_main():
    flags = parse_args()
    is_static = bool(flags.static_endpoint)  # true if the string has a value

    runtime = DistributedRuntime(asyncio.get_running_loop(), is_static)

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

    if flags.static_endpoint:
        kwargs["endpoint_id"] = flags.static_endpoint
    if flags.model_name:
        kwargs["model_name"] = flags.model_name
    if flags.model_path:
        kwargs["model_path"] = flags.model_path

    if is_static:
        # out=dyn://<static_endpoint>
        engine_type = EngineType.Static
    else:
        # out=auto, most common
        engine_type = EngineType.Dynamic
    e = EntrypointArgs(engine_type, **kwargs)
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
