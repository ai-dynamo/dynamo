#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B-Q8_0.gguf --extra-engine-args args.json`

import argparse
import asyncio
from pathlib import Path

import uvloop

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"

configure_dynamo_logging()


async def run_mocker():
    args = parse_args()

    # Create distributed runtime
    distributed_runtime = DistributedRuntime(asyncio.get_running_loop(), False)

    # Create engine configuration
    entrypoint_args = EntrypointArgs(
        engine_type=EngineType.Mocker,
        model_name=args.model_name,
        endpoint_id=args.endpoint,
        http_port=args.http_port,
        extra_engine_args=args.extra_engine_args,
    )

    # Create and run the engine
    # NOTE: only supports dyn endpoint for now
    engine_config = await make_engine(distributed_runtime, entrypoint_args)
    await run_input(distributed_runtime, args.endpoint, engine_config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mocker engine for testing Dynamo LLM infrastructure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--extra-engine-args",
        type=Path,
        help="Path to JSON file with mocker configuration "
        "(num_gpu_blocks, speedup_ratio, etc.)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Dynamo endpoint string (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mocker-engine",
        help="Model name for API responses (default: mocker-engine)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        help="Run as HTTP server on this port (e.g., 8080)",
    )

    return parser.parse_args()


def main():
    uvloop.run(run_mocker())


if __name__ == "__main__":
    main()
