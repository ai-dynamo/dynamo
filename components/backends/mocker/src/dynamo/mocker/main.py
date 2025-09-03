#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B-Q8_0.gguf --extra-engine-args args.json`

import argparse
import os
from pathlib import Path

import uvloop

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from . import __version__

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"

configure_dynamo_logging()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    args = cmd_line_args()

    # Create engine configuration
    entrypoint_args = EntrypointArgs(
        engine_type=EngineType.Mocker,
        model_path=args.model_path,
        model_name=args.model_name,
        endpoint_id=args.endpoint,
        extra_engine_args=args.extra_engine_args,
    )

    # Create and run the engine
    # NOTE: only supports dyn endpoint for now
    engine_config = await make_engine(runtime, entrypoint_args)
    await run_input(runtime, args.endpoint, engine_config)


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="Mocker engine for testing Dynamo LLM infrastructure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Mocker {__version__}"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model directory or HuggingFace model ID for tokenizer",
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
        default=None,
        help="Model name for API responses (default: mocker-engine)",
    )
    parser.add_argument(
        "--extra-engine-args",
        type=Path,
        help="Path to JSON file with mocker configuration "
        "(num_gpu_blocks, speedup_ratio, etc.)",
    )

    return parser.parse_args()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
