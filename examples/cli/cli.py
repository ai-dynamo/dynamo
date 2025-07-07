# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example cli using the Python bindings.
# Usage: `python cli.py text mistralrs --model-path <your-model>`.
# If `--model-path` not provided defaults to Qwen3 0.6B.
# Must be in a virtualenv with the bindings (or wheel) installed.

import argparse
import asyncio
import sys
from pathlib import Path

import uvloop

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime


def parse_args():
    """
    Parses command-line arguments for the program.
    """
    parser = argparse.ArgumentParser(
        description="Run a Dynamo LLM engine with configurable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default values in help
    )

    # Positional arguments (replacing sys.argv[1] and sys.argv[2])
    parser.add_argument(
        "input_source",
        type=str,
        help="Input source for the engine: 'text', 'http', 'stdin', 'batch:file.jsonl', 'dyn://<name>'",
    )
    parser.add_argument(
        "output_type",
        type=str,
        help="Output type (engine type): 'echo', 'mistralrs', 'llamacpp', 'dyn'",
    )

    # Optional arguments corresponding to EntrypointArgs fields
    # model_path: Option<PathBuf>
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("Qwen/Qwen3-0.6B"),
        help="Path to the model directory.",
    )
    # model_name: Option<String>
    parser.add_argument("--model-name", type=str, help="Name of the model to load.")
    # model_config: Option<PathBuf>
    parser.add_argument(
        "--model-config", type=Path, help="Path to the model configuration file."
    )
    # context_length: Option<u32>
    parser.add_argument(
        "--context-length", type=int, help="Maximum context length for the model (u32)."
    )
    # template_file: Option<PathBuf>
    parser.add_argument(
        "--template-file",
        type=Path,
        help="Path to the template file for text generation.",
    )
    # kv_cache_block_size: Option<u32>
    parser.add_argument(
        "--kv-cache-block-size", type=int, help="KV cache block size (u32)."
    )
    # http_port: Option<u16>
    parser.add_argument("--http-port", type=int, help="HTTP port for the engine (u16).")

    args = parser.parse_args()
    return args


async def run():
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, False)

    args = parse_args()

    input = args.input_source
    output = args.output_type

    engine_type_map = {
        "echo": EngineType.Echo,
        "mistralrs": EngineType.MistralRs,
        "llamacpp": EngineType.LlamaCpp,
        "dyn": EngineType.Dynamic,
    }
    engine_type = engine_type_map.get(output)
    if engine_type is None:
        print(f"Unsupported output type: {output}")
        sys.exit(1)

    # TODO: The "vllm", "sglang" and "trtllm" cases should call Python directly

    entrypoint_kwargs = {"model_path": args.model_path}
    if args.model_name is not None:
        entrypoint_kwargs["model_name"] = args.model_name
    if args.model_config is not None:
        entrypoint_kwargs["model_config"] = args.model_config
    if args.context_length is not None:
        entrypoint_kwargs["context_length"] = args.context_length
    if args.template_file is not None:
        entrypoint_kwargs["template_file"] = args.template_file
    if args.kv_cache_block_size is not None:
        entrypoint_kwargs["kv_cache_block_size"] = args.kv_cache_block_size
    if args.http_port is not None:
        entrypoint_kwargs["http_port"] = args.http_port

    e = EntrypointArgs(engine_type, **entrypoint_kwargs)
    engine = await make_engine(runtime, e)
    await run_input(runtime, input, engine)


if __name__ == "__main__":
    uvloop.run(run())
