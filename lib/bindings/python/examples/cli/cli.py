# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example cli using the Python bindings, similar to `dynamo-run`.
#
# Usage: `python cli.py in=text out=echo <your-model>`.
# `in` can be:
# - "http": OpenAI compliant HTTP server
# - "text": Interactive text chat
# - "batch:<file.jsonl>": Run all the prompts in the JSONL file, write out to a jsonl in current dir.
# - "stdin": Allows you to pipe something in: `echo prompt | python cli.py in=stdin out=...`
# - "dyn://name": Connect to nats/etcd and listen for requests from frontend.
#
# `out` can be:
# - "dyn": Run as the frontend node. Auto-discover workers and route traffic to them.
# - "sglang", "vllm", "trtllm", "echo": An LLM worker.
#
# Must be in a virtualenv with the Dynamo bindings (or wheel) installed.

import argparse
import asyncio
import sys
from pathlib import Path

import uvloop

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime


def parse_args():
    in_mode = "text"
    out_mode = "echo"
    batch_file = None  # Specific to in_mode="batch"

    # List to hold arguments that argparse will process (flags and model path)
    argparse_args = []

    # --- Step 1: Manual Pre-parsing for 'in=' and 'out=' ---
    # Iterate through sys.argv[1:] to extract in= and out=
    # and collect remaining arguments for argparse.
    for arg in sys.argv[1:]:
        if arg.startswith("in="):
            in_val = arg[len("in=") :]
            if in_val.startswith("batch:"):
                in_mode = "batch"
                batch_file = in_val[len("batch:") :]
            else:
                in_mode = in_val
        elif arg.startswith("out="):
            out_mode = arg[len("out=") :]
        else:
            # This argument is not 'in=' or 'out=', so it's either a flag or the model path
            argparse_args.append(arg)

    # --- Step 2: Argparse for flags and the model path ---
    parser = argparse.ArgumentParser(
        description="Dynamo example CLI: Connect inputs to an engine",
        usage="python cli.py in=text out=echo <your-model>",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve multi-line help formatting
    )

    # model_name: Option<String>
    parser.add_argument("--model-name", type=str, help="Name of the model to load.")
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

    # Add the positional model argument.
    # It's made optional (nargs='?') because its requirement depends on 'out_mode',
    # which is handled in post-parsing validation.
    parser.add_argument(
        "model",
        nargs="?",  # Make it optional for argparse, we'll validate manually
        help="Path to the model (e.g., Qwen/Qwen3-0.6B).\nRequired unless out=dyn.",
    )

    # Parse the arguments that were not 'in=' or 'out='
    flags = parser.parse_args(argparse_args)

    # --- Step 3: Post-parsing Validation and Final Assignment ---

    # Validate 'batch' mode requires a file path
    if in_mode == "batch" and not batch_file:
        parser.error("Batch mode requires a file path: in=batch:FILE")

    # Validate model path requirement based on 'out_mode'
    if out_mode != "dyn" and flags.model is None:
        parser.error("Model path is required unless out=dyn.")

    # Consolidate all parsed arguments into a dictionary
    parsed_args = {
        "in_mode": in_mode,
        "out_mode": out_mode,
        "batch_file": batch_file,  # Will be None if in_mode is not "batch"
        "model_path": flags.model,
        "flags": flags,
    }

    return parsed_args


async def run():
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "etcd", "nats")

    args = parse_args()

    engine_type_map = {
        "echo": EngineType.Echo,
        "dyn": EngineType.Dynamic,
    }
    out_mode = args["out_mode"]

    engine_type = engine_type_map.get(out_mode)
    if engine_type is None:
        print(f"Unsupported output type: {out_mode}")
        sys.exit(1)

    entrypoint_kwargs = {"model_path": args["model_path"]}

    flags = args["flags"]
    if flags.model_name is not None:
        entrypoint_kwargs["model_name"] = flags.model_name
    if flags.context_length is not None:
        entrypoint_kwargs["context_length"] = flags.context_length
    if flags.template_file is not None:
        entrypoint_kwargs["template_file"] = flags.template_file
    if flags.kv_cache_block_size is not None:
        entrypoint_kwargs["kv_cache_block_size"] = flags.kv_cache_block_size
    if flags.http_port is not None:
        entrypoint_kwargs["http_port"] = flags.http_port

    e = EntrypointArgs(engine_type, **entrypoint_kwargs)
    engine = await make_engine(runtime, e)

    await run_input(runtime, args["in_mode"], engine)


if __name__ == "__main__":
    uvloop.run(run())
