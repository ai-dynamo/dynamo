# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example cli using the Python bindings, similar to `dynamo-run`.
# Usage: `python cli.py in=text out=mistralrs <your-model>`.
# Must be in a virtualenv with the Dynamo bindings (or wheel) installed.

import argparse
import asyncio
import signal
import sys
from pathlib import Path

import uvloop

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime

subprocess_ref = None  # Global process reference for cleanup
subprocess_task = None  # Global async task reference for cleanup


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
        description="Dynamo CLI: Connect inputs to an engine",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve multi-line help formatting
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

    # TODO: Not yet used here
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Tensor parallel size for the model (e.g., 4).",
    )

    # Add the positional model argument.
    # It's made optional (nargs='?') because its requirement depends on 'out_mode',
    # which is handled in post-parsing validation.
    parser.add_argument(
        "model",
        nargs="?",  # Make it optional for argparse, we'll validate manually
        help="Path to the model (e.g., Qwen/Qwen3-0.6B).\n" "Required unless out=dyn.",
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


async def cleanup_subprocess_async():
    """Clean up the sglang/vllm/trtllm subprocess if it exists."""
    global subprocess_ref
    if subprocess_ref and subprocess_ref.returncode is None:
        subprocess_ref.terminate()
        try:
            await asyncio.wait_for(subprocess_ref.wait(), timeout=2)
        except asyncio.TimeoutError:
            subprocess_ref.kill()
            await subprocess_ref.wait()

        # Only cleanup once
        subprocess_ref = None


def signal_handler():
    """Handle signals in async context by cleaning up subprocess and exiting."""
    asyncio.create_task(cleanup_subprocess_async())
    sys.exit(0)


async def run():
    global subprocess_ref
    global subprocess_task

    # Register signal handlers
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)  # Ctrl-C
    loop.add_signal_handler(signal.SIGTERM, signal_handler)  # kill

    # If we find cases where subprocess does not stop we may need this. Seem OK so far.
    # atexit.register(cleanup_subprocess)

    runtime = DistributedRuntime(loop, False)

    args = parse_args()

    engine_type_map = {
        "echo": EngineType.Echo,
        "mistralrs": EngineType.MistralRs,
        "llamacpp": EngineType.LlamaCpp,
        "dyn": EngineType.Dynamic,
    }
    out_mode = args["out_mode"]

    # Handle subprocess execution for sglang and vllm
    if out_mode in ["sglang", "vllm", "trtllm"]:
        # Determine which script to run
        script_name = f"{out_mode}_inc.py"
        script_path = Path(__file__).parent / script_name
        if not script_path.exists():
            print(f"Error: Script '{script_path}' not found")
            sys.exit(1)

        # Build command with all relevant arguments
        cmd = [sys.executable, str(script_path)]

        # Add arguments if they exist
        if args["model_path"]:
            cmd.extend(["--model-path", args["model_path"]])

        flags = args["flags"]
        if flags.model_name:
            cmd.extend(["--model-name", flags.model_name])
        if flags.context_length:
            cmd.extend(["--context-length", str(flags.context_length)])
        if flags.kv_cache_block_size:
            cmd.extend(["--kv-cache-block-size", str(flags.kv_cache_block_size)])

        # Start subprocess in background and stream output
        print(f"Starting {out_mode} subprocess: {' '.join(cmd)}")

        async def stream_subprocess_output():
            global subprocess_ref
            subprocess_ref = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )

            try:
                if subprocess_ref.stdout is not None:
                    async for line in subprocess_ref.stdout:
                        print(f"Engine: {line.decode().rstrip()}")
                    await subprocess_ref.wait()
            except asyncio.CancelledError:
                # Task was cancelled, terminate the subprocess
                await cleanup_subprocess_async()
                raise

        task = asyncio.create_task(stream_subprocess_output())

        # Store the task reference for potential cleanup
        subprocess_task = task

        # Set out_mode to "dyn" because we talk to the subprocess over NATS
        out_mode = "dyn"

    engine_type = engine_type_map.get(out_mode)
    if engine_type is None:
        print(f"Unsupported output type: {out_mode}")
        sys.exit(1)

    entrypoint_kwargs = {"model_path": args["model_path"]}

    flags = args["flags"]
    if flags.model_name is not None:
        entrypoint_kwargs["model_name"] = flags.model_name
    if flags.model_config is not None:
        entrypoint_kwargs["model_config"] = flags.model_config
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

    try:
        await run_input(runtime, args["in_mode"], engine)
    finally:
        # Clean up subprocess when main execution finishes
        await cleanup_subprocess_async()

        # Cancel the subprocess task if it exists
        if subprocess_task:
            subprocess_task.cancel()
            try:
                await subprocess_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    uvloop.run(run())
