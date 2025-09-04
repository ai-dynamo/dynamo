#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Usage: `python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B-Q8_0.gguf`
# Now supports vLLM-style individual arguments for MockEngineArgs

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import uvloop

from dynamo.llm import EngineType, EntrypointArgs, make_engine, run_input
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from . import __version__

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def create_temp_engine_args_file(args) -> Path:
    """
    Create a temporary JSON file with MockEngineArgs from CLI arguments.
    Returns the path to the temporary file.
    """
    engine_args = {}

    # Only include non-None values that differ from defaults
    # Note: argparse converts hyphens to underscores in attribute names
    # Extract all potential engine arguments, using None as default for missing attributes
    engine_args = {
        "num_gpu_blocks": getattr(args, "num_gpu_blocks", None),
        "block_size": getattr(args, "block_size", None),
        "max_num_seqs": getattr(args, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None),
        "enable_prefix_caching": getattr(args, "enable_prefix_caching", None),
        "enable_chunked_prefill": getattr(args, "enable_chunked_prefill", None),
        "watermark": getattr(args, "watermark", None),
        "speedup_ratio": getattr(args, "speedup_ratio", None),
        "dp_size": getattr(args, "dp_size", None),
        "startup_time": getattr(args, "startup_time", None),
    }

    # Remove None values to only include explicitly set arguments
    engine_args = {k: v for k, v in engine_args.items() if v is not None}

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(engine_args, f, indent=2)
        temp_path = Path(f.name)

    logger.debug(f"Created temporary MockEngineArgs file at {temp_path}")
    logger.debug(f"MockEngineArgs: {engine_args}")

    return temp_path


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    args = cmd_line_args()

    # Handle extra_engine_args: either use provided file or create from CLI args
    if args.extra_engine_args:
        # User provided explicit JSON file
        extra_engine_args_path = args.extra_engine_args
        logger.info(f"Using provided MockEngineArgs from {extra_engine_args_path}")
    else:
        # Create temporary JSON file from CLI arguments
        extra_engine_args_path = create_temp_engine_args_file(args)
        logger.info("Created MockEngineArgs from CLI arguments")

    try:
        # Create engine configuration
        entrypoint_args = EntrypointArgs(
            engine_type=EngineType.Mocker,
            model_path=args.model_path,
            model_name=args.model_name,
            endpoint_id=args.endpoint,
            extra_engine_args=extra_engine_args_path,
        )

        # Create and run the engine
        # NOTE: only supports dyn endpoint for now
        engine_config = await make_engine(runtime, entrypoint_args)
        await run_input(runtime, args.endpoint, engine_config)
    finally:
        # Clean up temporary file if we created one
        if not args.extra_engine_args and extra_engine_args_path.exists():
            try:
                extra_engine_args_path.unlink()
                logger.debug(f"Cleaned up temporary file {extra_engine_args_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="Mocker engine for testing Dynamo LLM infrastructure with vLLM-style CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Mocker {__version__}"
    )

    # Basic configuration
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
        help="Model name for API responses (default: derived from model-path)",
    )

    # MockEngineArgs parameters (similar to vLLM style)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        dest="num_gpu_blocks",  # Maps to num_gpu_blocks in MockEngineArgs
        default=None,
        help="Number of GPU blocks for KV cache (default: 16384)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Token block size for KV cache blocks (default: 64)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences per iteration (default: 256)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per iteration (default: 8192)",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        dest="enable_prefix_caching",
        default=None,
        help="Enable automatic prefix caching (default: True)",
    )
    parser.add_argument(
        "--no-enable-prefix-caching",
        action="store_false",
        dest="enable_prefix_caching",
        default=None,
        help="Disable automatic prefix caching",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        dest="enable_chunked_prefill",
        default=None,
        help="Enable chunked prefill (default: True)",
    )
    parser.add_argument(
        "--no-enable-chunked-prefill",
        action="store_false",
        dest="enable_chunked_prefill",
        default=None,
        help="Disable chunked prefill",
    )
    parser.add_argument(
        "--watermark",
        type=float,
        default=None,
        help="Watermark value for the mocker engine (default: 0.01)",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=None,
        help="Speedup ratio for mock execution (default: 1.0)",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        dest="dp_size",
        default=None,
        help="Number of data parallel replicas (default: 1)",
    )
    parser.add_argument(
        "--startup-time",
        type=float,
        default=None,
        help="Simulated engine startup time in seconds (default: None)",
    )

    # Legacy support - allow direct JSON file specification
    parser.add_argument(
        "--extra-engine-args",
        type=Path,
        help="Path to JSON file with mocker configuration. "
        "If provided, overrides individual CLI arguments.",
    )

    return parser.parse_args()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
