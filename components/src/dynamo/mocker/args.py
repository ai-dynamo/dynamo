#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

from . import __version__

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"
DEFAULT_PREFILL_ENDPOINT = f"dyn://{DYN_NAMESPACE}.prefill.generate"

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
        "is_prefill": getattr(args, "is_prefill_worker", None),
        "is_decode": getattr(args, "is_decode_worker", None),
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


def validate_worker_type_args(args):
    """
    Validate that is_prefill_worker and is_decode_worker are not both True.
    Raises ValueError if validation fails.
    """
    if args.is_prefill_worker and args.is_decode_worker:
        raise ValueError(
            "Cannot specify both --is-prefill-worker and --is-decode-worker. "
            "A worker must be either prefill, decode, or aggregated (neither flag set)."
        )


def parse_args():
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
        default=None,
        help=f"Dynamo endpoint string (default: {DEFAULT_ENDPOINT} for aggregated/decode, {DEFAULT_PREFILL_ENDPOINT} for prefill)",
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

    # Worker type configuration
    parser.add_argument(
        "--is-prefill-worker",
        action="store_true",
        default=False,
        help="Register as Prefill model type instead of Chat+Completions (default: False)",
    )
    parser.add_argument(
        "--is-decode-worker",
        action="store_true",
        default=False,
        help="Mark this as a decode worker which does not publish KV events and skips prefill cost estimation (default: False)",
    )

    args = parser.parse_args()
    validate_worker_type_args(args)

    # Set endpoint default based on worker type if not explicitly provided
    if args.endpoint is None:
        if args.is_prefill_worker:
            args.endpoint = DEFAULT_PREFILL_ENDPOINT
            logger.debug(f"Using default prefill endpoint: {args.endpoint}")
        else:
            args.endpoint = DEFAULT_ENDPOINT
            logger.debug(f"Using default endpoint: {args.endpoint}")

    return args
