#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across router benchmark scripts."""

import json
import logging
import os

import numpy as np
from prefix_data_generator.synthesizer import Synthesizer

# Default values
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_SEED = 0
DEFAULT_BLOCK_SIZE = 64
DEFAULT_MOONCAKE_BLOCK_SIZE = 512


def setup_logger(name: str) -> logging.Logger:
    """Setup and return a logger with standard formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def add_common_args(parser):
    """Add common CLI arguments shared across benchmark scripts."""
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Server URL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--use-expected-osl",
        action="store_true",
        help="Pass agent_hints.osl to nvext for router output block tracking",
    )


def add_synthesis_args(parser):
    """Add CLI arguments for trace dataset synthesis, shared across benchmark scripts."""
    parser.add_argument(
        "--output-dir",
        type=str,
        default="real_data_benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default="mooncake_trace.jsonl",
        help="Path to the input mooncake-style trace dataset file",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Number of requests to synthesize (default: use all from input file)",
    )
    parser.add_argument(
        "--speedup-ratio",
        type=float,
        default=1.0,
        help="Factor to speed up request intervals (default: 1.0)",
    )
    parser.add_argument(
        "--prefix-len-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for prefix lengths (default: 1.0)",
    )
    parser.add_argument(
        "--prefix-root-multiplier",
        type=int,
        default=1,
        help="Number of times to replicate the core radix tree (default: 1)",
    )
    parser.add_argument(
        "--prompt-len-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for leaf path lengths (default: 1.0, use <1 for shorter prompts)",
    )
    parser.add_argument(
        "--max-isl",
        type=int,
        default=None,
        help="Maximum input sequence length to include in output (default: None, no filtering)",
    )
    parser.add_argument(
        "--min-isl",
        type=int,
        default=None,
        help="Minimum input sequence length to include in output (default: None, no filtering)",
    )
    parser.add_argument(
        "--min-osl",
        type=int,
        default=None,
        help="Minimum output sequence length - clips values below this threshold (default: None, no clipping)",
    )
    parser.add_argument(
        "--max-osl",
        type=int,
        default=None,
        help="Maximum output sequence length - clips values above this threshold (default: None, no clipping)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_MOONCAKE_BLOCK_SIZE,
        help=f"Block size for prefilling and decoding (default: {DEFAULT_MOONCAKE_BLOCK_SIZE})",
    )


def resolve_tokenizer(args):
    """Set tokenizer to model if not specified."""
    if args.tokenizer is None:
        args.tokenizer = args.model


def get_common_aiperf_flags():
    """Return common aiperf flags used across benchmarks."""
    return [
        "--endpoint-type",
        "chat",
        "--endpoint",
        "v1/chat/completions",
        "--streaming",
        "--extra-inputs",
        "ignore_eos:true",
        "--no-gpu-telemetry",
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]


def get_aiperf_cmd_for_trace(
    model,
    tokenizer,
    input_dataset,
    artifact_dir,
    seed,
    block_size,
    url="http://localhost:8888",
):
    """Build the aiperf CLI command for a mooncake trace run."""
    cmd = [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--url",
        url,
        "--input-file",
        f"{input_dataset}",
        "--custom-dataset-type",
        "mooncake_trace",
        "--fixed-schedule",
        "--fixed-schedule-auto-offset",
        "--prompt-input-tokens-block-size",
        str(block_size),
        "--random-seed",
        str(seed),
        "--artifact-dir",
        artifact_dir,
    ]
    cmd.extend(get_common_aiperf_flags())
    return cmd


def prepare_trace_dataset(args, output_dir, logger):
    """Prepare a trace dataset, optionally synthesizing or modifying it.

    Handles three paths:
    1. No synthesis needed: use the original dataset as-is
    2. Expected OSL injection only: inject agent_hints.osl into nvext
    3. Full synthesis: generate synthetic data from the input dataset

    Returns:
        tuple[list[dict], str]: (list of request dicts, path to the trace file)
    """
    needs_synthesis = (
        args.num_requests is not None
        or args.speedup_ratio != 1.0
        or args.prefix_len_multiplier != 1.0
        or args.prefix_root_multiplier != 1
        or args.prompt_len_multiplier != 1.0
        or args.max_isl is not None
        or args.min_isl is not None
        or args.min_osl is not None
        or args.max_osl is not None
    )

    if not needs_synthesis and not args.use_expected_osl:
        # No synthesis or modification needed, use original dataset
        trace_dataset_path = args.input_dataset
        logger.info(
            f"Using original trace dataset (no synthesis parameters modified): {trace_dataset_path}"
        )
        requests = []
        with open(args.input_dataset, "r") as f:
            for line in f:
                requests.append(json.loads(line.strip()))
        return requests, trace_dataset_path

    if not needs_synthesis and args.use_expected_osl:
        # Only inject agent_hints.osl into nvext, no other synthesis
        logger.info("Injecting agent_hints.osl into original trace dataset...")

        requests = []
        with open(args.input_dataset, "r") as f:
            for line in f:
                requests.append(json.loads(line.strip()))

        for request in requests:
            osl = request.get("output_tokens", 0)
            if "nvext" not in request:
                request["nvext"] = {}
            request["nvext"].setdefault("agent_hints", {})["osl"] = osl

        trace_dataset_path = os.path.join(output_dir, "trace_with_expected_osl.jsonl")
        with open(trace_dataset_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        logger.info(f"Modified trace data saved to: {trace_dataset_path}")
        return requests, trace_dataset_path

    # Generate synthetic data based on input dataset
    logger.info("Generating synthetic trace data...")
    logger.info(f"  Base dataset: {args.input_dataset}")
    logger.info(f"  Num requests: {args.num_requests if args.num_requests else 'all'}")
    logger.info(f"  Speedup ratio: {args.speedup_ratio}")
    logger.info(f"  Prefix len multiplier: {args.prefix_len_multiplier}")
    logger.info(f"  Prefix root multiplier: {args.prefix_root_multiplier}")
    logger.info(f"  Prompt len multiplier: {args.prompt_len_multiplier}")
    logger.info(
        f"  Max ISL: {args.max_isl if args.max_isl else 'no limit'} (filtering)"
    )
    logger.info(
        f"  Min ISL: {args.min_isl if args.min_isl else 'no limit'} (filtering)"
    )
    logger.info(
        f"  Min OSL: {args.min_osl if args.min_osl else 'no clipping'} (clipping)"
    )
    logger.info(
        f"  Max OSL: {args.max_osl if args.max_osl else 'no clipping'} (clipping)"
    )
    logger.info(f"  Random seed: {args.seed}")

    np.random.seed(args.seed)

    synthesizer = Synthesizer(
        args.input_dataset,
        block_size=args.block_size,
        speedup_ratio=args.speedup_ratio,
        prefix_len_multiplier=args.prefix_len_multiplier,
        prefix_root_multiplier=args.prefix_root_multiplier,
        prompt_len_multiplier=args.prompt_len_multiplier,
    )

    if args.num_requests is None:
        with open(args.input_dataset, "r") as f:
            num_requests = sum(1 for _ in f)
        logger.info(f"Using all {num_requests} requests from input dataset")
    else:
        num_requests = args.num_requests

    requests = synthesizer.synthesize_requests(
        num_requests,
        max_isl=args.max_isl,
        min_isl=args.min_isl,
        min_osl=args.min_osl,
        max_osl=args.max_osl,
    )
    logger.info(f"Generated {len(requests)} synthetic requests")

    trace_dataset_path = os.path.join(output_dir, "synthetic_trace.jsonl")

    if args.use_expected_osl:
        for request in requests:
            osl = request.get("output_tokens", 0)
            if "nvext" not in request:
                request["nvext"] = {}
            request["nvext"].setdefault("agent_hints", {})["osl"] = osl
        logger.info("Injected agent_hints.osl into nvext for each request")

    with open(trace_dataset_path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")

    logger.info(f"Synthetic trace data saved to: {trace_dataset_path}")
    return requests, trace_dataset_path
