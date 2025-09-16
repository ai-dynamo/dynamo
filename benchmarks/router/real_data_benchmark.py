#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
import subprocess

import numpy as np
from prefix_data_generator.synthesizer import Synthesizer

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_genai_perf_cmd_for_trace(
    model,
    tokenizer,
    input_file,
    artifact_dir,
    seed,
    url="http://localhost:8888",
):
    """Build genai-perf command for trace file input"""
    return [
        "genai-perf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--endpoint-type",
        "chat",
        "--endpoint",
        "v1/chat/completions",
        "--streaming",
        "--url",
        url,
        "--input-file",
        input_file,
        "--random-seed",
        str(seed),
        "--artifact-dir",
        artifact_dir,
        "--",
        "-v",
        "--max-threads",
        "256",
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]


def run_benchmark_with_trace(
    model,
    tokenizer,
    trace_file,
    artifact_dir,
    url,
    seed,
):
    """Run genai-perf benchmark with a trace file"""
    genai_perf_cmd = get_genai_perf_cmd_for_trace(
        model,
        tokenizer,
        trace_file,
        artifact_dir,
        seed,
        url,
    )

    logger.info(f"Running genai-perf with trace file: {trace_file}")
    logger.info(f"Command: {' '.join(genai_perf_cmd)}")

    try:
        # Run genai-perf and let it output directly to terminal
        subprocess.run(genai_perf_cmd, check=True)

        logger.info("Genai-perf profiling completed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"Genai-perf failed with error code: {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark with real or synthesized mooncake-style trace data"
    )

    # Model and server configuration
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
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
        default="http://localhost:8080",
        help="Server URL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="real_data_benchmark_results",
        help="Output directory for results",
    )

    # Trace file and synthesis configuration (similar to synthesizer.py)
    parser.add_argument(
        "--input-file",
        type=str,
        default="mooncake_trace.jsonl",
        help="Path to the input mooncake-style trace file",
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
        "--block-size",
        type=int,
        default=512,
        help="Block size for prefilling and decoding (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    args = parser.parse_args()

    # Use tokenizer from model if not specified
    if args.tokenizer is None:
        args.tokenizer = args.model

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine whether to use original or synthesized data
    # Check if any synthesis parameters are non-default
    needs_synthesis = (
        args.num_requests is not None
        or args.speedup_ratio != 1.0
        or args.prefix_len_multiplier != 1.0
        or args.prefix_root_multiplier != 1
        or args.prompt_len_multiplier != 1.0
        or args.max_isl is not None
    )

    if not needs_synthesis:
        # No synthesis needed, use original file
        trace_file_path = args.input_file
        logger.info(
            f"Using original trace file (no synthesis parameters modified): {trace_file_path}"
        )
    else:
        # Generate synthetic data based on input file
        logger.info("Generating synthetic trace data...")
        logger.info(f"  Base file: {args.input_file}")
        logger.info(
            f"  Num requests: {args.num_requests if args.num_requests else 'all'}"
        )
        logger.info(f"  Speedup ratio: {args.speedup_ratio}")
        logger.info(f"  Prefix len multiplier: {args.prefix_len_multiplier}")
        logger.info(f"  Prefix root multiplier: {args.prefix_root_multiplier}")
        logger.info(f"  Prompt len multiplier: {args.prompt_len_multiplier}")
        logger.info(f"  Max ISL: {args.max_isl if args.max_isl else 'no limit'}")
        logger.info(f"  Random seed: {args.seed}")

        # Set random seed for reproducibility
        np.random.seed(args.seed)

        # Create synthesizer
        synthesizer = Synthesizer(
            args.input_file,
            block_size=args.block_size,
            speedup_ratio=args.speedup_ratio,
            prefix_len_multiplier=args.prefix_len_multiplier,
            prefix_root_multiplier=args.prefix_root_multiplier,
            prompt_len_multiplier=args.prompt_len_multiplier,
        )

        # Determine number of requests
        if args.num_requests is None:
            # Count requests in original file
            with open(args.input_file, "r") as f:
                num_requests = sum(1 for _ in f)
            logger.info(f"Using all {num_requests} requests from input file")
        else:
            num_requests = args.num_requests

        # Generate synthetic requests
        requests = synthesizer.synthesize_requests(num_requests, args.max_isl)
        logger.info(f"Generated {len(requests)} synthetic requests")

        # Save synthetic data to a permanent file in output directory
        synthetic_trace_filename = "synthetic_trace.jsonl"
        trace_file_path = os.path.join(args.output_dir, synthetic_trace_filename)

        # Write synthetic data to file
        with open(trace_file_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        logger.info(f"Synthetic trace data saved to: {trace_file_path}")

    # Run benchmark with the trace file
    artifact_dir = os.path.join(args.output_dir, "genai_perf_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    run_benchmark_with_trace(
        args.model,
        args.tokenizer,
        trace_file_path,
        artifact_dir,
        args.url,
        args.seed,
    )

    logger.info(f"Results saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
