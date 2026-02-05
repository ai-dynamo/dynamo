#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Agent benchmark script for running concurrency-based benchmarks with multi-turn
conversation traces. Uses aiperf with concurrency mode to run multiple sessions
in parallel while maintaining sequential ordering within each session.

Expected input JSON format (JSONL - one JSON object per line):
{
    "session_id": "conv_0",           # Groups turns into conversations (required for multi-turn)
    "input_length": 9176,             # Number of input tokens (required)
    "output_length": 152,             # Number of output tokens (required)
    "hash_ids": [0, 1, 2, ...],       # List of hash IDs for prefix caching (optional)
    "delay": 500                      # Delay in ms before this turn (optional, applied after previous turn completes)
}
"""

import argparse
import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_aiperf_cmd(
    model,
    tokenizer,
    input_dataset,
    artifact_dir,
    seed,
    concurrency,
    url="http://localhost:8000",
):
    """Build aiperf command for concurrency-based trace benchmarking."""
    cmd = [
        "aiperf",
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
        input_dataset,
        "--custom-dataset-type",
        "mooncake_trace",
        "--concurrency",
        str(concurrency),
        "--random-seed",
        str(seed),
        "--artifact-dir",
        artifact_dir,
        "--no-gpu-telemetry",
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]
    return cmd


def prepare_dataset(input_dataset, output_path, delay_override=None):
    """
    Prepare the dataset, optionally overriding delay values.

    Args:
        input_dataset: Path to input JSONL file
        output_path: Path to write modified dataset
        delay_override: If set, override all delay values with this value (in ms).
                       Use 0 to remove delays entirely.

    Returns:
        Path to the dataset to use (original or modified)
    """
    if delay_override is None:
        return input_dataset

    logger.info(f"Overriding delay values with: {delay_override}ms")

    requests = []
    with open(input_dataset, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))

    # Track sessions to know which entries are first turns (no delay on first turn)
    session_first_seen = set()

    for request in requests:
        session_id = request.get("session_id")

        if session_id is not None and session_id not in session_first_seen:
            # First turn of a session - remove delay if present
            session_first_seen.add(session_id)
            request.pop("delay", None)
        elif delay_override == 0:
            # Remove delay entirely
            request.pop("delay", None)
        else:
            # Override delay for subsequent turns
            request["delay"] = delay_override

    with open(output_path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")

    logger.info(f"Modified dataset saved to: {output_path}")
    return output_path


def run_benchmark(
    model,
    tokenizer,
    trace_dataset,
    artifact_dir,
    url,
    seed,
    concurrency,
):
    """Run aiperf benchmark with concurrency mode."""
    aiperf_cmd = get_aiperf_cmd(
        model,
        tokenizer,
        trace_dataset,
        artifact_dir,
        seed,
        concurrency,
        url,
    )

    logger.info(f"Running aiperf with concurrency={concurrency}")
    logger.info(f"Dataset: {trace_dataset}")
    logger.info(f"Command: {' '.join(aiperf_cmd)}")

    try:
        subprocess.run(aiperf_cmd, check=True)
        logger.info("AIPerf profiling completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"AIPerf failed with error code: {e.returncode}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run concurrency-based benchmark with multi-turn conversation traces"
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
        default="http://localhost:8000",
        help="Server URL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="agent_benchmark_results",
        help="Output directory for results",
    )

    # Dataset configuration
    parser.add_argument(
        "--input-dataset",
        type=str,
        required=True,
        help="Path to the input trace dataset file (JSONL format)",
    )

    # Benchmark configuration
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent sessions to maintain (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=None,
        help="Override delay (ms) between turns within a session. "
        "If not set, uses delay values from the trace file. "
        "Set to 0 to remove all delays.",
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

    # Prepare dataset (apply delay override if specified)
    if args.delay is not None:
        modified_dataset_path = os.path.join(args.output_dir, "modified_trace.jsonl")
        trace_dataset_path = prepare_dataset(
            args.input_dataset, modified_dataset_path, args.delay
        )
    else:
        trace_dataset_path = args.input_dataset
        logger.info(f"Using original trace dataset: {trace_dataset_path}")

    # Run benchmark
    artifact_dir = os.path.join(args.output_dir, "aiperf_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    run_benchmark(
        args.model,
        args.tokenizer,
        trace_dataset_path,
        artifact_dir,
        args.url,
        args.seed,
        args.concurrency,
    )

    logger.info(f"Results saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
