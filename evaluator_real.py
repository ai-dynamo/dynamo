#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
import http.client

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

ttft_base = 0.0
throughput_base = 0.0


def get_aiperf_cmd_for_trace(
    model,
    tokenizer,
    input_dataset,
    artifact_dir,
    seed,
    url="http://localhost:8888",
):
    return [
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
        f"{input_dataset}",
        "--custom-dataset-type",
        "mooncake_trace",
        "--fixed-schedule-auto-offset",
        "--random-seed",
        str(seed),
        "--artifact-dir",
        artifact_dir,
        "-v",
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]


def run_benchmark_with_trace(
    model,
    tokenizer,
    trace_dataset,
    artifact_dir,
    url,
    seed,
):
    """Run aiperf benchmark with a trace dataset"""
    aiperf_cmd = get_aiperf_cmd_for_trace(
        model,
        tokenizer,
        trace_dataset,
        artifact_dir,
        seed,
        url,
    )

    logger.info(f"Running aiperf with trace dataset: {trace_dataset}")
    logger.info(f"Command: {' '.join(aiperf_cmd)}")

    try:
        # Run aiperf and let it output directly to terminal
        subprocess.run(aiperf_cmd, check=True)

        logger.info("AIPerf profiling completed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"AIPerf failed with error code: {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        raise


def benchmark_on_dataset():
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
        default="http://localhost:8000",
        help="Server URL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="real_data_benchmark_results",
        help="Output directory for results",
    )

    # Trace dataset and synthesis configuration (similar to synthesizer.py)
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
        or args.min_isl is not None
        or args.min_osl is not None
        or args.max_osl is not None
    )

    if not needs_synthesis:
        # No synthesis needed, use original dataset
        trace_dataset_path = args.input_dataset
        logger.info(
            f"Using original trace dataset (no synthesis parameters modified): {trace_dataset_path}"
        )
    else:
        # Generate synthetic data based on input dataset
        logger.info("Generating synthetic trace data...")
        logger.info(f"  Base dataset: {args.input_dataset}")
        logger.info(
            f"  Num requests: {args.num_requests if args.num_requests else 'all'}"
        )
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

        # Set random seed for reproducibility
        np.random.seed(args.seed)

        # Create synthesizer
        synthesizer = Synthesizer(
            args.input_dataset,
            block_size=args.block_size,
            speedup_ratio=args.speedup_ratio,
            prefix_len_multiplier=args.prefix_len_multiplier,
            prefix_root_multiplier=args.prefix_root_multiplier,
            prompt_len_multiplier=args.prompt_len_multiplier,
        )

        # Determine number of requests
        if args.num_requests is None:
            # Count requests in original dataset
            with open(args.input_dataset, "r") as f:
                num_requests = sum(1 for _ in f)
            logger.info(f"Using all {num_requests} requests from input dataset")
        else:
            num_requests = args.num_requests

        # Generate synthetic requests
        requests = synthesizer.synthesize_requests(
            num_requests,
            max_isl=args.max_isl,
            min_isl=args.min_isl,
            min_osl=args.min_osl,
            max_osl=args.max_osl,
        )
        logger.info(f"Generated {len(requests)} synthetic requests")

        # Save synthetic data to a permanent file in output directory
        synthetic_trace_filename = "synthetic_trace.jsonl"
        trace_dataset_path = os.path.join(args.output_dir, synthetic_trace_filename)

        # Write synthetic data to file
        with open(trace_dataset_path, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        logger.info(f"Synthetic trace data saved to: {trace_dataset_path}")

    # Run benchmark with the trace dataset
    artifact_dir = os.path.join(args.output_dir, "aiperf_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    run_benchmark_with_trace(
        args.model,
        args.tokenizer,
        trace_dataset_path,
        artifact_dir,
        args.url,
        args.seed,
    )

    logger.info(f"Results saved to: {artifact_dir}")

    # FIXME: hardcoded for quick setup.
    with open(os.path.join("/app/repo/real_data_benchmark_results/aiperf_artifacts", "profile_export_aiperf.json"), "r") as f:
        result_summary = json.load(f)

        if "time_to_first_token" not in result_summary:
            logger.error("Required metrics not found in aiperf results.")
            return None


        time_to_first_token = result_summary["time_to_first_token"]["avg"]
        time_to_second_token = result_summary["time_to_second_token"]["avg"]
        inter_token_latency = result_summary["inter_token_latency"]["avg"]
        request_throughput = result_summary["request_throughput"]["avg"]
        total_token_throughput = result_summary["total_token_throughput"]["avg"]

        global ttft_base, throughput_base

        if ttft_base == 0.0:
            ttft_base = time_to_first_token
        if throughput_base == 0.0:
            throughput_base = total_token_throughput

        eps = 1e-9
        throughput_ratio = (request_throughput + eps) / (throughput_base + eps)
        ttft_ratio = (time_to_first_token + eps) / (ttft_base + eps)
        fitness = math.log(throughput_ratio) - math.log(ttft_ratio)

        return {
            "fitness": fitness,
            "signature": [float(time_to_first_token), float(inter_token_latency), float(total_token_throughput)],
            "time_to_first_token": float(time_to_first_token),
            "time_to_second_token": float(time_to_second_token),
            "inter_token_latency": float(inter_token_latency),
            "request_throughput": float(request_throughput),
            "total_token_throughput": float(total_token_throughput),
        }

    return None

def wait_for_service(host="localhost", port=8000, path="/health", timeout=30, interval=0.5):
    start_time = time.time()
    while True:
        try:
            conn = http.client.HTTPConnection(host, port)
            conn.request("GET", path)
            resp = conn.getresponse()
            if resp.status == 200:
                print("Service is ready.")
                return True
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except:
                pass
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Service not ready after {timeout} seconds")
        time.sleep(interval)


if __name__ == "__main__":
    logger.info("Compiling Dynamo...")

    COMPILE_CMD = (
        "cd /app/repo && "
        ". dynamo/bin/activate && "
        "cd /app/repo/lib/bindings/python && "
        "maturin develop --release --uv --strip && "
        "cd /app/repo && "
        "uv pip install -e ."
    )
    subprocess.run(COMPILE_CMD, shell=True, check=True)

    logger.info("Starting Dynamo frontend...")

    FRONTEND_CMD = [
        "cd /app/repo && "
        "dynamo/bin/python -m dynamo.frontend "
        "--router-mode kv ",
        "--router-reset-states ",
        "--http-port 8000",
    ]

    # Make sure always run the newest frontend. Here we're evolving the
    # select worker function, which is part of the frontend.
    frontend_process = subprocess.Popen(FRONTEND_CMD, shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wait_for_service()

    sys.argv = [
        "evaluator_real.py",
        # "--num-requests", "10",
        "--input-dataset", "/app/repo/mooncake_trace.jsonl",
    ]

    logger.info("Running benchmark...")
    start_time = time.time()
    output = benchmark_on_dataset()

    logger.info("Shutting down Dynamo frontend...")
    frontend_process.terminate()
    frontend_process.wait(timeout=10)

    logger.info("Benchmark completed.")
    elapsed_time = time.time() - start_time
    if output is not None:
        output["elapsed_time"] = elapsed_time
        print(json.dumps({"output": output, "metainfo": "Success"}))
    else:
        print(json.dumps({"output": {"elapsed_time": elapsed_time}, "metainfo": "Error"}))