#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
import subprocess
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

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


def get_aiperf_cmd(
    model,
    tokenizer,  # Add tokenizer parameter
    prefix_ratio,
    isl,
    osl,
    requests,
    concurrency,
    seed,
    num_prefix_prompts,
    artifact_dir,
    url="http://localhost:8888",
):
    """Build aiperf command based on prefix ratio"""
    prefix_length = int(isl * prefix_ratio)
    synthetic_input_length = int(isl * (1 - prefix_ratio))

    return [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        tokenizer,  # Use the tokenizer parameter instead of model
        "--endpoint-type",
        "chat",
        "--endpoint",
        "v1/chat/completions",
        "--streaming",
        "--url",
        url,
        "--synthetic-input-tokens-mean",
        str(synthetic_input_length),
        "--synthetic-input-tokens-stddev",
        str(round(synthetic_input_length / 4)),
        "--output-tokens-mean",
        str(osl),
        "--output-tokens-stddev",
        str(round(osl / 4)),
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        '{"nvext":{"ignore_eos":true}}',
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(requests),
        "--num-dataset-entries",
        str(requests),
        "--random-seed",
        str(seed),
        "--prefix-prompt-length",
        str(prefix_length),
        "--num-prefix-prompts",
        str(num_prefix_prompts),
        "--artifact-dir",
        artifact_dir,
        "-v",
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]


def get_aiperf_result(artifact_dir: str) -> dict:
    """Parse aiperf results from JSON file"""
    json_file_path = None
    for root, _, files in os.walk(artifact_dir):
        if "profile_export_aiperf.json" in files:
            json_file_path = os.path.join(root, "profile_export_aiperf.json")
            break

    if json_file_path is None:
        raise FileNotFoundError(
            f"profile_export_aiperf.json not found in {artifact_dir}"
        )

    with open(json_file_path, "r") as f:
        return json.load(f)


def run_benchmark_single_url(
    model,
    tokenizer,  # Add tokenizer parameter
    prefix_ratio,
    isl,
    osl,
    requests,
    concurrency,
    seed,
    num_prefix_prompts,
    artifact_dir,
    url,
) -> Optional[Dict]:
    """Run aiperf benchmark for a single URL"""
    aiperf_cmd = get_aiperf_cmd(
        model,
        tokenizer,  # Pass tokenizer parameter
        prefix_ratio,
        isl,
        osl,
        requests,
        concurrency,
        seed,
        num_prefix_prompts,
        artifact_dir,
        url,
    )

    logger.info(f"Running command for URL {url}: {' '.join(aiperf_cmd)}")

    try:
        # Run aiperf and let it output directly to terminal
        subprocess.run(aiperf_cmd, check=True)

        logger.info(f"AIPerf profiling completed successfully for URL {url}")

        aiperf_result = get_aiperf_result(artifact_dir)
        return aiperf_result

    except subprocess.CalledProcessError as e:
        logger.error(f"AIPerf failed for URL {url} with error code: {e.returncode}")
        return None


def aggregate_results(results: List[Optional[Dict]]) -> Optional[Dict]:
    """Aggregate results from multiple URLs"""
    if not results:
        return None

    # For TTFT, we take the average across all URLs
    # For throughput, we sum across all URLs (total system throughput)
    ttft_values = [r["time_to_first_token"]["avg"] for r in results if r is not None]
    throughput_values = [
        r["output_token_throughput"]["avg"] for r in results if r is not None
    ]

    if not ttft_values or not throughput_values:
        return None

    aggregated = {
        "time_to_first_token": {"avg": sum(ttft_values) / len(ttft_values)},
        "output_token_throughput": {
            "avg": sum(throughput_values)  # Total throughput across all URLs
        },
    }

    return aggregated


def run_benchmark(
    model,
    tokenizer,  # Add tokenizer parameter
    prefix_ratio,
    isl,
    osl,
    requests,
    concurrency,
    seed,
    num_prefix_prompts,
    output_dir,
    urls,
) -> Optional[Dict]:
    """Run aiperf benchmark for a specific prefix ratio"""
    logger.info(
        f"Running benchmark with prefix_ratio={prefix_ratio}, seed={seed}, URLs={urls}"
    )

    # If single URL, maintain existing behavior
    if isinstance(urls, str):
        urls = [urls]

    if len(urls) == 1:
        artifact_dir = f"{output_dir}/prefix_ratio_{prefix_ratio}_seed_{seed}"
        os.makedirs(artifact_dir, exist_ok=True)

        return run_benchmark_single_url(
            model,
            tokenizer,  # Pass tokenizer parameter
            prefix_ratio,
            isl,
            osl,
            requests,
            concurrency,
            seed,
            num_prefix_prompts,
            artifact_dir,
            urls[0],
        )

    # Multiple URLs: split requests and concurrency
    num_urls = len(urls)
    base_requests_per_url = requests // num_urls
    remainder_requests = requests % num_urls
    base_concurrency_per_url = max(1, concurrency // num_urls)

    # Launch parallel processes
    processes = []
    artifact_dirs = []

    for i, url in enumerate(urls):
        # Distribute remainder requests to first few URLs
        url_requests = base_requests_per_url + (1 if i < remainder_requests else 0)

        artifact_dir = f"{output_dir}/prefix_ratio_{prefix_ratio}_seed_{seed}_url_{i}"
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_dirs.append(artifact_dir)

        aiperf_cmd = get_aiperf_cmd(
            model,
            tokenizer,  # Pass tokenizer parameter
            prefix_ratio,
            isl,
            osl,
            url_requests,
            base_concurrency_per_url,
            seed,
            num_prefix_prompts,
            artifact_dir,
            url,
        )

        logger.info(f"Launching process for URL {url}: {' '.join(aiperf_cmd)}")

        # Run process without capturing output - let it stream to terminal
        process = subprocess.Popen(aiperf_cmd)
        processes.append((process, url, artifact_dir))

    # Wait for all processes to complete and collect results
    results: List[Optional[Dict]] = []
    for process, url, artifact_dir in processes:
        return_code = process.wait()

        if return_code == 0:
            logger.info(f"AIPerf completed successfully for URL {url}")

            try:
                aiperf_result = get_aiperf_result(artifact_dir)
                results.append(aiperf_result)
            except Exception as e:
                logger.error(f"Failed to get results for URL {url}: {e}")
                results.append(None)
        else:
            logger.error(f"AIPerf failed for URL {url} with error code: {return_code}")
            results.append(None)

    # Aggregate results
    return aggregate_results(results)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prefix ratios and plot results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Tokenizer name (defaults to model)",
    )
    parser.add_argument(
        "--url",
        type=str,
        nargs="+",  # Accept multiple URLs
        default=["http://localhost:8000"],
        # default=["http://localhost:8090", "http://localhost:8090"],
        help="Server URL(s). Can specify multiple URLs for parallel benchmarking",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kv_router",
        help="Output directory for results",
    )
    parser.add_argument("--num-prefix-prompts", type=int, default=20)
    parser.add_argument("--isl", type=int, default=14000, help="Input sequence length")
    parser.add_argument("--osl", type=int, default=200, help="Output sequence length")
    parser.add_argument("--requests", type=int, default=200, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrency level")
    parser.add_argument("--seed", type=int, default=0, help="Initial random seed")
    parser.add_argument(
        "--prefix-ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="List of prefix ratios to test",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store results
    prefix_ratios = []
    ttft_values = []
    throughput_values = []

    current_seed = args.seed

    # Run benchmarks for each prefix ratio
    for prefix_ratio in args.prefix_ratios:
        result = run_benchmark(
            args.model,
            args.tokenizer,
            prefix_ratio,
            args.isl,
            args.osl,
            args.requests,
            args.concurrency,
            current_seed,
            args.num_prefix_prompts,
            args.output_dir,
            args.url,  # Now passing list of URLs
        )

        if result is not None:
            ttft = result["time_to_first_token"]["avg"]
            throughput = result["output_token_throughput"]["avg"]

            prefix_ratios.append(prefix_ratio)
            ttft_values.append(ttft)
            throughput_values.append(throughput)

            logger.info(
                f"Prefix ratio {prefix_ratio}: TTFT={ttft:.2f}ms, Throughput={throughput:.2f} tokens/s"
            )

        current_seed += 1

    # Create plots
    if prefix_ratios and ttft_values and throughput_values:
        # Plot TTFT vs Prefix Ratio
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(prefix_ratios, ttft_values, "bo-", linewidth=2, markersize=8)
        plt.xlabel("Prefix Ratio")
        plt.ylabel("Time to First Token (ms)")
        plt.title("TTFT vs Prefix Ratio")
        plt.grid(True, alpha=0.3)
        for i, (pr, ttft) in enumerate(zip(prefix_ratios, ttft_values)):
            plt.annotate(
                f"{ttft:.1f}ms",
                (pr, ttft),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # Plot Throughput vs Prefix Ratio
        plt.subplot(1, 2, 2)
        plt.plot(prefix_ratios, throughput_values, "ro-", linewidth=2, markersize=8)
        plt.xlabel("Prefix Ratio")
        plt.ylabel("Output Token Throughput (tokens/s)")
        plt.title("Throughput vs Prefix Ratio")
        plt.grid(True, alpha=0.3)
        for i, (pr, thpt) in enumerate(zip(prefix_ratios, throughput_values)):
            plt.annotate(
                f"{thpt:.1f}",
                (pr, thpt),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.tight_layout()

        # Save plot
        plot_path = f"{args.output_dir}/prefix_ratio_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Performance plot saved to {plot_path}")

        # Save results to JSON
        results_data = {
            "prefix_ratios": prefix_ratios,
            "ttft_values": ttft_values,
            "throughput_values": throughput_values,
            "config": {
                "model": args.model,
                "tokenizer": args.tokenizer,
                "isl": args.isl,
                "osl": args.osl,
                "requests": args.requests,
                "concurrency": args.concurrency,
                "initial_seed": args.seed,
            },
        }

        results_path = f"{args.output_dir}/results_summary.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Results summary saved to {results_path}")

    else:
        logger.error("No successful benchmark results to plot")


if __name__ == "__main__":
    main()
