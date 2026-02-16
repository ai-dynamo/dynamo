#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from common import (
    add_common_args,
    get_common_aiperf_flags,
    resolve_tokenizer,
    setup_logger,
)

logger = setup_logger(__name__)


def get_aiperf_cmd(
    model,
    tokenizer,
    prefix_ratio,
    isl,
    osl,
    requests,
    concurrency,
    seed,
    num_prefix_prompts,
    artifact_dir,
    url="http://localhost:8888",
    use_expected_osl=False,
    collect_gpu_sku=False,
):
    """Build aiperf command based on prefix ratio"""
    prefix_length = int(isl * prefix_ratio)
    synthetic_input_length = int(isl * (1 - prefix_ratio))

    # Build nvext JSON with optional agent_hints.osl
    nvext_dict = {"ignore_eos": True}
    if use_expected_osl:
        nvext_dict["agent_hints"] = {"osl": osl}
    nvext_json = json.dumps({"nvext": nvext_dict})

    cmd = [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
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
        nvext_json,
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
        "--dataset-sampling-strategy",
        "shuffle",
    ]
    cmd.extend(get_common_aiperf_flags(collect_gpu_sku=collect_gpu_sku))
    return cmd


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


def extract_gpu_sku_from_aiperf_artifact(artifact_dir: str) -> Optional[str]:
    """Extract GPU model name (SKU) from aiperf export when GPU telemetry was enabled.

    Reads profile_export_aiperf.json (telemetry_data.endpoints[*].gpus[*].gpu_name)
    or the first line of gpu_telemetry_export.jsonl (gpu_model_name). Returns the
    first GPU name found, or None if not present or on error.
    """
    try:
        for root, _, files in os.walk(artifact_dir):
            if "profile_export_aiperf.json" in files:
                json_path = os.path.join(root, "profile_export_aiperf.json")
                with open(json_path, "r") as f:
                    data = json.load(f)
                telemetry = data.get("telemetry_data")
                if isinstance(telemetry, dict):
                    endpoints = telemetry.get("endpoints") or {}
                    for ep_data in endpoints.values():
                        if not isinstance(ep_data, dict):
                            continue
                        gpus = ep_data.get("gpus") or {}
                        for gpu_summary in gpus.values():
                            if isinstance(gpu_summary, dict):
                                name = gpu_summary.get("gpu_name")
                                if name:
                                    return name.strip()
            if "gpu_telemetry_export.jsonl" in files:
                jsonl_path = os.path.join(root, "gpu_telemetry_export.jsonl")
                with open(jsonl_path, "r") as f:
                    first_line = f.readline()
                if first_line.strip():
                    record = json.loads(first_line)
                    name = record.get("gpu_model_name")
                    if name:
                        return name.strip()
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.debug("Could not extract GPU SKU from aiperf artifact: %s", e)
    return None


def run_benchmark(
    model,
    tokenizer,
    prefix_ratio,
    isl,
    osl,
    requests,
    concurrency,
    seed,
    num_prefix_prompts,
    output_dir,
    url,
    use_expected_osl=False,
    collect_gpu_sku=False,
) -> Optional[Dict]:
    """Run aiperf benchmark for a specific prefix ratio"""
    logger.info(
        f"Running benchmark with prefix_ratio={prefix_ratio}, seed={seed}, url={url}"
    )

    artifact_dir = f"{output_dir}/prefix_ratio_{prefix_ratio}_seed_{seed}"
    os.makedirs(artifact_dir, exist_ok=True)

    aiperf_cmd = get_aiperf_cmd(
        model,
        tokenizer,
        prefix_ratio,
        isl,
        osl,
        requests,
        concurrency,
        seed,
        num_prefix_prompts,
        artifact_dir,
        url,
        use_expected_osl,
        collect_gpu_sku=collect_gpu_sku,
    )

    logger.info(f"Command: {' '.join(aiperf_cmd)}")

    try:
        subprocess.run(aiperf_cmd, check=True)
        logger.info("AIPerf profiling completed successfully")
        return get_aiperf_result(artifact_dir)
    except subprocess.CalledProcessError as e:
        logger.error(f"AIPerf failed with error code: {e.returncode}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prefix ratios and plot results"
    )

    add_common_args(parser)

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
    parser.add_argument(
        "--prefix-ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="List of prefix ratios to test",
    )
    parser.add_argument(
        "--collect-gpu-sku",
        action="store_true",
        help="Enable aiperf GPU telemetry (pynvml) and record GPU model name in results config",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        metavar="N",
        help="Tensor parallel size used for the run (recorded in results config)",
    )

    args = parser.parse_args()
    resolve_tokenizer(args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store results
    prefix_ratios = []
    ttft_p25_values = []
    ttft_p50_values = []
    ttft_p75_values = []
    itl_p25_values = []
    itl_p50_values = []
    itl_p75_values = []

    current_seed = args.seed
    gpu_sku = None

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
            args.url,
            args.use_expected_osl,
            collect_gpu_sku=args.collect_gpu_sku,
        )

        if result is not None:
            ttft = result["time_to_first_token"]
            itl = result["inter_token_latency"]

            prefix_ratios.append(prefix_ratio)
            ttft_p25_values.append(ttft["p25"])
            ttft_p50_values.append(ttft["p50"])
            ttft_p75_values.append(ttft["p75"])
            itl_p25_values.append(itl["p25"])
            itl_p50_values.append(itl["p50"])
            itl_p75_values.append(itl["p75"])

            if args.collect_gpu_sku and gpu_sku is None:
                artifact_dir = (
                    f"{args.output_dir}/prefix_ratio_{prefix_ratio}_seed_{current_seed}"
                )
                gpu_sku = extract_gpu_sku_from_aiperf_artifact(artifact_dir)
                if gpu_sku:
                    logger.info("Collected GPU SKU from aiperf: %s", gpu_sku)

            logger.info(
                f"Prefix ratio {prefix_ratio}: TTFT p50={ttft['p50']:.2f}ms (p25={ttft['p25']:.2f}, p75={ttft['p75']:.2f}), "
                f"ITL p50={itl['p50']:.2f}ms (p25={itl['p25']:.2f}, p75={itl['p75']:.2f})"
            )

        current_seed += 1

    # Create plots
    if prefix_ratios and ttft_p50_values and itl_p50_values:
        plt.figure(figsize=(12, 5))

        # Plot TTFT vs Prefix Ratio with shaded p25-p75 region
        plt.subplot(1, 2, 1)
        plt.fill_between(
            prefix_ratios,
            ttft_p25_values,
            ttft_p75_values,
            alpha=0.3,
            color="blue",
            label="p25-p75",
        )
        plt.plot(
            prefix_ratios,
            ttft_p50_values,
            "bo-",
            linewidth=2,
            markersize=8,
            label="p50",
        )
        plt.xlabel("Prefix Ratio")
        plt.ylabel("Time to First Token (ms)")
        plt.title("TTFT vs Prefix Ratio")
        plt.grid(True, alpha=0.3)
        plt.legend()
        for i, (pr, p50) in enumerate(zip(prefix_ratios, ttft_p50_values)):
            plt.annotate(
                f"{p50:.1f}ms",
                (pr, p50),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # Plot ITL vs Prefix Ratio with shaded p25-p75 region
        plt.subplot(1, 2, 2)
        plt.fill_between(
            prefix_ratios,
            itl_p25_values,
            itl_p75_values,
            alpha=0.3,
            color="red",
            label="p25-p75",
        )
        plt.plot(
            prefix_ratios, itl_p50_values, "ro-", linewidth=2, markersize=8, label="p50"
        )
        plt.xlabel("Prefix Ratio")
        plt.ylabel("Inter-Token Latency (ms)")
        plt.title("ITL vs Prefix Ratio")
        plt.grid(True, alpha=0.3)
        plt.legend()
        for i, (pr, p50) in enumerate(zip(prefix_ratios, itl_p50_values)):
            plt.annotate(
                f"{p50:.1f}ms",
                (pr, p50),
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
        config = {
            "model": args.model,
            "tokenizer": args.tokenizer,
            "isl": args.isl,
            "osl": args.osl,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "initial_seed": args.seed,
        }
        if gpu_sku is not None:
            config["gpu_sku"] = gpu_sku
        if args.tensor_parallel_size is not None:
            config["tp"] = args.tensor_parallel_size
        results_data = {
            "prefix_ratios": prefix_ratios,
            "ttft_p25_values": ttft_p25_values,
            "ttft_p50_values": ttft_p50_values,
            "ttft_p75_values": ttft_p75_values,
            "itl_p25_values": itl_p25_values,
            "itl_p50_values": itl_p50_values,
            "itl_p75_values": itl_p75_values,
            "config": config,
        }

        results_path = f"{args.output_dir}/results_summary.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Results summary saved to {results_path}")

    else:
        logger.error("No successful benchmark results to plot")


if __name__ == "__main__":
    main()
