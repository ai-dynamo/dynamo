#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Priority queue benchmark: splits a trace into priority tiers, runs a baseline
(no priority tagging) and a priority-tagged run with the same split, then
produces a bar chart comparing TTFT across tiers."""

import argparse
import copy
import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from common import (
    add_common_args,
    add_synthesis_args,
    get_aiperf_cmd_for_trace,
    prepare_trace_dataset,
    resolve_tokenizer,
    setup_logger,
)

logger = setup_logger(__name__)

TIERS = ["low", "medium", "high"]


def parse_float_list(s):
    """Parse a comma-separated string into a list of floats."""
    return [float(x.strip()) for x in s.split(",")]


def split_trace(requests, distribution, seed):
    """Split requests into priority tiers by distribution. Deterministic given seed."""
    rng = np.random.RandomState(seed)
    labels = rng.choice(len(distribution), size=len(requests), p=distribution)
    return {
        tier: [r for r, label in zip(requests, labels) if label == i]
        for i, tier in enumerate(TIERS)
    }


def offset_hash_ids(tier_requests):
    """Return a deep copy of tier_requests with all hash_ids shifted by max_hash_id + 1.

    Preserves the prefix tree structure (relative ordering and sharing)
    while ensuring no KV cache hits from a previous run.
    """
    max_hash_id = max(
        h
        for requests in tier_requests.values()
        for req in requests
        for h in req["hash_ids"]
    )
    offset = max_hash_id + 1
    shifted = {}
    for tier, requests in tier_requests.items():
        shifted[tier] = []
        for req in requests:
            r = copy.copy(req)
            r["hash_ids"] = [h + offset for h in r["hash_ids"]]
            shifted[tier].append(r)
    return shifted


def write_trace_file(requests, path):
    """Write a list of request dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")


def run_concurrent_streams(
    args, tier_requests, priority_values, run_dir, tag_priority, logger, seed=None
):
    """Launch concurrent aiperf subprocesses for each tier.

    Args:
        tag_priority: If True, inject nvext.agent_hints.latency_sensitivity per tier.
    """
    processes = []
    log_files = []
    for tier, pj in zip(TIERS, priority_values):
        tier_dir = os.path.join(run_dir, f"{tier}_priority")
        os.makedirs(tier_dir, exist_ok=True)

        trace_path = os.path.join(tier_dir, "trace.jsonl")
        write_trace_file(tier_requests[tier], trace_path)

        artifact_dir = os.path.join(tier_dir, "aiperf_artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        cmd = get_aiperf_cmd_for_trace(
            args.model,
            args.tokenizer,
            trace_path,
            artifact_dir,
            seed if seed is not None else args.seed,
            args.block_size,
            args.url,
        )
        cmd.extend(["--log-level", "WARNING", "--ui-type", "none"])
        if tag_priority:
            cmd.extend(
                [
                    "--extra-inputs",
                    json.dumps({"nvext": {"agent_hints": {"latency_sensitivity": pj}}}),
                ]
            )

        log_path = os.path.join(tier_dir, "aiperf.log")
        log_file = open(log_path, "w")
        log_files.append(log_file)

        label = "priority" if tag_priority else "baseline"
        logger.info(f"Launching {tier} tier ({label}, latency_sensitivity={pj})")
        logger.info(f"  Command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((tier, proc))

    failed = []
    for tier, proc in processes:
        proc.wait()
        if proc.returncode == 0:
            logger.info(f"  {tier} tier completed successfully")
        else:
            logger.error(f"  {tier} tier failed with exit code {proc.returncode}")
            failed.append(tier)

    for log_file in log_files:
        log_file.close()

    if failed:
        label = "priority" if tag_priority else "baseline"
        logger.error(f"Failed tiers in {label} run: {', '.join(failed)}")
        logger.error("Check the aiperf.log files in each tier directory for details")
        raise SystemExit(1)


def load_ttft(run_dir, tier):
    """Load TTFT stats from an aiperf result JSON."""
    result_path = os.path.join(
        run_dir, f"{tier}_priority", "aiperf_artifacts", "profile_export_aiperf.json"
    )
    with open(result_path, "r") as f:
        data = json.load(f)
    ttft = data["time_to_first_token"]
    return ttft["p50"], ttft["p25"], ttft["p75"]


def plot_ttft_comparison(baseline_dir, priority_dir, output_path, priority_values):
    """Create a grouped bar chart comparing TTFT between baseline and priority runs."""
    x = np.arange(len(TIERS))
    width = 0.35

    baseline_medians = []
    baseline_lo = []
    baseline_hi = []
    priority_medians = []
    priority_lo = []
    priority_hi = []

    for tier in TIERS:
        p50, p25, p75 = load_ttft(baseline_dir, tier)
        baseline_medians.append(p50)
        baseline_lo.append(p50 - p25)
        baseline_hi.append(p75 - p50)

        p50, p25, p75 = load_ttft(priority_dir, tier)
        priority_medians.append(p50)
        priority_lo.append(p50 - p25)
        priority_hi.append(p75 - p50)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        baseline_medians,
        width,
        yerr=[baseline_lo, baseline_hi],
        label="Baseline (no priority)",
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        priority_medians,
        width,
        yerr=[priority_lo, priority_hi],
        label="With latency_sensitivity",
        capsize=4,
    )

    tier_labels = [
        f"{tier.capitalize()}\n(ls={pj})" for tier, pj in zip(TIERS, priority_values)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token by Priority Tier")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info(f"Plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Priority benchmark: compare TTFT with and without priority tagging"
    )

    add_common_args(parser)
    add_synthesis_args(parser)

    parser.add_argument(
        "--priority-distribution",
        type=str,
        default="0.5,0.3,0.2",
        help="Comma-separated fractions for low/medium/high tiers (default: 0.5,0.3,0.2)",
    )
    parser.add_argument(
        "--priority-values",
        type=str,
        default="0,1,2",
        help="Comma-separated latency_sensitivity values for low/medium/high tiers (default: 0,1,2)",
    )

    args = parser.parse_args()
    resolve_tokenizer(args)

    distribution = parse_float_list(args.priority_distribution)
    priority_values = parse_float_list(args.priority_values)

    if len(distribution) != len(TIERS):
        parser.error(
            f"--priority-distribution must have {len(TIERS)} values, got {len(distribution)}"
        )
    if len(priority_values) != len(TIERS):
        parser.error(
            f"--priority-values must have {len(TIERS)} values, got {len(priority_values)}"
        )
    if abs(sum(distribution) - 1.0) > 1e-6:
        parser.error(
            f"--priority-distribution must sum to 1.0, got {sum(distribution)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare the trace dataset (synthesis if needed)
    requests, _ = prepare_trace_dataset(args, args.output_dir, logger)

    # Split into priority tiers (deterministic via seed)
    tier_requests = split_trace(requests, distribution, args.seed)
    for tier in TIERS:
        logger.info(f"  {tier} priority: {len(tier_requests[tier])} requests")

    # Offset hash_ids for the priority run so it starts with a cold KV cache,
    # keeping the comparison fair. Same seed for both runs so prompts match.
    priority_tier_requests = offset_hash_ids(tier_requests)

    # Run 1: Baseline (same split, no priority tagging)
    baseline_dir = os.path.join(args.output_dir, "baseline")
    logger.info("=== Running baseline (no priority tagging) ===")
    run_concurrent_streams(
        args,
        tier_requests,
        priority_values,
        baseline_dir,
        tag_priority=False,
        logger=logger,
        seed=args.seed,
    )

    # Run 2: With priority tagging (offset hash_ids for cold cache)
    priority_dir = os.path.join(args.output_dir, "priority")
    logger.info("=== Running with priority tagging ===")
    run_concurrent_streams(
        args,
        priority_tier_requests,
        priority_values,
        priority_dir,
        tag_priority=True,
        logger=logger,
        seed=args.seed,
    )

    # Plot comparison
    plot_path = os.path.join(args.output_dir, "ttft_comparison.png")
    plot_ttft_comparison(baseline_dir, priority_dir, plot_path, priority_values)

    logger.info(f"All runs completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
