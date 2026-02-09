#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Priority queue benchmark: splits a trace into priority tiers and runs
concurrent aiperf streams with different nvext.priority_jump values."""

import argparse
import json
import os
import subprocess

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


def write_trace_file(requests, path):
    """Write a list of request dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Priority benchmark: split trace into tiers and run concurrent aiperf streams"
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
        default="0.0,0.4,0.8",
        help="Comma-separated priority_jump values for low/medium/high tiers (default: 0.0,0.4,0.8)",
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

    # Launch concurrent aiperf subprocesses
    processes = []
    log_files = []
    for tier, pj in zip(TIERS, priority_values):
        tier_dir = os.path.join(args.output_dir, f"{tier}_priority")
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
            args.seed,
            args.block_size,
            args.url,
        )
        cmd.extend(["--extra-inputs", json.dumps({"nvext": {"priority_jump": pj}})])

        log_path = os.path.join(tier_dir, "aiperf.log")
        log_file = open(log_path, "w")
        log_files.append(log_file)

        logger.info(f"Launching {tier} priority stream (priority_jump={pj})")
        logger.info(f"  Trace: {trace_path}")
        logger.info(f"  Log: {log_path}")
        logger.info(f"  Command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((tier, proc))

    # Wait for all subprocesses to complete
    failed = []
    for tier, proc in processes:
        proc.wait()
        if proc.returncode == 0:
            logger.info(f"{tier} priority stream completed successfully")
        else:
            logger.error(
                f"{tier} priority stream failed with exit code {proc.returncode}"
            )
            failed.append(tier)

    for log_file in log_files:
        log_file.close()

    if failed:
        logger.error(f"Failed tiers: {', '.join(failed)}")
        logger.error("Check the aiperf.log files in each tier directory for details")
        raise SystemExit(1)

    logger.info(f"All priority streams completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
