#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess

from common import (
    add_common_args,
    add_synthesis_args,
    get_aiperf_cmd_for_trace,
    prepare_trace_dataset,
    resolve_tokenizer,
    setup_logger,
    validate_worker_participation,
    worker_participation_requested,
)

logger = setup_logger(__name__)


def run_benchmark_with_trace(
    model,
    tokenizer,
    trace_dataset,
    artifact_dir,
    url,
    seed,
    block_size,
    verify_worker_participation=False,
    minimum_prefill_workers=0,
    minimum_decode_workers=0,
):
    """Run aiperf benchmark with a trace dataset"""
    aiperf_cmd = get_aiperf_cmd_for_trace(
        model,
        tokenizer,
        trace_dataset,
        artifact_dir,
        seed,
        block_size,
        url,
        verify_worker_participation,
    )

    logger.info(f"Running aiperf with trace dataset: {trace_dataset}")
    logger.info(f"Command: {' '.join(aiperf_cmd)}")

    try:
        # Run aiperf and let it output directly to terminal
        subprocess.run(aiperf_cmd, check=True)

        logger.info("AIPerf profiling completed successfully")
        if verify_worker_participation:
            validate_worker_participation(
                artifact_dir,
                minimum_prefill_workers,
                minimum_decode_workers,
                logger,
            )

    except subprocess.CalledProcessError as e:
        logger.error(f"AIPerf failed with error code: {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark with real or synthesized mooncake-style trace data"
    )

    add_common_args(parser)
    add_synthesis_args(parser)

    args = parser.parse_args()
    resolve_tokenizer(args)
    verify_worker_participation = worker_participation_requested(args)

    os.makedirs(args.output_dir, exist_ok=True)

    _, trace_dataset_path = prepare_trace_dataset(args, args.output_dir, logger)

    artifact_dir = os.path.join(args.output_dir, "aiperf_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    run_benchmark_with_trace(
        args.model,
        args.tokenizer,
        trace_dataset_path,
        artifact_dir,
        args.url,
        args.seed,
        args.block_size,
        verify_worker_participation,
        args.minimum_prefill_workers,
        args.minimum_decode_workers,
    )

    logger.info(f"Results saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
