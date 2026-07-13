# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-benchmark result collection for vLLM workers."""

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from vllm.config import VllmConfig

from .instrumented_scheduler import ENV_FPM_BENCHMARK_OUTPUT_PATH

logger = logging.getLogger(__name__)


async def wait_and_load_benchmark(
    bench_cfg: dict,
    vllm_config: VllmConfig,
    *,
    dp_range_resolver: Callable[[VllmConfig], tuple[int, int]],
) -> dict:
    """Wait for benchmark result files and aggregate across DP ranks."""
    base_path = Path(
        os.environ.get(ENV_FPM_BENCHMARK_OUTPUT_PATH, bench_cfg["output_path"])
    )
    timeout = int(bench_cfg.get("timeout", 300))

    try:
        dp_start, dp_size = dp_range_resolver(vllm_config)
    except Exception:
        logger.warning(
            "Could not determine DP range, assuming single rank",
            exc_info=True,
        )
        dp_start, dp_size = 0, 1

    rank_paths = []
    for dp_rank in range(dp_start, dp_start + dp_size):
        if dp_rank == 0:
            rank_paths.append(base_path)
        else:
            stem, ext = os.path.splitext(str(base_path))
            rank_paths.append(Path(f"{stem}_dp{dp_rank}{ext}"))

    logger.info(
        "Waiting for benchmark to complete (files: %s, timeout: %ds)...",
        rank_paths,
        timeout,
    )

    deadline = time.monotonic() + timeout
    for path in rank_paths:
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Benchmark did not complete within {timeout}s. Missing: {path}"
                )
            await asyncio.sleep(0.1)

    merged: dict = {}
    for index, path in enumerate(rank_paths):
        with open(path) as benchmark_file:
            data = json.load(benchmark_file)
        if data.get("valid") is False:
            raise RuntimeError(
                f"Self-benchmark produced incomplete results at {path}: "
                f"coverage={data.get('coverage')} "
                f"skipped_points={data.get('skipped_points')} "
                f"missing_phases={data.get('missing_phases')}"
            )
        if index == 0:
            merged = data
            for result in merged.get("results", []):
                result["point"]["dp_rank"] = dp_start
            continue

        dp_rank = dp_start + index
        for result in data.get("results", []):
            result["point"]["dp_rank"] = dp_rank
        merged.setdefault("results", []).extend(data.get("results", []))
        merged_coverage = merged.get("coverage")
        rank_coverage = data.get("coverage")
        if isinstance(merged_coverage, dict) and isinstance(rank_coverage, dict):
            for key in ("expected_points", "completed_points", "skipped_points"):
                merged_coverage[key] = merged_coverage.get(key, 0) + rank_coverage.get(
                    key, 0
                )
        merged.setdefault("skipped_points", []).extend(data.get("skipped_points", []))

    logger.info(
        "Benchmark complete, %d points across %d rank(s)",
        len(merged.get("results", [])),
        len(rank_paths),
    )
    return merged
