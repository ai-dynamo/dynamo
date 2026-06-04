# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from dynamo.sglang.capacity import local_dp_rank_bounds

logger = logging.getLogger(__name__)

_DEFAULT_BENCHMARK_OUTPUT_PATH = "/tmp/benchmark_results.json"


def apply_benchmark_env(parsed_args: Any, cli_args: list[str]) -> None:
    """Map Dynamo benchmark env vars onto native SGLang benchmark args."""

    mode = os.environ.get("DYN_BENCHMARK_MODE")
    if mode and getattr(parsed_args, "benchmark_mode", None) is None:
        if mode not in {"prefill", "decode", "agg"}:
            raise ValueError(
                "DYN_BENCHMARK_MODE must be one of: prefill, decode, agg; "
                f"got {mode!r}"
            )
        parsed_args.benchmark_mode = mode

    _apply_int_env(
        parsed_args,
        cli_args,
        "benchmark_prefill_granularity",
        "--benchmark-prefill-granularity",
        "DYN_BENCHMARK_PREFILL_GRANULARITY",
    )
    _apply_int_env(
        parsed_args,
        cli_args,
        "benchmark_decode_length_granularity",
        "--benchmark-decode-length-granularity",
        "DYN_BENCHMARK_DECODE_LENGTH_GRANULARITY",
    )
    _apply_int_env(
        parsed_args,
        cli_args,
        "benchmark_decode_batch_granularity",
        "--benchmark-decode-batch-granularity",
        "DYN_BENCHMARK_DECODE_BATCH_GRANULARITY",
    )
    _apply_int_env(
        parsed_args,
        cli_args,
        "benchmark_warmup_iterations",
        "--benchmark-warmup-iterations",
        "DYN_BENCHMARK_WARMUP_ITERATIONS",
    )
    _apply_int_env(
        parsed_args,
        cli_args,
        "benchmark_timeout",
        "--benchmark-timeout",
        "DYN_BENCHMARK_TIMEOUT",
    )

    output_path = os.environ.get("DYN_BENCHMARK_OUTPUT_PATH")
    if output_path and not _has_cli_flag(cli_args, "--benchmark-output-path"):
        parsed_args.benchmark_output_path = output_path


def prepare_benchmark_output_path(server_args: Any, worker_id: object) -> None:
    """Make the default benchmark output path unique per worker instance."""

    if getattr(server_args, "benchmark_mode", None) is None:
        return
    if (
        getattr(server_args, "benchmark_output_path", None)
        != _DEFAULT_BENCHMARK_OUTPUT_PATH
    ):
        return
    short_id = str(worker_id)[-8:]
    server_args.benchmark_output_path = f"/tmp/benchmark_results_{short_id}.json"


def benchmark_config(server_args: Any) -> Optional[dict[str, Any]]:
    mode = getattr(server_args, "benchmark_mode", None)
    if mode is None:
        return None
    return {
        "mode": mode,
        "prefill_isl_granularity": getattr(
            server_args, "benchmark_prefill_granularity", 16
        ),
        "decode_length_granularity": getattr(
            server_args, "benchmark_decode_length_granularity", 6
        ),
        "decode_batch_size_granularity": getattr(
            server_args, "benchmark_decode_batch_granularity", 6
        ),
        "warmup_iterations": getattr(server_args, "benchmark_warmup_iterations", 5),
        "output_path": getattr(
            server_args, "benchmark_output_path", _DEFAULT_BENCHMARK_OUTPUT_PATH
        ),
        "timeout": getattr(server_args, "benchmark_timeout", 300),
    }


async def wait_and_load_benchmark(server_args: Any) -> dict[str, Any]:
    cfg = benchmark_config(server_args)
    if cfg is None:
        return {"status": "error", "message": "benchmark mode is not enabled"}

    base_path = Path(cfg["output_path"])
    timeout = int(cfg.get("timeout", 300))
    dp_start, dp_end = local_dp_rank_bounds(server_args)
    rank_paths = []
    for dp_rank in range(dp_start, dp_end):
        if dp_rank == 0:
            rank_paths.append(base_path)
        else:
            stem, ext = os.path.splitext(str(base_path))
            rank_paths.append(Path(f"{stem}_dp{dp_rank}{ext}"))

    logger.info(
        "Waiting for SGLang self-benchmark to complete (files: %s, timeout: %ds)",
        rank_paths,
        timeout,
    )
    deadline = time.monotonic() + timeout
    for path in rank_paths:
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"SGLang self-benchmark did not complete within {timeout}s. "
                    f"Missing: {path}"
                )
            await asyncio.sleep(0.1)

    merged: dict[str, Any] = {}
    for i, path in enumerate(rank_paths):
        with open(path) as f:
            data = json.load(f)
        dp_rank = dp_start + i
        if i == 0:
            merged = data
            for result in merged.get("results", []):
                result.setdefault("point", {})["dp_rank"] = dp_rank
        else:
            for result in data.get("results", []):
                result.setdefault("point", {})["dp_rank"] = dp_rank
            merged.setdefault("results", []).extend(data.get("results", []))

    logger.info(
        "SGLang self-benchmark complete, %d point(s) across %d rank(s)",
        len(merged.get("results", [])),
        len(rank_paths),
    )
    return merged


def _apply_int_env(
    parsed_args: Any, cli_args: list[str], attr: str, flag: str, env_var: str
) -> None:
    raw = os.environ.get(env_var)
    if raw is not None and not _has_cli_flag(cli_args, flag):
        setattr(parsed_args, attr, int(raw))


def _has_cli_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)
