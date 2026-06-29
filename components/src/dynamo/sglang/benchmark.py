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
        "benchmark_prefill_kv_read_granularity",
        "--benchmark-prefill-kv-read-granularity",
        "DYN_BENCHMARK_PREFILL_KV_READ_GRANULARITY",
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
        "prefill_kv_read_granularity": getattr(
            server_args, "benchmark_prefill_kv_read_granularity", 1
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

    expected_model = getattr(server_args, "model_path", None)
    logger.info(
        "Waiting for SGLang self-benchmark to complete (files: %s, timeout: %ds)",
        rank_paths,
        timeout,
    )
    deadline = time.monotonic() + timeout
    loaded: list[tuple[int, dict[str, Any]]] = []
    for i, path in enumerate(rank_paths):
        data = await _await_completed_result(path, deadline, timeout, expected_model)
        loaded.append((dp_start + i, data))

    # The producer stamps every rank file with the current run_id and, at
    # startup, rewrites it to an invalid "running" sentinel, so a stale file
    # from a previous (possibly crashed) run is never accepted. As a final
    # cross-rank guard, all merged rank files must agree on the run_id.
    run_ids = {d.get("run_id") for _, d in loaded if d.get("run_id")}
    if len(run_ids) > 1:
        logger.warning(
            "SGLang self-benchmark rank files have mismatched run_ids %s; "
            "results may combine different runs",
            run_ids,
        )

    merged: dict[str, Any] = {}
    for i, (dp_rank, data) in enumerate(loaded):
        if i == 0:
            merged = data
            for result in merged.get("results", []):
                result.setdefault("point", {})["dp_rank"] = dp_rank
        else:
            for result in data.get("results", []):
                result.setdefault("point", {})["dp_rank"] = dp_rank
            merged.setdefault("results", []).extend(data.get("results", []))

    logger.info(
        "SGLang self-benchmark complete (run_id=%s), %d point(s) across %d rank(s)",
        merged.get("run_id"),
        len(merged.get("results", [])),
        len(rank_paths),
    )
    return merged


async def _await_completed_result(
    path: Path, deadline: float, timeout: int, expected_model: Any
) -> dict[str, Any]:
    """Poll until ``path`` holds a COMPLETED self-benchmark result.

    The producer writes an invalid "running" sentinel at startup and atomically
    replaces it with the completed results when the sweep finishes, so we wait
    for ``status == "complete"`` rather than mere file existence; otherwise we
    would accept the in-progress sentinel (or a stale prior-run file).
    """
    while True:
        data = _read_json(path)
        if data is not None and _is_completed_result(data):
            _warn_on_model_mismatch(path, data, expected_model)
            return data
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"SGLang self-benchmark did not complete within {timeout}s. "
                f"Incomplete or missing: {path}"
            )
        await asyncio.sleep(0.1)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Mid-write or transiently unreadable; treat as not-ready and retry.
        return None


def _is_completed_result(data: dict[str, Any]) -> bool:
    status = data.get("status")
    if status == "complete":
        return data.get("valid", True) is not False
    # Backward compatibility with producers that predate the status field.
    return status is None and "results" in data


def _warn_on_model_mismatch(
    path: Path, data: dict[str, Any], expected_model: Any
) -> None:
    if expected_model is None:
        return
    actual = data.get("identity", {}).get("model_path")
    if actual is not None and actual != expected_model:
        logger.warning(
            "SGLang self-benchmark file %s was produced for model %r but this "
            "worker serves %r; results may be stale or misrouted",
            path,
            actual,
            expected_model,
        )


def _apply_int_env(
    parsed_args: Any, cli_args: list[str], attr: str, flag: str, env_var: str
) -> None:
    raw = os.environ.get(env_var)
    if raw is not None and not _has_cli_flag(cli_args, flag):
        setattr(parsed_args, attr, int(raw))


def _has_cli_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)
