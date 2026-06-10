# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM startup self-benchmark integration helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)

logger = logging.getLogger(__name__)

DEFAULT_PREFILL_ISL_GRANULARITY = 16
DEFAULT_DECODE_CONTEXT_GRANULARITY = 6
DEFAULT_DECODE_BATCH_GRANULARITY = 6
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_TIMEOUT_S = 300
TRTLLM_SELF_BENCHMARK_RUNTIME_KEY = "trtllm_self_benchmark"

_OPTION_ATTRS = {
    "benchmark_prefill_isl_granularity": "--benchmark-prefill-granularity",
    "benchmark_decode_context_granularity": "--benchmark-decode-length-granularity",
    "benchmark_decode_batch_granularity": "--benchmark-decode-batch-granularity",
    "benchmark_warmup_iterations": "--benchmark-warmup-iterations",
    "benchmark_output_path": "--benchmark-output-path",
    "benchmark_timeout_s": "--benchmark-timeout",
}


@dataclass(frozen=True)
class TrtllmSelfBenchmarkConfig:
    """Backend-neutral Dynamo config resolved to TRT-LLM's native surface."""

    mode: str
    prefill_isl_granularity: int = DEFAULT_PREFILL_ISL_GRANULARITY
    decode_context_granularity: int = DEFAULT_DECODE_CONTEXT_GRANULARITY
    decode_batch_granularity: int = DEFAULT_DECODE_BATCH_GRANULARITY
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    output_path: str = ""
    timeout_s: int = DEFAULT_TIMEOUT_S


def benchmark_options_without_mode(config: Any) -> list[str]:
    """Return self-benchmark option flags set while benchmark mode is absent."""

    if getattr(config, "benchmark_mode", None) is not None:
        return []
    return [
        flag
        for attr, flag in _OPTION_ATTRS.items()
        if getattr(config, attr, None) not in (None, "")
    ]


def build_self_benchmark_config(config: Any) -> TrtllmSelfBenchmarkConfig | None:
    """Resolve Dynamo benchmark config to a concrete TRT-LLM config."""

    mode = getattr(config, "benchmark_mode", None)
    if mode is None:
        unexpected = benchmark_options_without_mode(config)
        if unexpected:
            raise ValueError(
                "Self-benchmark options require --benchmark-mode: "
                + ", ".join(unexpected)
            )
        return None

    output_path = getattr(config, "benchmark_output_path", None) or str(
        Path(tempfile.gettempdir())
        / f"trtllm_self_benchmark_{os.getpid()}_{uuid.uuid4().hex}.json"
    )
    cfg = TrtllmSelfBenchmarkConfig(
        mode=mode,
        prefill_isl_granularity=_positive_int_or_default(
            getattr(config, "benchmark_prefill_isl_granularity", None),
            DEFAULT_PREFILL_ISL_GRANULARITY,
            "benchmark_prefill_isl_granularity",
        ),
        decode_context_granularity=_positive_int_or_default(
            getattr(config, "benchmark_decode_context_granularity", None),
            DEFAULT_DECODE_CONTEXT_GRANULARITY,
            "benchmark_decode_context_granularity",
        ),
        decode_batch_granularity=_positive_int_or_default(
            getattr(config, "benchmark_decode_batch_granularity", None),
            DEFAULT_DECODE_BATCH_GRANULARITY,
            "benchmark_decode_batch_granularity",
        ),
        warmup_iterations=_non_negative_int_or_default(
            getattr(config, "benchmark_warmup_iterations", None),
            DEFAULT_WARMUP_ITERATIONS,
            "benchmark_warmup_iterations",
        ),
        output_path=output_path,
        timeout_s=_positive_int_or_default(
            getattr(config, "benchmark_timeout_s", None),
            DEFAULT_TIMEOUT_S,
            "benchmark_timeout_s",
        ),
    )
    return cfg


def prepare_self_benchmark(
    config: Any, engine_args: dict[str, Any]
) -> TrtllmSelfBenchmarkConfig | None:
    """Validate, clean stale output, and inject TRT-LLM benchmark args."""

    cfg = build_self_benchmark_config(config)
    if cfg is None:
        return None
    validate_self_benchmark_supported(config, engine_args)
    remove_self_benchmark_output(cfg.output_path)
    apply_self_benchmark_engine_args(engine_args, cfg)
    return cfg


def validate_self_benchmark_supported(config: Any, engine_args: dict[str, Any]) -> None:
    """Fail fast for TRT-LLM modes that do not support self-benchmarking."""

    disaggregation_mode = getattr(config, "disaggregation_mode", None)
    if getattr(disaggregation_mode, "name", None) == "ENCODE":
        raise ValueError("TRT-LLM self-benchmark is not supported for encode workers")
    if bool(engine_args.get("enable_attention_dp", False)):
        raise ValueError(
            "TRT-LLM self-benchmark is not supported with enable_attention_dp"
        )
    for unsupported in ("encode_only", "mm_encoder_only"):
        if bool(engine_args.get(unsupported, False)):
            raise ValueError(
                f"TRT-LLM self-benchmark is not supported with {unsupported}"
            )


def apply_self_benchmark_engine_args(
    engine_args: dict[str, Any], cfg: TrtllmSelfBenchmarkConfig
) -> None:
    """Inject TRT-LLM native self-benchmark kwargs into LLM engine args."""

    engine_args.update(
        {
            "self_benchmark_mode": cfg.mode,
            "self_benchmark_prefill_granularity": cfg.prefill_isl_granularity,
            "self_benchmark_decode_length_granularity": cfg.decode_context_granularity,
            "self_benchmark_decode_batch_granularity": cfg.decode_batch_granularity,
            "self_benchmark_warmup_iterations": cfg.warmup_iterations,
            "self_benchmark_output_path": cfg.output_path,
            "self_benchmark_timeout": cfg.timeout_s,
        }
    )


def trtllm_self_benchmark_cli_args(cfg: TrtllmSelfBenchmarkConfig) -> list[str]:
    """Return the equivalent ``trtllm-serve`` CLI arguments."""

    return [
        "--self_benchmark_mode",
        cfg.mode,
        "--self_benchmark_prefill_granularity",
        str(cfg.prefill_isl_granularity),
        "--self_benchmark_decode_length_granularity",
        str(cfg.decode_context_granularity),
        "--self_benchmark_decode_batch_granularity",
        str(cfg.decode_batch_granularity),
        "--self_benchmark_warmup_iterations",
        str(cfg.warmup_iterations),
        "--self_benchmark_output_path",
        cfg.output_path,
        "--self_benchmark_timeout",
        str(cfg.timeout_s),
    ]


def remove_self_benchmark_output(output_path: str) -> None:
    """Remove a stale rank-0 benchmark result before TRT-LLM startup."""

    try:
        Path(output_path).unlink()
    except FileNotFoundError:
        return


async def wait_for_self_benchmark_output(
    cfg: TrtllmSelfBenchmarkConfig,
    *,
    worker_id: str = "",
) -> dict[str, Any]:
    """Poll, parse, and normalize TRT-LLM's rank-0 benchmark JSON output."""

    output = Path(cfg.output_path)
    deadline = time.monotonic() + cfg.timeout_s
    logger.info(
        "Waiting for TRT-LLM self-benchmark output (file=%s, timeout=%ss)",
        output,
        cfg.timeout_s,
    )
    while not output.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"TRT-LLM self-benchmark did not write {output} within "
                f"{cfg.timeout_s}s"
            )
        await asyncio.sleep(0.1)

    try:
        with output.open() as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid TRT-LLM self-benchmark JSON in {output}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid TRT-LLM self-benchmark JSON in {output}: root must be an object"
        )

    normalized = normalize_self_benchmark_payload(payload, worker_id=worker_id)
    logger.info(
        "Loaded TRT-LLM self-benchmark output (points=%d, timed_out=%s)",
        len(normalized.get("results", [])),
        normalized.get("metadata", {}).get("timed_out", False),
    )
    return normalized


def normalize_self_benchmark_payload(
    payload: dict[str, Any],
    *,
    worker_id: str = "",
) -> dict[str, Any]:
    """Convert TRT-LLM JSON into Dynamo's benchmark ``results[].fpms[]`` shape."""

    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("TRT-LLM self-benchmark JSON field results must be a list")

    warnings: list[str] = []
    if payload.get("timed_out") is True:
        warnings.append("TRT-LLM self-benchmark timed out; publishing partial data")
        logger.warning(warnings[-1])

    results: list[dict[str, Any]] = []
    profile: dict[str, dict[str, Any]] = {"prefill": {}, "decode": {}}
    skipped_points: list[dict[str, Any]] = []

    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        point = raw_result.get("point") or {}
        if not isinstance(point, dict):
            continue
        point_type = str(point.get("point_type", ""))
        skipped_reason = raw_result.get("skipped_reason")
        iteration_stats = raw_result.get("iteration_stats") or []
        if not isinstance(iteration_stats, list):
            iteration_stats = []

        result = dict(raw_result)
        result["point"] = dict(point)
        result["iteration_stats"] = iteration_stats
        result["fpms"] = []
        normalized_key = _normalized_key(point)
        if normalized_key is not None:
            result["normalized_key"] = normalized_key

        if skipped_reason:
            skipped_points.append(
                {
                    "point": dict(point),
                    "skipped_reason": skipped_reason,
                    "normalized_key": normalized_key,
                }
            )
            results.append(result)
            continue

        fpms = [
            _fpm_to_dict(_fpm_from_iteration_stat(point, stat, worker_id))
            for stat in iteration_stats
            if isinstance(stat, dict)
        ]
        result["fpms"] = fpms
        results.append(result)

        if normalized_key is None:
            continue
        table_name = "prefill" if point_type == "prefill" else "decode"
        if table_name not in profile:
            continue
        profile[table_name][_profile_key(point)] = {
            "key": normalized_key,
            "point": dict(point),
            "fpms": fpms,
            "iteration_stats": iteration_stats,
        }

    return {
        "status": "ok",
        "backend": "trtllm",
        "results": results,
        "profile": profile,
        "metadata": {
            "backend": "trtllm",
            "timed_out": payload.get("timed_out") is True,
            "warnings": warnings,
            "skipped_points": skipped_points,
            "raw_trtllm_self_benchmark": payload,
        },
    }


def _fpm_from_iteration_stat(
    point: dict[str, Any], stat: dict[str, Any], worker_id: str
) -> ForwardPassMetrics:
    point_type = str(point.get("point_type", ""))
    batch_size = _int_or(point.get("batch_size"), 1)
    isl = _int_or(point.get("isl"), 0)
    context_length = _int_or(point.get("context_length"), 0)
    ibs = stat.get("inflightBatchingStats") or {}
    if not isinstance(ibs, dict):
        ibs = {}

    queued_num_decode = _int_or(ibs.get("numPausedRequests"), 0) + _int_or(
        ibs.get("numQueuedGenRequests"), 0
    )
    queued_sum_decode_kv_tokens = _int_or(ibs.get("numPausedKvTokens"), 0) + _int_or(
        ibs.get("numQueuedGenKvTokens"), 0
    )
    scheduled = ScheduledRequestMetrics(
        num_prefill_requests=_int_or(
            ibs.get("numContextRequests"),
            batch_size if point_type == "prefill" else 0,
        ),
        sum_prefill_tokens=_int_or(
            ibs.get("numCtxTokens"),
            isl * batch_size if point_type == "prefill" else 0,
        ),
        sum_prefill_kv_tokens=_int_or(
            ibs.get("numCtxKvTokens"),
            context_length * batch_size if point_type == "prefill" else 0,
        ),
        num_decode_requests=_int_or(
            ibs.get("numGenRequests"),
            batch_size if point_type == "decode" else 0,
        ),
        sum_decode_kv_tokens=_int_or(
            ibs.get("numGenKvTokens"),
            context_length * batch_size if point_type == "decode" else 0,
        ),
    )
    queued = QueuedRequestMetrics(
        num_prefill_requests=_int_or(ibs.get("numQueuedContextRequests"), 0),
        sum_prefill_tokens=_int_or(ibs.get("numQueuedCtxTokens"), 0),
        num_decode_requests=queued_num_decode,
        sum_decode_kv_tokens=queued_sum_decode_kv_tokens,
    )
    attention_dp_rank = stat.get("attentionDpRank")
    dp_rank = _int_or(attention_dp_rank, 0)
    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        wall_time=_wall_time_secs(stat),
        scheduled_requests=scheduled,
        queued_requests=queued,
    )


def _wall_time_secs(stat: dict[str, Any]) -> float:
    for key in ("hostStepTimeMS", "iterLatencyMS", "prevDeviceStepTimeMS"):
        value = stat.get(key)
        if isinstance(value, (int, float)):
            return float(value) / 1000.0
    return 0.0


def _fpm_to_dict(fpm: ForwardPassMetrics) -> dict[str, Any]:
    return msgspec.to_builtins(fpm)


def _normalized_key(point: dict[str, Any]) -> int | list[int] | None:
    point_type = str(point.get("point_type", ""))
    if point_type == "prefill":
        return _int_or(point.get("isl"), 0)
    if point_type == "decode":
        return [
            _int_or(point.get("context_length"), 0),
            _int_or(point.get("batch_size"), 1),
        ]
    return None


def _profile_key(point: dict[str, Any]) -> str:
    point_type = str(point.get("point_type", ""))
    if point_type == "prefill":
        return str(_int_or(point.get("isl"), 0))
    return f"{_int_or(point.get('context_length'), 0)},{_int_or(point.get('batch_size'), 1)}"


def _int_or(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_int_or_default(value: Any, default: int, name: str) -> int:
    resolved = default if value is None else _int_or(value, default)
    if resolved <= 0:
        raise ValueError(f"{name} must be > 0")
    return resolved


def _non_negative_int_or_default(value: Any, default: int, name: str) -> int:
    resolved = default if value is None else _int_or(value, default)
    if resolved < 0:
        raise ValueError(f"{name} must be >= 0")
    return resolved
