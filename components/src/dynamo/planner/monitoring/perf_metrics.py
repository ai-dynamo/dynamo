# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-deployment FPM data fetching for planner regression bootstrap.

Priority chain:
  1. Call ``get_perf_metrics`` Dynamo endpoint (PR 7779 self-benchmark)
  2. Fallback: convert legacy profiler npz to synthetic FPMs
  3. If both fail: raise
"""

import asyncio
import json
import logging
import os
from typing import Optional

import numpy as np

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.monitoring.worker_info import WorkerInfo

logger = logging.getLogger(__name__)


async def fetch_pre_deployment_metrics(
    runtime: "object",  # DistributedRuntime; typed loosely to avoid hard import
    namespace: str,
    worker_info: WorkerInfo,
    profile_results_dir: Optional[str],
    component_type: SubComponentType,
) -> list[ForwardPassMetrics]:
    """Fetch pre-deployment engine perf data as an FPM list.

    1. Try ``get_perf_metrics`` endpoint (PR 7779 self-benchmark)
    2. Fallback: convert legacy profiler data (npz or JSON) to synthetic FPMs
    3. If both fail: raise

    Args:
        runtime: DistributedRuntime instance.
        namespace: Dynamo namespace.
        worker_info: WorkerInfo for the target component.
        profile_results_dir: Path to legacy profiler npz data (fallback).
        component_type: PREFILL or DECODE.

    Returns:
        List of ForwardPassMetrics suitable for regression bootstrap.
    """
    fpms = await _try_endpoint(runtime, namespace, worker_info, component_type)
    if fpms:
        logger.info(
            f"Loaded {len(fpms)} pre-deployment FPMs from get_perf_metrics endpoint"
        )
        return fpms

    if profile_results_dir:
        try:
            fpms = _convert_profiling_data_to_fpms(profile_results_dir, component_type)
            if fpms:
                logger.info(
                    f"Loaded {len(fpms)} FPMs from legacy profiler npz at {profile_results_dir}"
                )
                return fpms
        except Exception as e:
            logger.warning(
                f"Failed to load profiling npz from {profile_results_dir}: {e}"
            )

    raise RuntimeError(
        "Failed to obtain pre-deployment performance data. "
        "Either enable --benchmark-mode on the vLLM worker (get_perf_metrics endpoint) "
        "or provide profiling results via --profile-results-dir."
    )


async def _try_endpoint(
    runtime: "object",
    namespace: str,
    worker_info: WorkerInfo,
    component_type: SubComponentType,
) -> list[ForwardPassMetrics]:
    """Try to fetch benchmark FPMs from the get_perf_metrics Dynamo endpoint."""
    if not worker_info.component_name:
        return []

    try:
        endpoint = runtime.endpoint(  # type: ignore[attr-defined]
            f"{namespace}.{worker_info.component_name}.get_perf_metrics"
        )
        client = await endpoint.client()
        await asyncio.sleep(0.1)

        response_stream = await client.round_robin(None)
        benchmark_data = None
        async for resp in response_stream:
            benchmark_data = resp.data()
            break

        if benchmark_data is None:
            return []

        if isinstance(benchmark_data, str):
            benchmark_data = json.loads(benchmark_data)

        if isinstance(benchmark_data, dict) and benchmark_data.get("status") == "error":
            logger.info(
                f"get_perf_metrics returned error: {benchmark_data.get('message')}"
            )
            return []

        fpms = _extract_fpms_from_benchmark(benchmark_data, component_type)
        if not fpms:
            logger.warning(
                "get_perf_metrics returned data but no valid FPMs were extracted "
                "(possible schema mismatch)"
            )
        return fpms
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.info(f"get_perf_metrics endpoint not available: {e}")
        return []
    except Exception as e:
        logger.warning(f"get_perf_metrics unexpected error: {e}")
        return []


def _extract_fpms_from_benchmark(
    benchmark_data: dict,
    component_type: SubComponentType,
) -> list[ForwardPassMetrics]:
    """Extract ForwardPassMetrics from PR 7779 benchmark results dict."""
    import msgspec

    results = benchmark_data.get("results", [])
    fpms: list[ForwardPassMetrics] = []

    target_types: set[str] = set()
    if component_type == SubComponentType.PREFILL:
        target_types = {"prefill"}
    elif component_type == SubComponentType.DECODE:
        target_types = {"decode"}
    else:
        target_types = {"prefill", "decode"}

    for result in results:
        point = result.get("point", {})
        point_type = point.get("point_type", "")
        if point_type not in target_types:
            continue
        for fpm_dict in result.get("fpms", []):
            try:
                raw = json.dumps(fpm_dict).encode()
                fpm = msgspec.json.decode(raw, type=ForwardPassMetrics)
                if fpm.wall_time > 0:
                    fpms.append(fpm)
            except Exception as e:
                logger.warning(f"Failed to decode FPM entry: {e}")
                continue

    return fpms


def _convert_profiling_data_to_fpms(
    profile_results_dir: str,
    component_type: SubComponentType,
) -> list[ForwardPassMetrics]:
    """Convert legacy profiler data (npz or JSON) to synthetic ForwardPassMetrics."""
    fpms: list[ForwardPassMetrics] = []

    if component_type in (SubComponentType.PREFILL,):
        fpms.extend(_convert_prefill_profiling(profile_results_dir))
    if component_type in (SubComponentType.DECODE,):
        fpms.extend(_convert_decode_profiling(profile_results_dir))

    return fpms


def _convert_prefill_profiling(profile_results_dir: str) -> list[ForwardPassMetrics]:
    npz_path = os.path.join(
        profile_results_dir, "selected_prefill_interpolation", "raw_data.npz"
    )
    json_path = os.path.join(profile_results_dir, "prefill_raw_data.json")

    prefill_isl: np.ndarray
    prefill_ttft: np.ndarray

    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            prefill_isl = data["prefill_isl"]
            prefill_ttft = data["prefill_ttft"]
    elif os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            prefill_isl = np.array(data["prefill_isl"])
            prefill_ttft = np.array(data["prefill_ttft"])
    else:
        raise FileNotFoundError(
            f"Prefill profiling data not found at {npz_path} or {json_path}"
        )

    fpms = []
    for isl_val, ttft_ms in zip(prefill_isl, prefill_ttft):
        fpms.append(
            ForwardPassMetrics(
                wall_time=float(ttft_ms) / 1000.0,
                scheduled_requests=ScheduledRequestMetrics(
                    num_prefill_requests=1,
                    sum_prefill_tokens=int(isl_val),
                ),
            )
        )
    return fpms


def _convert_decode_profiling(profile_results_dir: str) -> list[ForwardPassMetrics]:
    npz_path = os.path.join(
        profile_results_dir, "selected_decode_interpolation", "raw_data.npz"
    )
    json_path = os.path.join(profile_results_dir, "decode_raw_data.json")

    x_kv_usage: np.ndarray
    y_context_length: np.ndarray
    z_itl: np.ndarray
    max_kv_tokens: int

    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            x_kv_usage = data["x_kv_usage"]
            y_context_length = data["y_context_length"]
            z_itl = data["z_itl"]
            max_kv_tokens = int(data["max_kv_tokens"][0])
    elif os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            x_kv_usage = np.array(data["x_kv_usage"])
            y_context_length = np.array(data["y_context_length"])
            z_itl = np.array(data["z_itl"])
            max_kv_tokens = int(data["max_kv_tokens"])
    else:
        raise FileNotFoundError(
            f"Decode profiling data not found at {npz_path} or {json_path}"
        )

    fpms = []
    for kv_usage, ctx_len, itl_ms in zip(x_kv_usage, y_context_length, z_itl):
        sum_decode_kv = int(round(float(kv_usage) * max_kv_tokens))
        batch_size = (
            max(1, int(round(sum_decode_kv / float(ctx_len)))) if ctx_len > 0 else 1
        )
        fpms.append(
            ForwardPassMetrics(
                wall_time=float(itl_ms) / 1000.0,
                scheduled_requests=ScheduledRequestMetrics(
                    num_decode_requests=batch_size,
                    sum_decode_kv_tokens=sum_decode_kv,
                ),
            )
        )
    return fpms
