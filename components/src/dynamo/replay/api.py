# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry points spanning shared offline and Dynamo online replay."""

import json
import os
from typing import Any, Literal, TypedDict, overload

from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
from dynamo.replay.report import PlannerReplayDetails, ReplayReport
from typing_extensions import Unpack


class _CommonReplayOptions(TypedDict, total=False):
    extra_engine_args: Any
    prefill_engine_args: Any
    decode_engine_args: Any
    router_config: Any
    aic_perf_config: Any
    num_workers: int
    num_prefill_workers: int
    num_decode_workers: int
    replay_concurrency: int | None
    router_mode: Literal["round_robin", "kv_router"]
    arrival_speedup_ratio: float
    model_name: str | None
    sla_ttft_ms: float | None
    sla_itl_ms: float | None
    sla_e2e_ms: float | None
    planner_config: Any
    benchmark_granularity: int
    capture_per_request: bool
    capture_planner_details: bool


class _TraceReplayOptions(_CommonReplayOptions, total=False):
    trace_block_size: int | None
    trace_format: str
    trace_shared_prefix_ratio: float
    trace_num_prefix_groups: int
    report_jsonl_path: str | os.PathLike[str] | None
    max_sim_time_ms: float | None


class _SyntheticReplayOptions(_CommonReplayOptions, total=False):
    request_rate: float | None
    arrival_interval_ms: float | None
    arrival_seed: int
    turns_per_session: int
    shared_prefix_ratio: float
    num_prefix_groups: int
    inter_turn_delay_ms: float


def _normalize_trace_files(trace_files):
    if isinstance(trace_files, (str, os.PathLike)):
        return [trace_files]
    return list(trace_files)


def _planner_config_arg(planner_config):
    """Normalize a planner config to the JSON form ``prepare_planner_replay``
    expects: a dict is json-encoded; a str (path or inline JSON) passes through."""
    if isinstance(planner_config, dict):
        return json.dumps(planner_config)
    return planner_config


def _materialize_offline_report(
    native,
    *,
    planner: PlannerReplayDetails | None,
) -> ReplayReport:
    return ReplayReport(
        summary=native.summary,
        per_request=native.per_request,
        coverage=native.coverage,
        planner=planner,
    )


@overload
def run_trace_replay(
    trace_files,
    *,
    replay_mode: Literal["offline"] = "offline",
    **kwargs: Unpack[_TraceReplayOptions],
) -> ReplayReport:
    ...


@overload
def run_trace_replay(
    trace_files,
    *,
    replay_mode: Literal["online"],
    **kwargs: Unpack[_TraceReplayOptions],
) -> dict[str, Any]:
    ...


@overload
def run_trace_replay(
    trace_files,
    *,
    replay_mode: str,
    **kwargs: Unpack[_TraceReplayOptions],
) -> ReplayReport | dict[str, Any]:
    ...


def run_trace_replay(
    trace_files,
    *,
    extra_engine_args=None,
    prefill_engine_args=None,
    decode_engine_args=None,
    router_config=None,
    aic_perf_config=None,
    num_workers=1,
    num_prefill_workers=1,
    num_decode_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
    trace_block_size=None,
    trace_format="mooncake",
    trace_shared_prefix_ratio=0.0,
    trace_num_prefix_groups=0,
    report_jsonl_path=None,
    max_sim_time_ms=None,
    model_name=None,
    sla_ttft_ms=None,
    sla_itl_ms=None,
    sla_e2e_ms=None,
    planner_config=None,
    benchmark_granularity=8,
    capture_per_request=False,
    capture_planner_details=True,
) -> ReplayReport | dict[str, Any]:
    """Run trace replay.

    ``wall_time_ms`` and derived throughput measure Rust runtime construction
    and execution. Planner creation and bootstrap happen before that boundary.
    """
    trace_files = _normalize_trace_files(trace_files)
    replay_kwargs = {
        "extra_engine_args": extra_engine_args,
        "prefill_engine_args": prefill_engine_args,
        "decode_engine_args": decode_engine_args,
        "router_config": router_config,
        "aic_perf_config": aic_perf_config,
        "num_workers": num_workers,
        "num_prefill_workers": num_prefill_workers,
        "num_decode_workers": num_decode_workers,
        "replay_concurrency": replay_concurrency,
        "replay_mode": replay_mode,
        "router_mode": router_mode,
        "arrival_speedup_ratio": arrival_speedup_ratio,
        "trace_block_size": trace_block_size,
        "trace_format": trace_format,
        "trace_shared_prefix_ratio": trace_shared_prefix_ratio,
        "trace_num_prefix_groups": trace_num_prefix_groups,
        "report_jsonl_path": report_jsonl_path,
        "max_sim_time_ms": max_sim_time_ms,
        "model_name": model_name,
        "sla_ttft_ms": sla_ttft_ms,
        "sla_itl_ms": sla_itl_ms,
        "sla_e2e_ms": sla_e2e_ms,
        "capture_per_request": capture_per_request,
        "capture_planner_details": capture_planner_details,
    }
    if capture_per_request and replay_mode == "online":
        raise ValueError(
            "capture_per_request only supports replay_mode='offline'; "
            "use report_jsonl_path for online request records"
        )
    if planner_config is not None:
        # Planner replay is offline-only; reject controls the
        # planner path ignores so callers fail fast instead of silently getting an
        # offline planner run (matches the CLI's guardrails).
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        if trace_format not in (
            "mooncake",
            "applied_compute_agentic",
            "dynamo",
        ):
            raise ValueError(
                "planner_config replay only supports trace_format='mooncake', "
                "'applied_compute_agentic', or 'dynamo'"
            )
        if trace_format != "dynamo" and len(trace_files) != 1:
            raise ValueError(
                f"planner_config replay with trace_format={trace_format!r} "
                "requires exactly one trace file"
            )
        if trace_format == "dynamo" and not trace_files:
            raise ValueError(
                "planner_config replay with trace_format='dynamo' "
                "requires at least one trace file"
            )
        from dynamo.replay.planner import planner_replay_adapter

        adapter_scope = planner_replay_adapter(
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            planner_config_arg=_planner_config_arg(planner_config),
            benchmark_granularity=benchmark_granularity,
            capture_details=capture_planner_details,
        )
        with adapter_scope as adapter:
            native = _run_mocker_trace_replay(
                trace_files,
                **replay_kwargs,
                scaling_policy=adapter,
            )
            return _materialize_offline_report(
                native,
                planner=adapter.finalize(native.lifecycle_operations),
            )
    result = _run_mocker_trace_replay(
        trace_files,
        **replay_kwargs,
        scaling_policy=None,
    )
    if replay_mode == "online":
        return result
    return _materialize_offline_report(result, planner=None)


@overload
def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    replay_mode: Literal["offline"] = "offline",
    **kwargs: Unpack[_SyntheticReplayOptions],
) -> ReplayReport:
    ...


@overload
def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    replay_mode: Literal["online"],
    **kwargs: Unpack[_SyntheticReplayOptions],
) -> dict[str, Any]:
    ...


@overload
def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    replay_mode: str,
    **kwargs: Unpack[_SyntheticReplayOptions],
) -> ReplayReport | dict[str, Any]:
    ...


def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    extra_engine_args=None,
    prefill_engine_args=None,
    decode_engine_args=None,
    router_config=None,
    aic_perf_config=None,
    num_workers=1,
    num_prefill_workers=1,
    num_decode_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
    request_rate=None,
    arrival_interval_ms=None,
    arrival_seed=42,
    turns_per_session=1,
    shared_prefix_ratio=0.0,
    num_prefix_groups=0,
    inter_turn_delay_ms=0.0,
    model_name=None,
    sla_ttft_ms=None,
    sla_itl_ms=None,
    sla_e2e_ms=None,
    planner_config=None,
    benchmark_granularity=8,
    capture_per_request=False,
    capture_planner_details=True,
) -> ReplayReport | dict[str, Any]:
    """Run synthetic replay with the same timing boundary as trace replay."""
    replay_kwargs = {
        "extra_engine_args": extra_engine_args,
        "prefill_engine_args": prefill_engine_args,
        "decode_engine_args": decode_engine_args,
        "router_config": router_config,
        "aic_perf_config": aic_perf_config,
        "num_workers": num_workers,
        "num_prefill_workers": num_prefill_workers,
        "num_decode_workers": num_decode_workers,
        "replay_concurrency": replay_concurrency,
        "replay_mode": replay_mode,
        "router_mode": router_mode,
        "arrival_speedup_ratio": arrival_speedup_ratio,
        "request_rate": request_rate,
        "arrival_interval_ms": arrival_interval_ms,
        "arrival_seed": arrival_seed,
        "turns_per_session": turns_per_session,
        "shared_prefix_ratio": shared_prefix_ratio,
        "num_prefix_groups": num_prefix_groups,
        "inter_turn_delay_ms": inter_turn_delay_ms,
        "model_name": model_name,
        "sla_ttft_ms": sla_ttft_ms,
        "sla_itl_ms": sla_itl_ms,
        "sla_e2e_ms": sla_e2e_ms,
        "capture_per_request": capture_per_request,
        "capture_planner_details": capture_planner_details,
    }
    if capture_per_request and replay_mode == "online":
        raise ValueError("capture_per_request only supports replay_mode='offline'")
    if planner_config is not None:
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        from dynamo.replay.planner import planner_replay_adapter

        adapter_scope = planner_replay_adapter(
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            planner_config_arg=_planner_config_arg(planner_config),
            benchmark_granularity=benchmark_granularity,
            capture_details=capture_planner_details,
        )
        with adapter_scope as adapter:
            native = _run_mocker_synthetic_trace_replay(
                input_tokens,
                output_tokens,
                request_count,
                **replay_kwargs,
                scaling_policy=adapter,
            )
            return _materialize_offline_report(
                native,
                planner=adapter.finalize(native.lifecycle_operations),
            )
    result = _run_mocker_synthetic_trace_replay(
        input_tokens,
        output_tokens,
        request_count,
        **replay_kwargs,
        scaling_policy=None,
    )
    if replay_mode == "online":
        return result
    return _materialize_offline_report(result, planner=None)
