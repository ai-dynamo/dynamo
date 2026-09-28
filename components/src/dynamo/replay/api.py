# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry points spanning shared offline and Dynamo online replay."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, overload

from typing_extensions import Unpack

from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
from dynamo.replay.report import (
    PlannerReplayDetails,
    ReplayReport,
    ReplayTelemetryDetails,
)


def _planner_replay_adapter():
    """Load Planner replay lazily to break the replay.api/mocker import cycle.

    ``dynamo.replay.planner`` imports ``dynamo.mocker``, whose package
    initializer imports ``dynamo.replay.api`` for compatibility wrappers.
    Importing Planner at module scope therefore fails in spawned workers while
    this module is still partially initialized.
    """

    from dynamo.replay.planner import planner_replay_adapter

    return planner_replay_adapter


@dataclass(frozen=True)
class TelemetryOptions:
    """Optional policy-neutral telemetry for an offline replay.

    In-memory capture is the default only when no callback or JSONL sink is set.
    """

    sample_interval_ms: float = 1_000.0
    capture_in_memory: bool | None = None
    callback: Callable[[dict[str, Any]], None] | None = None
    jsonl_path: str | os.PathLike[str] | None = None


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
    performance_model_metadata: dict[str, Any] | None
    benchmark_granularity: int
    capture_per_request: bool
    capture_planner_details: bool
    telemetry_options: TelemetryOptions | None


class _TraceReplayOptions(_CommonReplayOptions, total=False):
    agentic_lanes: int | None
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
    native_telemetry = native.telemetry
    telemetry = (
        None
        if native_telemetry is None
        else ReplayTelemetryDetails(
            sample_interval_ms=float(native_telemetry["sample_interval_ms"]),
            samples=list(native_telemetry["samples"]),
        )
    )
    return ReplayReport(
        summary=native.summary,
        per_request=native.per_request,
        coverage=native.coverage,
        planner=planner,
        telemetry=telemetry,
    )


def _telemetry_kwargs(options: TelemetryOptions | None) -> dict[str, Any]:
    if options is None:
        return {}
    capture_in_memory = options.capture_in_memory
    if capture_in_memory is None:
        capture_in_memory = options.callback is None and options.jsonl_path is None
    if (
        not capture_in_memory
        and options.callback is None
        and options.jsonl_path is None
    ):
        raise ValueError("TelemetryOptions needs at least one sink")
    return {
        "capture_telemetry": capture_in_memory,
        "telemetry_sample_interval_ms": options.sample_interval_ms,
        "telemetry_callback": options.callback,
        "telemetry_jsonl_path": options.jsonl_path,
    }


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
    agentic_lanes=None,
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
    performance_model_metadata=None,
    benchmark_granularity=8,
    capture_per_request=False,
    capture_planner_details=True,
    telemetry_options=None,
) -> ReplayReport | dict[str, Any]:
    """Run trace replay.

    ``wall_time_ms`` and derived throughput measure Rust runtime construction
    and execution. Planner creation and bootstrap happen before that boundary.

    Pass ``TelemetryOptions`` to enable policy-neutral sampling; omitting it
    leaves telemetry disabled. Callbacks and JSONL writes run synchronously on
    the replay loop, so their latency contributes to replay wall time. The
    final buffered-file flush happens after the simulator finalizes
    ``wall_time_ms``; time the outer API call when measuring end-to-end
    persistence overhead. JSONL output is opened lazily on the first sample.
    If a later write fails, replay fails; completed prior lines remain, and the
    failing final line may be partial.
    """
    if isinstance(agentic_lanes, bool) or (
        agentic_lanes is not None and not isinstance(agentic_lanes, int)
    ):
        raise TypeError("agentic_lanes must be an integer or None")
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
        "agentic_lanes": agentic_lanes,
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
    replay_kwargs.update(_telemetry_kwargs(telemetry_options))
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
        adapter_scope = _planner_replay_adapter()(
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            planner_config_arg=_planner_config_arg(planner_config),
            performance_model_metadata=performance_model_metadata,
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
    performance_model_metadata=None,
    benchmark_granularity=8,
    capture_per_request=False,
    capture_planner_details=True,
    telemetry_options=None,
) -> ReplayReport | dict[str, Any]:
    """Run synthetic replay with the same optional ``TelemetryOptions`` contract."""
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
    replay_kwargs.update(_telemetry_kwargs(telemetry_options))
    if capture_per_request and replay_mode == "online":
        raise ValueError("capture_per_request only supports replay_mode='offline'")
    if planner_config is not None:
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        adapter_scope = _planner_replay_adapter()(
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            planner_config_arg=_planner_config_arg(planner_config),
            performance_model_metadata=performance_model_metadata,
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
