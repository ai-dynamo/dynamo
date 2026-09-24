# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry points spanning shared offline and Dynamo online replay."""

import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, TypedDict, overload

from packaging.version import InvalidVersion, Version
from typing_extensions import Unpack

from dynamo import _core
from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
from dynamo.replay.report import PlannerReplayDetails, ReplayReport


def _planner_replay_adapter():
    """Load Planner replay lazily to break the replay.api/mocker import cycle.

    ``dynamo.replay.planner`` imports ``dynamo.mocker``, whose package
    initializer imports ``dynamo.replay.api`` for compatibility wrappers.
    Importing Planner at module scope therefore fails in spawned workers while
    this module is still partially initialized.
    """

    from dynamo.replay.planner import planner_replay_adapter

    return planner_replay_adapter


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
    replay_spec_json: str,
    affinity: dict[str, Any] | None = None,
    **kwargs: Unpack[_TraceReplayOptions],
) -> str:
    ...


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
    replay_spec_json=None,
    affinity=None,
) -> ReplayReport | dict[str, Any] | str:
    """Run trace replay.

    ``wall_time_ms`` and derived throughput measure Rust runtime construction
    and execution. Planner creation and bootstrap happen before that boundary.

    ``replay_spec_json`` accepts the canonical AISimulate execution payload for
    static KV-router replay and returns canonical report JSON. The existing
    Dynamo runner uses this path for conversation affinity and AgentX profiles;
    traffic, engine and capture controls come exclusively from that payload.
    """
    if replay_spec_json is not None:
        if not isinstance(replay_spec_json, str):
            raise TypeError("replay_spec_json must be a JSON string")
        if replay_mode != "offline" or router_mode != "kv_router":
            raise ValueError("canonical replay requires offline kv_router mode")
        if (
            _normalize_trace_files(trace_files)
            or any(
                value is not None
                for value in (
                    extra_engine_args,
                    prefill_engine_args,
                    decode_engine_args,
                    replay_concurrency,
                    agentic_lanes,
                    planner_config,
                    max_sim_time_ms,
                    report_jsonl_path,
                    model_name,
                    sla_ttft_ms,
                    sla_itl_ms,
                    sla_e2e_ms,
                    trace_block_size,
                    performance_model_metadata,
                )
            )
            or (
                num_workers != 1
                or num_prefill_workers != 1
                or num_decode_workers != 1
                or arrival_speedup_ratio != 1.0
                or trace_format != "mooncake"
                or trace_shared_prefix_ratio != 0.0
                or trace_num_prefix_groups != 0
                or capture_per_request
                or not capture_planner_details
                or benchmark_granularity != 8
            )
        ):
            raise ValueError(
                "replay_spec_json owns traffic, engines and capture; legacy "
                "replay arguments and Planner scaling cannot be combined with it"
            )
        compiled_version = getattr(_core, "AISIMULATE_CORE_VERSION", None)
        replay_api_version = getattr(_core, "AISIMULATE_REPLAY_API_VERSION", None)
        try:
            installed_version = version("aisimulate")
            versions_match = Version(installed_version) == Version(
                str(compiled_version)
            )
        except (PackageNotFoundError, InvalidVersion) as error:
            raise ValueError(
                "Dynamo canonical replay requires valid matching AISimulate Python "
                f"and compiled core versions (compiled={compiled_version}). "
                "Install matching AISimulate and a Dynamo runtime built with "
                "--features aic-forward-pass."
            ) from error
        if (
            not versions_match
            or type(replay_api_version) is not int
            or replay_api_version != 1
        ):
            raise ValueError(
                "Dynamo canonical replay requires matching AISimulate Python and "
                "compiled core versions and native replay API 1; "
                f"installed={installed_version}, compiled={compiled_version}, "
                f"native_api={replay_api_version}. Install the matching Dynamo "
                "runtime built with --features aic-forward-pass."
            )
        return _run_mocker_trace_replay(
            [],
            router_mode=router_mode,
            router_config=router_config,
            aic_perf_config=aic_perf_config,
            replay_spec_json=replay_spec_json,
            affinity_json=(
                json.dumps(affinity, allow_nan=False) if affinity is not None else None
            ),
        )
    if affinity is not None:
        raise ValueError("affinity requires a canonical replay_spec_json payload")
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
