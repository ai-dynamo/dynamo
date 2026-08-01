# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay


def _normalize_trace_files(trace_files):
    if isinstance(trace_files, (str, os.PathLike)):
        return [trace_files]
    return list(trace_files)


def _planner_config_arg(planner_config):
    """Normalize a planner config to the JSON-string form ``_prepare_planner_replay``
    expects: a dict is json-encoded; a str (path or inline JSON) passes through."""
    if isinstance(planner_config, dict):
        return json.dumps(planner_config)
    return planner_config


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
):
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
    }
    if planner_config is not None:
        # Planner replay is offline-only; reject controls the
        # planner path ignores so callers fail fast instead of silently getting an
        # offline planner run (matches the CLI's guardrails).
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        if trace_format not in ("mooncake", "dynamo"):
            raise ValueError(
                "planner_config replay only supports trace_format='mooncake' or 'dynamo'"
            )
        if report_jsonl_path is not None:
            raise ValueError("report_jsonl_path is not supported with planner_config")
        if max_sim_time_ms is not None:
            raise ValueError("max_sim_time_ms is not supported with planner_config")
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
        from dynamo.replay.main import _planner_replay_adapter

        adapter_scope = _planner_replay_adapter(
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            planner_config_arg=_planner_config_arg(planner_config),
            benchmark_granularity=benchmark_granularity,
        )
        with adapter_scope as adapter:
            trace_report = _run_mocker_trace_replay(
                trace_files,
                **replay_kwargs,
                scaling_policy=adapter,
            )
            return adapter.finalize(trace_report)
    return _run_mocker_trace_replay(
        trace_files,
        **replay_kwargs,
        scaling_policy=None,
    )


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
):
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
    }
    if planner_config is not None:
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        from dynamo.replay.main import _planner_replay_adapter

        adapter_scope = _planner_replay_adapter(
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            planner_config_arg=_planner_config_arg(planner_config),
            benchmark_granularity=benchmark_granularity,
        )
        with adapter_scope as adapter:
            trace_report = _run_mocker_synthetic_trace_replay(
                input_tokens,
                output_tokens,
                request_count,
                **replay_kwargs,
                scaling_policy=adapter,
            )
            return adapter.finalize(trace_report)
    return _run_mocker_synthetic_trace_replay(
        input_tokens,
        output_tokens,
        request_count,
        **replay_kwargs,
        scaling_policy=None,
    )
