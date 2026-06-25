# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay


def _planner_config_arg(planner_config):
    """Normalize a planner config to the JSON-string form ``_run_planner_replay``
    expects: a dict is json-encoded; a str (path or inline JSON) passes through."""
    if isinstance(planner_config, dict):
        return json.dumps(planner_config)
    return planner_config


def run_trace_replay(
    trace_file,
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
    trace_block_size=512,
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
    if planner_config is not None:
        # Planner replay is offline-only and Mooncake-only; reject controls the
        # planner path ignores so callers fail fast instead of silently getting an
        # offline planner run (matches the CLI's guardrails).
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        if trace_format != "mooncake":
            raise ValueError(
                "planner_config replay only supports trace_format='mooncake'"
            )
        if report_jsonl_path is not None:
            raise ValueError("report_jsonl_path is not supported with planner_config")
        if max_sim_time_ms is not None:
            raise ValueError("max_sim_time_ms is not supported with planner_config")
        # Planner-in-the-loop: the Rust bridge owns the sim loop and calls back into
        # the Python planner adapter once per PlannerTick (main._run_planner_replay),
        # returning a ReplayPlannerReport (its .trace_report matches the static dict).
        from dynamo.replay.main import _run_planner_replay

        return _run_planner_replay(
            trace_file=trace_file,
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            router_config=router_config,
            num_workers=num_workers,
            num_prefill_workers=num_prefill_workers,
            num_decode_workers=num_decode_workers,
            router_mode=router_mode,
            arrival_speedup_ratio=arrival_speedup_ratio,
            trace_block_size=trace_block_size,
            model_name=model_name,
            planner_config_arg=_planner_config_arg(planner_config),
            benchmark_granularity=benchmark_granularity,
            sla_ttft_ms=sla_ttft_ms,
            sla_itl_ms=sla_itl_ms,
            sla_e2e_ms=sla_e2e_ms,
            replay_concurrency=replay_concurrency,
        )
    return _run_mocker_trace_replay(
        trace_file,
        extra_engine_args=extra_engine_args,
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
        router_config=router_config,
        aic_perf_config=aic_perf_config,
        num_workers=num_workers,
        num_prefill_workers=num_prefill_workers,
        num_decode_workers=num_decode_workers,
        replay_concurrency=replay_concurrency,
        replay_mode=replay_mode,
        router_mode=router_mode,
        arrival_speedup_ratio=arrival_speedup_ratio,
        trace_block_size=trace_block_size,
        trace_format=trace_format,
        trace_shared_prefix_ratio=trace_shared_prefix_ratio,
        trace_num_prefix_groups=trace_num_prefix_groups,
        report_jsonl_path=report_jsonl_path,
        max_sim_time_ms=max_sim_time_ms,
        model_name=model_name,
        # Goodput SLA (offline replay only): when set, the report carries
        # goodput_* keys classifying SLA-satisfying requests.
        sla_ttft_ms=sla_ttft_ms,
        sla_itl_ms=sla_itl_ms,
        sla_e2e_ms=sla_e2e_ms,
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
    arrival_interval_ms=1.0,
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
    if planner_config is not None:
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        from dynamo.replay.main import SyntheticWorkload, _run_planner_replay

        return _run_planner_replay(
            trace_file=None,
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            router_config=router_config,
            num_workers=num_workers,
            num_prefill_workers=num_prefill_workers,
            num_decode_workers=num_decode_workers,
            router_mode=router_mode,
            arrival_speedup_ratio=arrival_speedup_ratio,
            trace_block_size=512,
            model_name=model_name,
            planner_config_arg=_planner_config_arg(planner_config),
            benchmark_granularity=benchmark_granularity,
            sla_ttft_ms=sla_ttft_ms,
            sla_itl_ms=sla_itl_ms,
            sla_e2e_ms=sla_e2e_ms,
            replay_concurrency=replay_concurrency,
            synthetic=SyntheticWorkload(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_count=request_count,
                arrival_interval_ms=arrival_interval_ms,
                turns_per_session=turns_per_session,
                shared_prefix_ratio=shared_prefix_ratio,
                num_prefix_groups=num_prefix_groups,
                inter_turn_delay_ms=inter_turn_delay_ms,
            ),
        )
    return _run_mocker_synthetic_trace_replay(
        input_tokens,
        output_tokens,
        request_count,
        extra_engine_args=extra_engine_args,
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
        router_config=router_config,
        aic_perf_config=aic_perf_config,
        num_workers=num_workers,
        num_prefill_workers=num_prefill_workers,
        num_decode_workers=num_decode_workers,
        replay_concurrency=replay_concurrency,
        replay_mode=replay_mode,
        router_mode=router_mode,
        arrival_speedup_ratio=arrival_speedup_ratio,
        arrival_interval_ms=arrival_interval_ms,
        turns_per_session=turns_per_session,
        shared_prefix_ratio=shared_prefix_ratio,
        num_prefix_groups=num_prefix_groups,
        inter_turn_delay_ms=inter_turn_delay_ms,
        model_name=model_name,
        # Goodput SLA for synthetic-static (offline): emits goodput_* in the report.
        sla_ttft_ms=sla_ttft_ms,
        sla_itl_ms=sla_itl_ms,
        sla_e2e_ms=sla_e2e_ms,
    )
