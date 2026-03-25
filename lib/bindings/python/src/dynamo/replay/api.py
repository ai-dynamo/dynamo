# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay


def run_trace_replay(
    trace_file,
    *,
    extra_engine_args=None,
    router_config=None,
    num_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
):
    return _run_mocker_trace_replay(
        trace_file,
        extra_engine_args=extra_engine_args,
        router_config=router_config,
        num_workers=num_workers,
        replay_concurrency=replay_concurrency,
        replay_mode=replay_mode,
        router_mode=router_mode,
        arrival_speedup_ratio=arrival_speedup_ratio,
    )


def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    extra_engine_args=None,
    router_config=None,
    num_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
    arrival_interval_ms=1.0,
    turns_per_session=1,
    shared_prefix_ratio=0.0,
    num_prefix_groups=0,
    inter_turn_delay_ms=0.0,
):
    return _run_mocker_synthetic_trace_replay(
        input_tokens,
        output_tokens,
        request_count,
        extra_engine_args=extra_engine_args,
        router_config=router_config,
        num_workers=num_workers,
        replay_concurrency=replay_concurrency,
        replay_mode=replay_mode,
        router_mode=router_mode,
        arrival_speedup_ratio=arrival_speedup_ratio,
        arrival_interval_ms=arrival_interval_ms,
        turns_per_session=turns_per_session,
        shared_prefix_ratio=shared_prefix_ratio,
        num_prefix_groups=num_prefix_groups,
        inter_turn_delay_ms=inter_turn_delay_ms,
    )
