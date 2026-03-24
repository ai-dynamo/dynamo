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
    )
