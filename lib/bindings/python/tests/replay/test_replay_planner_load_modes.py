# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner-bridge load modes: synthetic workloads and closed-loop concurrency.

Drives ``PlannerReplayBridge`` directly (advance_to -> finalize, no Python planner
adapter) to cover the constructors added so the planner path matches the non-planner
path: a concurrency cap on a trace, and synthetic open/closed-loop workloads.
"""

import pytest

from dynamo.mocker import MockEngineArgs, PlannerReplayBridge

from .replay_utils import _vllm_args, _write_trace_and_args

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]


def _drive_to_completion(bridge: PlannerReplayBridge) -> dict:
    """One large advance drains every event (arrival + concurrency alike)."""
    tick = bridge.advance_to(1.0e15)
    assert tick["is_done"], "replay should finish within the advance window"
    return bridge.finalize()


@pytest.mark.parametrize("replay_concurrency", [None, 2])
def test_planner_bridge_from_synthetic_agg(replay_concurrency):
    # Synthetic workload through the planner bridge: arrival (None) and closed-loop
    # (Some) both run every request to completion. Previously the planner bridge
    # only accepted a Mooncake trace file.
    bridge = PlannerReplayBridge.from_synthetic(
        input_tokens=64,
        output_tokens=16,
        request_count=8,
        extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
        num_workers=2,
        replay_concurrency=replay_concurrency,
        arrival_interval_ms=1.0,
    )
    report = _drive_to_completion(bridge)
    assert report["completed_requests"] == 8


def test_planner_bridge_from_synthetic_shared_prefix():
    # Prefix-cache sharing knobs apply to synthetic planner workloads too.
    bridge = PlannerReplayBridge.from_synthetic(
        input_tokens=128,
        output_tokens=8,
        request_count=8,
        extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
        num_workers=2,
        replay_concurrency=4,
        shared_prefix_ratio=0.5,
        num_prefix_groups=2,
    )
    report = _drive_to_completion(bridge)
    assert report["completed_requests"] == 8


def test_planner_bridge_trace_closed_loop(tmp_path):
    # A concurrency cap on a Mooncake trace file (closed-loop) through the planner
    # bridge — previously only arrival-timestamp replay was reachable here.
    trace_path = _write_trace_and_args(tmp_path)
    bridge = PlannerReplayBridge(
        trace_file=str(trace_path),
        extra_engine_args=_vllm_args(),
        num_workers=2,
        replay_concurrency=2,
    )
    report = _drive_to_completion(bridge)
    assert report["completed_requests"] > 0
