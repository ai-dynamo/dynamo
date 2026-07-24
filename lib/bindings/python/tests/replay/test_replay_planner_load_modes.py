# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay load modes: synthetic workloads and closed-loop concurrency.

Exercises the single offline-replay entrypoints (``run_synthetic_trace_replay`` /
``run_trace_replay`` with no ``planner_config``) across the load modes that the
planner path previously needed bespoke bridge constructors for: synthetic
fixed and Poisson open-loop workloads, closed-loop (concurrency-capped)
workloads, synthetic prefix-cache sharing, and a concurrency cap on a Mooncake
trace file. With the unified event-driven path these all run through the same
multi-worker runtime, so a bare run drives every request to completion.
"""

import pytest

from dynamo.mocker import MockEngineArgs, PlannerReplayBridge
from dynamo.replay import run_synthetic_trace_replay, run_trace_replay

from .replay_utils import _vllm_args, _write_trace_and_args

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]


@pytest.mark.parametrize(
    "load_controller",
    [
        pytest.param({"arrival_interval_ms": 1.0}, id="fixed-open-loop"),
        pytest.param(
            {"request_rate": 1000.0, "arrival_seed": 17},
            id="poisson-open-loop",
        ),
        pytest.param({"replay_concurrency": 2}, id="closed-loop"),
    ],
)
def test_synthetic_agg_load_modes(load_controller):
    report = run_synthetic_trace_replay(
        64,
        16,
        8,
        extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
        num_workers=2,
        replay_mode="offline",
        **load_controller,
    )
    assert {
        key: report[key]
        for key in (
            "num_requests",
            "completed_requests",
            "total_input_tokens",
            "total_output_tokens",
        )
    } == {
        "num_requests": 8,
        "completed_requests": 8,
        "total_input_tokens": 512,
        "total_output_tokens": 128,
    }


def test_synthetic_shared_prefix_closed_loop():
    # Prefix-cache sharing knobs apply to synthetic closed-loop workloads too.
    report = run_synthetic_trace_replay(
        128,
        8,
        8,
        extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
        num_workers=2,
        replay_concurrency=4,
        replay_mode="offline",
        shared_prefix_ratio=0.5,
        num_prefix_groups=2,
    )
    assert report["completed_requests"] == 8


def test_trace_closed_loop(tmp_path):
    # A concurrency cap on a Mooncake trace file (closed-loop), driven to
    # completion through the unified multi-worker offline path.
    trace_path = _write_trace_and_args(tmp_path)
    report = run_trace_replay(
        str(trace_path),
        extra_engine_args=_vllm_args(),
        num_workers=2,
        replay_concurrency=2,
        replay_mode="offline",
    )
    assert report["completed_requests"] == 2


def test_planner_callback_error_preserves_python_exception_type():
    # A raising planner callback must propagate its original Python exception
    # type (here ValueError) out of bridge.run() — not a generic Exception — so
    # callback failures stay diagnosable (type + traceback preserved across the
    # Rust seam).
    class _RaisingPlanner:
        def initial_tick_ms(self):
            return 0.0  # Run the callback before the replay can finish.

        def on_tick(self, metrics):
            raise ValueError("boom from on_tick")

    bridge = PlannerReplayBridge.from_synthetic(
        input_tokens=64,
        output_tokens=16,
        request_count=8,
        extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
        num_workers=1,
        replay_concurrency=2,
    )
    with pytest.raises(ValueError, match="boom from on_tick"):
        bridge.run(_RaisingPlanner())
