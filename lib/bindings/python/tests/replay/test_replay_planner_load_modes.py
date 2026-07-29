# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay load modes: synthetic workloads and closed-loop concurrency.

Exercises the single offline-replay entrypoints (``run_synthetic_trace_replay`` /
``run_trace_replay`` with no ``planner_config``) across the load modes that the
planner path previously needed bespoke planner constructors for: synthetic
fixed and Poisson open-loop workloads, closed-loop (concurrency-capped)
workloads, synthetic prefix-cache sharing, and a concurrency cap on a Mooncake
trace file. With the unified event-driven path these all run through the same
multi-worker runtime, so a bare run drives every request to completion.
"""

import copy
import threading

import pytest

from dynamo._core import run_mocker_synthetic_trace_replay
from dynamo.mocker import MockEngineArgs
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
    # type (here ValueError) out of replay — not a generic Exception — so
    # callback failures stay diagnosable (type + traceback preserved across the
    # Rust seam).
    class _RaisingPlanner:
        def initial_tick_ms(self):
            return 0.0  # Run the callback before the replay can finish.

        def on_tick(self, metrics):
            raise ValueError("boom from on_tick")

    with pytest.raises(ValueError, match="boom from on_tick"):
        run_mocker_synthetic_trace_replay(
            64,
            16,
            8,
            extra_engine_args=MockEngineArgs(
                block_size=64,
                speedup_ratio=1000.0,
            ),
            num_workers=1,
            replay_concurrency=2,
            scaling_policy=_RaisingPlanner(),
        )


class _DisabledScalingPolicy:
    def initial_tick_ms(self):
        return float("inf")

    def on_tick(self, metrics):
        raise AssertionError("a disabled scaling policy must not receive ticks")


def _canonical_report(report):
    report = copy.deepcopy(report)
    for key in (
        "wall_time_ms",
        "processed_tokens_per_s",
        "processed_output_tokens_per_s",
    ):
        report.pop(key, None)
    return report


@pytest.mark.parametrize("disagg", [False, True])
def test_disabled_policy_is_semantically_identical_to_normal_replay(disagg):
    kwargs = {
        "input_tokens": 64,
        "output_tokens": 16,
        "request_count": 16,
        "replay_concurrency": 4,
    }
    if disagg:
        kwargs.update(
            prefill_engine_args=MockEngineArgs(
                block_size=64,
                speedup_ratio=1000.0,
                worker_type="prefill",
            ),
            decode_engine_args=MockEngineArgs(
                block_size=64,
                speedup_ratio=1000.0,
                worker_type="decode",
            ),
        )
    else:
        kwargs["extra_engine_args"] = MockEngineArgs(
            block_size=64,
            speedup_ratio=1000.0,
        )

    normal = run_mocker_synthetic_trace_replay(**kwargs)
    with_policy = run_mocker_synthetic_trace_replay(
        **kwargs,
        scaling_policy=_DisabledScalingPolicy(),
    )

    normal_report = _canonical_report(normal)
    policy_report = _canonical_report(with_policy)
    for key in ("prefill_worker_seconds", "decode_worker_seconds", "gpu_hours"):
        assert normal_report.pop(key) == pytest.approx(
            policy_report.pop(key), rel=1e-12, abs=1e-15
        )
    assert normal_report == policy_report


def test_scaling_policy_rejects_online_replay_before_dispatch():
    with pytest.raises(
        ValueError, match="scaling_policy only supports replay_mode='offline'"
    ):
        run_mocker_synthetic_trace_replay(
            64,
            16,
            8,
            extra_engine_args=MockEngineArgs(block_size=64),
            replay_mode="online",
            scaling_policy=_DisabledScalingPolicy(),
        )


def test_normal_replay_releases_gil_for_background_python_thread():
    started = threading.Event()
    stop = threading.Event()
    progress = 0

    def run_background():
        nonlocal progress
        started.set()
        while not stop.is_set():
            progress += 1

    thread = threading.Thread(target=run_background)
    thread.start()
    assert started.wait(timeout=1.0)
    progress_before_replay = progress

    try:
        run_mocker_synthetic_trace_replay(
            64,
            16,
            50_000,
            extra_engine_args=MockEngineArgs(block_size=64, speedup_ratio=1000.0),
            num_workers=2,
            replay_concurrency=16,
            scaling_policy=None,
        )
        progress_during_replay = progress - progress_before_replay
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert progress_during_replay > 0
    assert not thread.is_alive()
