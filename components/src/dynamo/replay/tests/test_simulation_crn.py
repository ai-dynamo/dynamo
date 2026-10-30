# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""Common random numbers in the Sweeper replay runner.

Trace replays (what a DGDR trace Run evaluates) are deterministic, so live
planner retune can compare candidates scored by different Runs of one
window; synthetic replays have no such option and must not receive it.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec

from dynamo.replay import ReplayReport
from dynamo.replay import api as replay_api
from dynamo.replay import simulation

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]

_REPORT = ReplayReport(
    summary={"completed_requests": 1, "output_throughput_tok_s": 1.0},
    per_request=None,
    coverage={},
    planner=None,
)


def _deployment(mode: str) -> BackendDeploymentSpec:
    if mode == "agg":
        return BackendDeploymentSpec(
            deployment_mode="agg",
            backend="vllm",
            backend_version="0.11.0",
            agg_engine_args={"timing_model": {"type": "polynomial"}},
            num_workers=2,
        )
    return BackendDeploymentSpec(
        deployment_mode="disagg",
        backend="vllm",
        backend_version="0.11.0",
        prefill_engine_args={
            "worker_type": "prefill",
            "timing_model": {"type": "polynomial"},
        },
        decode_engine_args={
            "worker_type": "decode",
            "timing_model": {"type": "polynomial"},
        },
        num_prefill_workers=1,
        num_decode_workers=2,
    )


def _strict_fake(real, seen: dict):
    """A replay entry point that binds its kwargs to ``real``'s signature."""

    signature = inspect.signature(real)

    def fake(*args, **kwargs):
        signature.bind(*args, **kwargs)
        seen.update(kwargs)
        return _REPORT

    return fake


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize(
    "workload",
    [
        {"trace_path": "window.jsonl.gz", "trace_format": "dynamo"},
        {"trace_paths": ["a.jsonl", "b.jsonl"], "trace_format": "mooncake"},
    ],
)
def test_trace_replays_are_deterministic(monkeypatch, mode, workload):
    seen: dict = {}
    monkeypatch.setattr(
        simulation,
        "run_trace_replay",
        _strict_fake(replay_api.run_trace_replay, seen),
    )
    spec = ReplaySpec(
        backend_deployment=_deployment(mode),
        workload=workload,
        goal={"target": "throughput"},
    )

    simulation.DynamoReplayRunnerFactory().create(0).run(spec)

    assert seen["_deterministic"] is True


@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_synthetic_replays_do_not_receive_the_flag(monkeypatch, mode):
    # The synthetic entry point has no such parameter; the strict fake fails
    # if the flag leaks into the shared kwargs.
    assert (
        "_deterministic"
        not in inspect.signature(replay_api.run_synthetic_trace_replay).parameters
    )
    seen: dict = {}
    monkeypatch.setattr(
        simulation,
        "run_synthetic_trace_replay",
        _strict_fake(replay_api.run_synthetic_trace_replay, seen),
    )
    spec = ReplaySpec(
        backend_deployment=_deployment(mode),
        workload={
            "trace_path": None,
            "isl": 512,
            "osl": 128,
            "num_request_ratio": 10.0,
            "concurrency": None,
            "request_rate": 20.0,
        },
        goal={"target": "throughput"},
    )

    simulation.DynamoReplayRunnerFactory().create(0).run(spec)

    assert seen  # the synthetic path ran
    assert "_deterministic" not in seen
