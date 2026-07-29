# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for the public multi-deployment Replay configuration."""

import asyncio
from dataclasses import replace

import pytest

import dynamo.replay.world as world_module
from dynamo.replay.world import (
    ReplayDeploymentConfig,
    ReplayGlobalPlannerConfig,
    ReplaySyntheticWorkload,
    ReplayTraceWorkload,
    _resolved_synthetic_arrival_interval_ms,
    _validate_deployment_engine_args,
    run_replay_world,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _deployment(**overrides) -> ReplayDeploymentConfig:
    deployment = ReplayDeploymentConfig(
        deployment_id="deployment",
        planner_config={},
        workload=ReplaySyntheticWorkload(
            input_tokens=128,
            output_tokens=32,
            request_count=10,
        ),
    )
    return replace(deployment, **overrides)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_total_gpus", True),
        ("max_total_gpus", 2.0),
        ("max_total_gpus", float("nan")),
        ("min_total_gpus", False),
        ("min_total_gpus", float("inf")),
    ],
)
def test_global_gpu_limits_require_integers(field, value):
    with pytest.raises(ValueError, match=rf"{field} must be an integer"):
        ReplayGlobalPlannerConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1, True, 512.0, float("nan")])
def test_trace_block_size_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="trace_block_size"):
        ReplayTraceWorkload("trace.jsonl", trace_block_size=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_tokens", True),
        ("output_tokens", 32.0),
        ("request_count", float("inf")),
        ("turns_per_session", 0),
        ("num_prefix_groups", -1),
        ("num_prefix_groups", False),
        ("arrival_seed", -1),
        ("arrival_seed", True),
        ("arrival_seed", 1.0),
    ],
)
def test_synthetic_workload_count_fields_require_integers(field, value):
    workload = ReplaySyntheticWorkload(
        input_tokens=128,
        output_tokens=32,
        request_count=10,
    )
    with pytest.raises(ValueError, match=field):
        replace(workload, **{field: value})


def test_synthetic_arrival_seed_must_fit_u64():
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        ReplaySyntheticWorkload(
            input_tokens=128,
            output_tokens=32,
            request_count=10,
            arrival_seed=1 << 64,
        )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_synthetic_request_rate_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="request_rate"):
        ReplaySyntheticWorkload(
            input_tokens=128,
            output_tokens=32,
            request_count=10,
            request_rate=value,
        )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
def test_synthetic_arrival_interval_must_be_finite_and_non_negative(value):
    with pytest.raises(ValueError, match="arrival_interval_ms"):
        ReplaySyntheticWorkload(
            input_tokens=128,
            output_tokens=32,
            request_count=10,
            arrival_interval_ms=value,
        )


def test_synthetic_open_loop_controls_are_mutually_exclusive():
    with pytest.raises(ValueError, match="at most one"):
        ReplaySyntheticWorkload(
            input_tokens=128,
            output_tokens=32,
            request_count=10,
            request_rate=5.0,
            arrival_interval_ms=1.0,
        )


@pytest.mark.parametrize(
    "workload",
    [
        ReplaySyntheticWorkload(
            input_tokens=128,
            output_tokens=32,
            request_count=10,
            request_rate=5.0,
        ),
        ReplaySyntheticWorkload(
            input_tokens=128,
            output_tokens=32,
            request_count=10,
            arrival_interval_ms=1.0,
        ),
    ],
)
def test_synthetic_closed_loop_rejects_an_open_loop_controller(workload):
    with pytest.raises(ValueError, match="at most one"):
        _deployment(workload=workload, replay_concurrency=2)


def test_synthetic_arrival_interval_resolution_preserves_legacy_default():
    workload = ReplaySyntheticWorkload(
        input_tokens=128,
        output_tokens=32,
        request_count=10,
    )

    assert _resolved_synthetic_arrival_interval_ms(workload, None) == 1.0
    assert _resolved_synthetic_arrival_interval_ms(workload, 2) is None
    assert (
        _resolved_synthetic_arrival_interval_ms(
            replace(workload, request_rate=5.0), None
        )
        is None
    )
    assert (
        _resolved_synthetic_arrival_interval_ms(
            replace(workload, arrival_interval_ms=0.0), None
        )
        == 0.0
    )


def test_trace_workload_accepts_replay_concurrency():
    deployment = ReplayDeploymentConfig(
        deployment_id="trace",
        planner_config={},
        workload=ReplayTraceWorkload("trace.jsonl"),
        replay_concurrency=2,
    )

    assert deployment.replay_concurrency == 2


@pytest.mark.parametrize(
    (
        "workload",
        "replay_concurrency",
        "expected_request_rate",
        "expected_arrival_interval_ms",
        "expected_arrival_seed",
    ),
    [
        pytest.param(
            ReplaySyntheticWorkload(128, 32, 10),
            None,
            None,
            1.0,
            42,
            id="legacy-fixed-interval",
        ),
        pytest.param(
            ReplaySyntheticWorkload(
                128,
                32,
                10,
                request_rate=7.5,
                arrival_seed=99,
            ),
            None,
            7.5,
            None,
            99,
            id="poisson-rate-and-seed",
        ),
        pytest.param(
            ReplaySyntheticWorkload(128, 32, 10, arrival_interval_ms=17.5),
            None,
            None,
            17.5,
            42,
            id="explicit-fixed-interval",
        ),
        pytest.param(
            ReplaySyntheticWorkload(128, 32, 10),
            3,
            None,
            None,
            42,
            id="closed-loop-concurrency",
        ),
    ],
)
def test_world_forwards_synthetic_load_controller_to_planner_bridge(
    monkeypatch: pytest.MonkeyPatch,
    workload: ReplaySyntheticWorkload,
    replay_concurrency: int | None,
    expected_request_rate: float | None,
    expected_arrival_interval_ms: float | None,
    expected_arrival_seed: int,
):
    pytest.importorskip("dynamo._core")
    import dynamo.replay.main as replay_main

    captured: dict = {}

    class _CapturedBridgeArguments(Exception):
        pass

    def capture_bridge_arguments(**kwargs):
        captured.update(kwargs)
        raise _CapturedBridgeArguments

    monkeypatch.setattr(
        replay_main,
        "_build_planner_replay",
        capture_bridge_arguments,
    )

    deployment = _deployment(
        planner_config={"mode": "agg"},
        workload=workload,
        replay_concurrency=replay_concurrency,
    )
    with pytest.raises(_CapturedBridgeArguments):
        run_replay_world([deployment])

    synthetic = captured["synthetic"]
    assert (
        synthetic.request_rate,
        synthetic.arrival_interval_ms,
        synthetic.arrival_seed,
        captured["replay_concurrency"],
    ) == (
        expected_request_rate,
        expected_arrival_interval_ms,
        expected_arrival_seed,
        replay_concurrency,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_workers", -1),
        ("num_prefill_workers", True),
        ("num_decode_workers", 1.0),
        ("replay_concurrency", 0),
        ("replay_concurrency", False),
        ("benchmark_granularity", 0),
        ("benchmark_granularity", -1),
        ("benchmark_granularity", True),
        ("benchmark_granularity", 8.0),
        ("benchmark_granularity", float("nan")),
    ],
)
def test_deployment_count_fields_require_valid_integers(field, value):
    with pytest.raises(ValueError, match=field):
        _deployment(**{field: value})


@pytest.mark.parametrize("deployment_id", ["", "   ", "\t\n"])
def test_deployment_id_rejects_empty_or_whitespace_only_values(deployment_id):
    with pytest.raises(ValueError, match="deployment_id must be non-empty"):
        _deployment(deployment_id=deployment_id)


@pytest.mark.parametrize("deployment_id", ["model/a?", "部署-A"])
def test_deployment_id_accepts_punctuation_and_unicode(deployment_id):
    assert _deployment(deployment_id=deployment_id).deployment_id == deployment_id


def test_aggregated_mode_rejects_role_specific_engine_args():
    for field in ("prefill_engine_args", "decode_engine_args"):
        deployment = _deployment(**{field: object()})
        with pytest.raises(ValueError, match="accepts only extra_engine_args"):
            _validate_deployment_engine_args(deployment, "agg")


def test_disaggregated_mode_rejects_aggregate_engine_args():
    deployment = _deployment(
        extra_engine_args=object(),
        prefill_engine_args=object(),
        decode_engine_args=object(),
    )

    with pytest.raises(ValueError, match="does not accept extra_engine_args"):
        _validate_deployment_engine_args(deployment, "disagg")


@pytest.mark.parametrize(
    ("prefill_engine_args", "decode_engine_args"),
    [(None, object()), (object(), None), (None, None)],
)
def test_disaggregated_mode_requires_both_role_engine_args(
    prefill_engine_args,
    decode_engine_args,
):
    deployment = _deployment(
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
    )

    with pytest.raises(
        ValueError, match="requires prefill_engine_args and decode_engine_args"
    ):
        _validate_deployment_engine_args(deployment, "disagg")


def test_valid_engine_argument_shapes_are_accepted():
    _validate_deployment_engine_args(_deployment(), "agg")
    _validate_deployment_engine_args(
        _deployment(
            prefill_engine_args=object(),
            decode_engine_args=object(),
        ),
        "disagg",
    )


def test_synchronous_world_api_rejects_a_running_event_loop():
    async def invoke() -> None:
        with pytest.raises(RuntimeError, match="active asyncio event loop"):
            run_replay_world([_deployment()])

    asyncio.run(invoke())


def test_setup_failure_closes_the_world_event_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("dynamo._core")
    loop = asyncio.new_event_loop()

    def fail_clock():
        raise RuntimeError("clock construction failed")

    monkeypatch.setattr(asyncio, "new_event_loop", lambda: loop)
    monkeypatch.setattr(world_module, "VirtualClock", fail_clock)

    with pytest.raises(RuntimeError, match="clock construction failed"):
        run_replay_world([_deployment()])

    assert loop.is_closed()
