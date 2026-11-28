# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from feedback_fixtures import SLO, sample

pytest.importorskip("dynamo.planner.core.types")
from autoscaling_arena.adapters.outcome_feedback import (  # noqa: E402
    LatencyCalibration,
    OutcomeFeedback,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("short_policy,expected_good", [("skip", 7), ("fail", 6)])
def test_feedback_uses_actual_completed_slo_counts_and_all_gpu_time(
    short_policy, expected_good
):
    feedback = OutcomeFeedback(slo=SLO, short_output_itl=short_policy)
    feedback.observe(sample())
    assert feedback.view(0) is None
    feedback.observe(
        sample(
            5,
            terminal=10,
            completed=9,
            good=7,
            strict_good=6,
            gpu_seconds=100,
            draining_d=1,
        )
    )
    result = feedback.view(5)
    assert result["good_rate"] == expected_good / 10
    assert result["goodput_per_gpu"] == expected_good / 100
    assert result["average_gpus"] == 20
    assert result["mean_ttft_ms"] == 80
    assert result["mean_itl_ms"] == 12
    assert result["basis"] == "past_completion_window"
    # A duplicate final event cannot double-count any requests or GPU time.
    final = copy.deepcopy(feedback.latest)
    final["kind"] = "final"
    feedback.observe(final)
    assert feedback.view(5) == result


def test_empty_completions_are_unknown_and_old_intervals_expire():
    feedback = OutcomeFeedback(slo=SLO, window_s=5)
    feedback.observe(sample())
    feedback.observe(sample(5, terminal=10, completed=10, good=7, gpu_seconds=60))
    feedback.observe(sample(10, terminal=10, completed=10, good=7, gpu_seconds=120))
    result = feedback.view(10)
    assert result["good_rate"] is None
    assert result["terminal_requests"] == 0
    assert result["gpu_seconds"] == 60


@pytest.mark.parametrize(
    "fault", ["version", "slo", "negative", "impossible", "overlap", "unbounded"]
)
def test_invalid_feedback_fails_before_entering_a_policy(fault):
    observation = sample(5, terminal=2, completed=2, good=1, gpu_seconds=60)
    data = observation["completion_feedback"]
    if fault == "version":
        data["schema_version"] = 0
    if fault == "slo":
        data["sla"] = {**SLO, "ttft_ms": 1}
    if fault == "negative":
        data["gpu_seconds"] = -1
    if fault == "impossible":
        data["good_requests"] = 3
    if fault == "overlap":
        observation["draining_decode_ids"] = [0]
    if fault == "unbounded":
        data.update(latency_samples=[(1, 1)] * 65, latency_population=65)
    with pytest.raises(ValueError):
        OutcomeFeedback(slo=SLO).observe(observation)


def test_feedback_rejects_time_travel_stale_state_and_counter_reset():
    feedback = OutcomeFeedback(slo=SLO)
    feedback.observe(sample(5, terminal=2, completed=2, good=1, gpu_seconds=60))
    with pytest.raises(ValueError, match="future"):
        feedback.view(4)
    with pytest.raises(ValueError, match="stale"):
        feedback.view(16)
    with pytest.raises(ValueError, match="advance"):
        feedback.observe(sample(4))
    with pytest.raises(ValueError, match="backwards"):
        feedback.observe(sample(10))


def test_calibration_measures_joint_success_not_success_of_mean_latencies():
    calibration = LatencyCalibration(100, 20)
    # Means (90 ms, 15 ms) pass both bounds, but each request violates one.
    observation = sample(5, pairs=[(150, 5), (30, 25)] * 16)
    calibration.observe(observation, ((1, 1), 90, 15, 0), (1, 1))
    assert calibration.fraction(90, 15, 0.99) == 0
    assert calibration.fraction(45, 7.5, 0.99) == 1
    calibration.observe(observation, ((1, 1), 90, 15, 0), (1, 1))
    assert len(calibration.residuals) == 32


@pytest.mark.parametrize("short_policy,expected", [("skip", 1), ("fail", 0)])
def test_calibration_respects_short_outputs(short_policy, expected):
    calibration = LatencyCalibration(100, 20, short_output_itl=short_policy)
    calibration.observe(sample(5, pairs=[(50, None)] * 32), ((1, 1), 50, 10, 0), (1, 1))
    assert calibration.fraction(50, 10, 0.99) == expected


@pytest.mark.parametrize(
    "fault", ["fleet", "drain", "start", "prediction_after_interval", "actual_fleet"]
)
def test_calibration_ignores_intervals_that_do_not_match_the_prediction(fault):
    calibration = LatencyCalibration(100, 20)
    observation = sample(5, pairs=[(50, 10)])
    prediction = (
        (2, 1) if fault == "fleet" else (1, 1),
        50,
        10,
        1 if fault == "prediction_after_interval" else 0,
    )
    if fault == "drain":
        observation["draining_decode_ids"] = [1]
    if fault == "start":
        observation["starting_decode_ids"] = [1]
    if fault == "actual_fleet":
        observation["active_decode_ids"] = [0, 1]
    calibration.observe(observation, prediction, (1, 1))
    assert not calibration.residuals
