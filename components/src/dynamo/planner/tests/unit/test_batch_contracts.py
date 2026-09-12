# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the generic batch scheduling core contracts."""

from __future__ import annotations

import math

import pytest
from dynamo.planner.core.types import (
    BatchDispatcherFeedback,
    BatchDrainLimitDecision,
    BatchJobDemand,
    BatchSchedulingObservation,
    PlannerEffects,
    PoolTrafficDemand,
    TickInput,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_batch_contracts_are_compatibility_safe_and_not_shared():
    tick_input = TickInput(now_s=1_700_000_000.0)
    first = PlannerEffects()
    second = PlannerEffects()

    assert tick_input.batch is None
    assert first.batch_drain_limits == []
    assert second.batch_drain_limits == []

    first.batch_drain_limits.append(
        BatchDrainLimitDecision(
            pool_id="pool-a",
            max_admission_rps=0.0,
            valid_until_s=1_700_000_030.0,
            decision_id="decision-1",
        )
    )
    assert second.batch_drain_limits == []


def test_batch_job_preserves_raw_counters_and_derives_remaining_requests():
    job = BatchJobDemand(
        observed_at_s=1_700_000_000.0,
        pool_id="pool-a",
        job_id="batch-123",
        status="in_progress",
        total_requests=1_000,
        completed_requests=275,
        failed_requests=25,
        deadline_at_s=1_700_003_600.0,
        work_class="chat-8k",
    )

    assert job.remaining_requests == 700

    with pytest.raises(ValueError, match="must not exceed"):
        BatchJobDemand(
            observed_at_s=1_700_000_000.0,
            pool_id="pool-a",
            job_id="batch-invalid",
            status="in_progress",
            total_requests=10,
            completed_requests=9,
            failed_requests=2,
            deadline_at_s=None,
            work_class="chat-8k",
        )


def test_batch_observation_distinguishes_unreported_cap_from_pause():
    observation = BatchSchedulingObservation(
        pool_traffic=[
            PoolTrafficDemand(
                observed_at_s=1_700_000_001.0,
                pool_id="pool-a",
                online_offered_rps=90.0,
            )
        ],
        dispatcher_feedback=[
            BatchDispatcherFeedback(
                observed_at_s=1_700_000_002.0,
                pool_id="pool-a",
                observation_window_s=30.0,
                queued_requests=700,
                inflight_requests=20,
                actual_dispatch_rps=0.0,
                applied_max_admission_rps=None,
            ),
            BatchDispatcherFeedback(
                observed_at_s=1_700_000_003.0,
                pool_id="pool-b",
                observation_window_s=30.0,
                queued_requests=300,
                inflight_requests=0,
                actual_dispatch_rps=0.0,
                applied_max_admission_rps=0.0,
            ),
        ],
    )

    assert observation.dispatcher_feedback[0].applied_max_admission_rps is None
    assert observation.dispatcher_feedback[1].applied_max_admission_rps == 0.0


def test_batch_drain_limit_rejects_negative_rates_but_allows_pause():
    paused = BatchDrainLimitDecision(
        pool_id="pool-a",
        max_admission_rps=0.0,
        valid_until_s=1_700_000_030.0,
        decision_id="decision-pause",
    )
    assert paused.max_admission_rps == 0.0

    with pytest.raises(ValueError, match="non-negative"):
        BatchDrainLimitDecision(
            pool_id="pool-a",
            max_admission_rps=-1.0,
            valid_until_s=1_700_000_030.0,
            decision_id="decision-invalid",
        )


@pytest.mark.parametrize("invalid_rate", [math.nan, math.inf, -math.inf])
def test_batch_contracts_reject_nonfinite_rates(invalid_rate: float):
    with pytest.raises(ValueError, match="finite and non-negative"):
        PoolTrafficDemand(
            observed_at_s=1_700_000_000.0,
            pool_id="pool-a",
            online_offered_rps=invalid_rate,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        BatchDrainLimitDecision(
            pool_id="pool-a",
            max_admission_rps=invalid_rate,
            valid_until_s=1_700_000_030.0,
            decision_id="decision-invalid",
        )


@pytest.mark.parametrize("invalid_identifier", ["", None, 7])
def test_batch_contracts_require_string_identifiers(invalid_identifier: object):
    with pytest.raises(ValueError, match="non-empty string"):
        BatchDrainLimitDecision(
            pool_id=invalid_identifier,  # type: ignore[arg-type]
            max_admission_rps=1.0,
            valid_until_s=1_700_000_030.0,
            decision_id="decision-invalid",
        )


@pytest.mark.parametrize("invalid_count", [-1, 1.5, True])
def test_batch_contracts_require_integral_counts(invalid_count: object):
    with pytest.raises(ValueError, match="non-negative integers"):
        BatchDispatcherFeedback(
            observed_at_s=1_700_000_000.0,
            pool_id="pool-a",
            observation_window_s=30.0,
            queued_requests=invalid_count,  # type: ignore[arg-type]
            inflight_requests=0,
            actual_dispatch_rps=0.0,
        )
