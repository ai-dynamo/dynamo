# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic tests for the pure single-pool batch scheduling policy."""

from __future__ import annotations

from dataclasses import replace

import pytest
from dynamo.planner.core.batch_policy import (
    BatchSchedulingPolicyConfig,
    plan_batch_schedule,
)
from dynamo.planner.core.types import (
    BatchDispatcherFeedback,
    BatchJobDemand,
    BatchSchedulingObservation,
    PoolTrafficDemand,
    TickInput,
    WorkerCounts,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

NOW_S = 1_700_000_000.0


def _config(**overrides: object) -> BatchSchedulingPolicyConfig:
    config = BatchSchedulingPolicyConfig(
        pool_id="pool-a",
        work_class="chat-8k",
        safe_rps_per_ready_replica=10.0,
        cold_start_margin_s=0.0,
        finalization_margin_s=0.0,
        max_observation_age_s=30.0,
        drain_lease_duration_s=15.0,
        min_replicas=1,
        max_replicas=20,
        scale_from_zero_replicas=1,
    )
    return replace(config, **overrides)


def _job(
    job_id: str,
    *,
    remaining_requests: int,
    deadline_offset_s: float | None,
    observed_offset_s: float = 0.0,
    work_class: str = "chat-8k",
    status: str = "in_progress",
) -> BatchJobDemand:
    total_requests = max(1_000, remaining_requests)
    completed_requests = total_requests - remaining_requests
    return BatchJobDemand(
        observed_at_s=NOW_S + observed_offset_s,
        pool_id="pool-a",
        job_id=job_id,
        status=status,
        total_requests=total_requests,
        completed_requests=completed_requests,
        failed_requests=0,
        deadline_at_s=(
            NOW_S + deadline_offset_s if deadline_offset_s is not None else None
        ),
        work_class=work_class,
    )


def _tick_input(
    *,
    ready_replicas: int | None = 10,
    online_offered_rps: float | None = 90.0,
    online_observed_offset_s: float = 0.0,
    jobs: list[BatchJobDemand] | None = None,
    dispatcher_feedback: list[BatchDispatcherFeedback] | None = None,
) -> TickInput:
    worker_counts = (
        WorkerCounts(ready_num_decode=ready_replicas)
        if ready_replicas is not None
        else None
    )
    return TickInput(
        now_s=NOW_S,
        worker_counts=worker_counts,
        batch=BatchSchedulingObservation(
            job_demands=jobs or [],
            pool_traffic=(
                [
                    PoolTrafficDemand(
                        observed_at_s=NOW_S + online_observed_offset_s,
                        pool_id="pool-a",
                        online_offered_rps=online_offered_rps,
                    )
                ]
                if online_offered_rps is not None
                else []
            ),
            dispatcher_feedback=dispatcher_feedback or [],
        ),
    )


def _plan(
    tick_input: TickInput,
    config: BatchSchedulingPolicyConfig | None = None,
):
    return plan_batch_schedule(
        tick_input,
        config or _config(),
        decision_id="decision-123",
    )


def test_work_conserving_headroom_drives_leased_drain_cap():
    plan = _plan(
        _tick_input(
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=600,
                    deadline_offset_s=100.0,
                )
            ]
        )
    )

    assert plan.replica_floor == 10
    assert plan.drain_limit.pool_id == "pool-a"
    assert plan.drain_limit.max_admission_rps == 10.0
    assert plan.drain_limit.valid_until_s == NOW_S + 15.0
    assert plan.drain_limit.decision_id == "decision-123"
    assert plan.diagnostics.required_batch_rps == 6.0
    assert plan.diagnostics.current_safe_headroom_rps == 10.0
    assert plan.diagnostics.predicted_finish_at_s == NOW_S + 60.0
    assert plan.diagnostics.minimum_deadline_slack_s == 40.0
    assert plan.diagnostics.infeasible is False


def test_optional_batch_cap_limits_drain_and_predicted_finish():
    plan = _plan(
        _tick_input(
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=600,
                    deadline_offset_s=100.0,
                )
            ]
        ),
        _config(max_batch_admission_rps=8.0),
    )

    assert plan.drain_limit.max_admission_rps == 8.0
    assert plan.diagnostics.planned_batch_capacity_rps == 8.0
    assert plan.diagnostics.predicted_finish_at_s == NOW_S + 75.0
    assert plan.diagnostics.minimum_deadline_slack_s == 25.0
    assert plan.diagnostics.infeasible is False


def test_edf_uses_cumulative_work_across_multiple_deadlines():
    plan = _plan(
        _tick_input(
            ready_replicas=5,
            online_offered_rps=20.0,
            jobs=[
                _job("job-a", remaining_requests=100, deadline_offset_s=10.0),
                _job("job-b", remaining_requests=300, deadline_offset_s=20.0),
            ],
        )
    )

    # Individual rates would be 10 and 15 RPS. EDF cumulative demand at the
    # second deadline is (100 + 300) / 20 = 20 RPS, which is the binding bound.
    assert plan.diagnostics.required_batch_rps == 20.0
    assert plan.diagnostics.required_replica_floor == 4
    # Current headroom permits a 30 RPS drain, so the coupled floor remains at
    # five replicas instead of allowing an unsafe simultaneous scale-down.
    assert plan.replica_floor == 5
    assert plan.drain_limit.max_admission_rps == 30.0
    assert plan.diagnostics.current_safe_headroom_rps == 30.0
    assert plan.diagnostics.planned_batch_capacity_rps == 30.0
    assert plan.diagnostics.predicted_finish_at_s == pytest.approx(NOW_S + 400.0 / 30.0)
    assert plan.diagnostics.minimum_deadline_slack_s == pytest.approx(20.0 / 3.0)
    assert plan.diagnostics.infeasible is False


def test_deadline_required_rps_scales_the_absolute_replica_floor():
    plan = _plan(
        _tick_input(
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=2_000,
                    deadline_offset_s=100.0,
                )
            ]
        )
    )

    assert plan.diagnostics.required_batch_rps == 20.0
    assert plan.diagnostics.required_replica_floor == 11
    assert plan.replica_floor == 11
    # Drain remains bounded by *current* safe headroom until replica 11 is ready.
    assert plan.drain_limit.max_admission_rps == 10.0
    assert plan.diagnostics.planned_batch_capacity_rps == 20.0


def test_cold_start_and_finalization_margins_reduce_deadline_window():
    plan = _plan(
        _tick_input(
            ready_replicas=1,
            online_offered_rps=0.0,
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=800,
                    deadline_offset_s=100.0,
                )
            ],
        ),
        _config(cold_start_margin_s=10.0, finalization_margin_s=10.0),
    )

    assert plan.diagnostics.required_batch_rps == 10.0
    assert plan.replica_floor == 1
    assert plan.diagnostics.predicted_finish_at_s == NOW_S + 90.0
    assert plan.diagnostics.minimum_deadline_slack_s == 10.0


def test_replica_ceiling_surfaces_infeasible_deadline_but_drains_safely():
    plan = _plan(
        _tick_input(
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=2_000,
                    deadline_offset_s=100.0,
                )
            ]
        ),
        _config(max_replicas=10),
    )

    assert plan.replica_floor == 10
    assert plan.drain_limit.max_admission_rps == 10.0
    assert plan.diagnostics.infeasible is True
    assert "required_replica_floor_exceeds_max_replicas" in (
        plan.diagnostics.infeasible_reasons
    )
    assert "negative_deadline_slack" in plan.diagnostics.infeasible_reasons
    assert plan.diagnostics.minimum_deadline_slack_s == -100.0


def test_batch_cap_below_required_rate_is_diagnosed_infeasible():
    plan = _plan(
        _tick_input(
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=1_000,
                    deadline_offset_s=100.0,
                )
            ]
        ),
        _config(max_batch_admission_rps=5.0),
    )

    assert plan.drain_limit.max_admission_rps == 5.0
    assert plan.diagnostics.infeasible is True
    assert "batch_cap_below_required_rps" in plan.diagnostics.infeasible_reasons


def test_zero_batch_cap_is_an_explicit_pause_with_fresh_inputs():
    plan = _plan(
        _tick_input(
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=100,
                    deadline_offset_s=None,
                )
            ]
        ),
        _config(max_batch_admission_rps=0.0),
    )

    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.required_batch_rps == 0.0
    assert plan.diagnostics.safety_paused is False


def test_exhausted_deadline_window_uses_max_floor_and_reports_infinite_rate():
    plan = _plan(
        _tick_input(
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=100,
                    deadline_offset_s=10.0,
                )
            ]
        ),
        _config(cold_start_margin_s=5.0, finalization_margin_s=5.0),
    )

    assert plan.replica_floor == 20
    assert plan.diagnostics.required_batch_rps == float("inf")
    assert "deadline_window_exhausted:job-a" in (plan.diagnostics.infeasible_reasons)


def test_stale_online_traffic_pauses_drain_and_holds_ready_floor():
    plan = _plan(
        _tick_input(
            ready_replicas=7,
            online_observed_offset_s=-31.0,
            jobs=[_job("job-a", remaining_requests=100, deadline_offset_s=100.0)],
        )
    )

    assert plan.replica_floor == 7
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.safety_paused is True
    assert plan.diagnostics.required_batch_rps is None
    assert plan.diagnostics.online_observation_stale is True
    assert "online_traffic_observation_stale_or_invalid" in (
        plan.diagnostics.infeasible_reasons
    )


def test_missing_capacity_pauses_drain_and_uses_max_floor():
    plan = _plan(
        _tick_input(
            ready_replicas=None,
            jobs=[_job("job-a", remaining_requests=100, deadline_offset_s=100.0)],
        )
    )

    assert plan.replica_floor == 20
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.capacity_observation_stale is True
    assert "capacity_observation_missing_or_invalid" in (
        plan.diagnostics.infeasible_reasons
    )


def test_missing_batch_snapshot_pauses_and_holds_ready_floor():
    plan = _plan(
        TickInput(
            now_s=NOW_S,
            worker_counts=WorkerCounts(ready_num_decode=7),
            batch=None,
        )
    )

    assert plan.replica_floor == 7
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.required_batch_rps is None
    assert "batch_observation_missing" in plan.diagnostics.infeasible_reasons


def test_fresh_active_job_bootstraps_zero_pool_while_missing_traffic_pauses_drain():
    plan = _plan(
        _tick_input(
            ready_replicas=0,
            online_offered_rps=None,
            jobs=[_job("job-a", remaining_requests=100, deadline_offset_s=None)],
        ),
        _config(min_replicas=0, max_replicas=4),
    )

    assert plan.replica_floor == 1
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.safety_paused is True
    assert plan.diagnostics.required_replica_floor == 1
    assert plan.diagnostics.active_job_count == 1
    assert plan.diagnostics.remaining_requests == 100
    assert "online_traffic_observation_missing" in (plan.diagnostics.infeasible_reasons)


def test_explicit_zero_traffic_is_fresh_and_still_bootstraps_zero_pool():
    plan = _plan(
        _tick_input(
            ready_replicas=0,
            online_offered_rps=0.0,
            jobs=[_job("job-a", remaining_requests=100, deadline_offset_s=None)],
        ),
        _config(min_replicas=0, max_replicas=4),
    )

    assert plan.replica_floor == 1
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.safety_paused is False
    assert plan.diagnostics.required_batch_rps == 0.0
    assert plan.diagnostics.required_replica_floor == 1
    assert plan.diagnostics.online_offered_rps == 0.0


def test_missing_traffic_without_active_jobs_does_not_bootstrap_zero_pool():
    plan = _plan(
        _tick_input(
            ready_replicas=0,
            online_offered_rps=None,
            jobs=[],
        ),
        _config(min_replicas=0, max_replicas=4),
    )

    assert plan.replica_floor == 0
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.required_replica_floor is None


def test_stale_active_job_does_not_bootstrap_zero_pool():
    plan = _plan(
        _tick_input(
            ready_replicas=0,
            online_offered_rps=None,
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=100,
                    deadline_offset_s=None,
                    observed_offset_s=-31.0,
                )
            ],
        ),
        _config(min_replicas=0, max_replicas=4),
    )

    assert plan.replica_floor == 0
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.required_replica_floor is None
    assert plan.diagnostics.job_observation_stale is True


def test_missing_traffic_holds_nonzero_ready_pool_without_extra_bootstrap():
    plan = _plan(
        _tick_input(
            ready_replicas=2,
            online_offered_rps=None,
            jobs=[_job("job-a", remaining_requests=100, deadline_offset_s=None)],
        ),
        _config(min_replicas=0, max_replicas=4),
    )

    assert plan.replica_floor == 2
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.required_replica_floor is None


def test_stale_active_job_pauses_drain_and_holds_ready_floor():
    plan = _plan(
        _tick_input(
            ready_replicas=7,
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=100,
                    deadline_offset_s=100.0,
                    observed_offset_s=-31.0,
                )
            ],
        )
    )

    assert plan.replica_floor == 7
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.job_observation_stale is True


def test_work_class_mismatch_pauses_and_holds_clipped_ready_floor():
    plan = _plan(
        _tick_input(
            ready_replicas=25,
            jobs=[
                _job(
                    "job-a",
                    remaining_requests=100,
                    deadline_offset_s=100.0,
                    work_class="embeddings",
                )
            ],
        )
    )

    assert plan.replica_floor == 20
    assert plan.drain_limit.max_admission_rps == 0.0
    assert plan.diagnostics.infeasible is True
    assert "work_class_mismatch:job-a:embeddings" in (
        plan.diagnostics.infeasible_reasons
    )


def test_latest_raw_job_counters_win_and_terminal_jobs_do_not_add_demand():
    old = _job(
        "job-a",
        remaining_requests=900,
        deadline_offset_s=100.0,
        observed_offset_s=-10.0,
    )
    latest = _job(
        "job-a",
        remaining_requests=300,
        deadline_offset_s=100.0,
    )
    terminal = _job(
        "job-b",
        remaining_requests=500,
        deadline_offset_s=50.0,
        status="completed",
    )

    plan = _plan(_tick_input(jobs=[old, latest, terminal]))

    assert plan.diagnostics.active_job_count == 1
    assert plan.diagnostics.remaining_requests == 300
    assert plan.diagnostics.required_batch_rps == 3.0


def test_dispatcher_feedback_is_diagnostic_and_freshness_is_explicit():
    feedback = BatchDispatcherFeedback(
        observed_at_s=NOW_S,
        pool_id="pool-a",
        observation_window_s=10.0,
        queued_requests=100,
        inflight_requests=5,
        actual_dispatch_rps=7.5,
        applied_max_admission_rps=8.0,
    )
    plan = _plan(
        _tick_input(
            jobs=[_job("job-a", remaining_requests=100, deadline_offset_s=None)],
            dispatcher_feedback=[feedback],
        )
    )

    assert plan.diagnostics.dispatcher_feedback_stale is False
    assert plan.diagnostics.dispatcher_observation_age_s == 0.0
    assert plan.diagnostics.actual_dispatch_rps == 7.5
    assert plan.diagnostics.applied_max_admission_rps == 8.0


def test_best_effort_drain_keeps_a_floor_that_can_sustain_it():
    plan = _plan(
        _tick_input(
            jobs=[_job("job-a", remaining_requests=100, deadline_offset_s=None)]
        )
    )

    assert plan.diagnostics.required_batch_rps == 0.0
    assert plan.drain_limit.max_admission_rps == 10.0
    assert plan.diagnostics.required_replica_floor == 9
    assert plan.replica_floor == 10
    assert plan.drain_limit.max_admission_rps <= (plan.replica_floor * 10.0 - 90.0)
    assert plan.diagnostics.minimum_deadline_slack_s is None


def test_drain_is_clipped_when_replica_ceiling_cannot_sustain_headroom():
    plan = _plan(
        _tick_input(
            jobs=[_job("job-a", remaining_requests=100, deadline_offset_s=None)]
        ),
        _config(max_replicas=9),
    )

    assert plan.diagnostics.current_safe_headroom_rps == 10.0
    assert plan.diagnostics.required_replica_floor == 9
    assert plan.replica_floor == 9
    assert plan.diagnostics.planned_batch_capacity_rps == 0.0
    assert plan.drain_limit.max_admission_rps == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"safe_rps_per_ready_replica": 0.0}, "positive and finite"),
        ({"cold_start_margin_s": -1.0}, "non-negative and finite"),
        ({"drain_lease_duration_s": 0.0}, "positive and finite"),
        ({"min_replicas": -1}, "min_replicas"),
        ({"min_replicas": 2, "max_replicas": 1}, "max_replicas"),
        ({"scale_from_zero_replicas": 0}, "scale_from_zero_replicas"),
        (
            {"scale_from_zero_replicas": 21},
            "scale_from_zero_replicas must be <= max_replicas",
        ),
        ({"max_batch_admission_rps": -1.0}, "non-negative and finite"),
    ],
)
def test_config_rejects_unsafe_values(overrides: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_decision_id_is_required():
    with pytest.raises(ValueError, match="decision_id"):
        plan_batch_schedule(_tick_input(), _config(), decision_id="")
