# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure batch scheduling policy for one aggregate inference pool.

The policy performs no observation collection and no actuation. It turns one
``TickInput`` snapshot into two independently consumable recommendations:

* an absolute replica lower bound for the existing scaling pipeline; and
* a leased maximum batch-admission rate for a dispatcher.

Deadline pressure is computed with an earliest-deadline-first cumulative demand
bound. Current drain is work-conserving, but is never allowed to consume more
than the capacity left after online offered load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from dynamo.planner.core.types import (
    BatchDispatcherFeedback,
    BatchDrainLimitDecision,
    BatchJobDemand,
    PoolTrafficDemand,
    TickInput,
)

_NON_DISPATCHABLE_STATUSES = frozenset(
    {
        "canceled",
        "cancelled",
        "cancelling",
        "completed",
        "expired",
        "failed",
        "finalizing",
    }
)


@dataclass(frozen=True)
class BatchSchedulingPolicyConfig:
    """Static assumptions for the single-pool POC policy.

    ``safe_rps_per_ready_replica`` is calibrated for ``work_class``. The two
    margins reserve time before each absolute job deadline. The optional cap is
    an independent upper bound on batch admission, even when more pool
    headroom is available.
    """

    pool_id: str
    work_class: str
    safe_rps_per_ready_replica: float
    cold_start_margin_s: float
    finalization_margin_s: float
    max_observation_age_s: float
    drain_lease_duration_s: float
    min_replicas: int
    max_replicas: int
    scale_from_zero_replicas: int = 1
    max_batch_admission_rps: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.pool_id:
            raise ValueError("pool_id must not be empty")
        if not self.work_class:
            raise ValueError("work_class must not be empty")
        _require_positive_finite(
            "safe_rps_per_ready_replica", self.safe_rps_per_ready_replica
        )
        _require_non_negative_finite("cold_start_margin_s", self.cold_start_margin_s)
        _require_non_negative_finite(
            "finalization_margin_s", self.finalization_margin_s
        )
        _require_non_negative_finite(
            "max_observation_age_s", self.max_observation_age_s
        )
        _require_positive_finite("drain_lease_duration_s", self.drain_lease_duration_s)
        if self.min_replicas < 0:
            raise ValueError("min_replicas must be non-negative")
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        if self.scale_from_zero_replicas <= 0:
            raise ValueError("scale_from_zero_replicas must be positive")
        if self.scale_from_zero_replicas > self.max_replicas:
            raise ValueError("scale_from_zero_replicas must be <= max_replicas")
        if self.max_batch_admission_rps is not None:
            _require_non_negative_finite(
                "max_batch_admission_rps", self.max_batch_admission_rps
            )


@dataclass(frozen=True)
class BatchSchedulingDiagnostics:
    """Auditable intermediate values from one batch policy evaluation."""

    required_batch_rps: Optional[float]
    current_safe_headroom_rps: float
    planned_batch_capacity_rps: float
    predicted_finish_at_s: Optional[float]
    minimum_deadline_slack_s: Optional[float]
    infeasible: bool
    infeasible_reasons: tuple[str, ...]
    safety_paused: bool
    active_job_count: int
    remaining_requests: int
    ready_replicas: Optional[int]
    required_replica_floor: Optional[int]
    online_offered_rps: Optional[float]
    online_observation_age_s: Optional[float]
    oldest_job_observation_age_s: Optional[float]
    capacity_observation_stale: bool
    online_observation_stale: bool
    job_observation_stale: bool
    dispatcher_feedback_stale: bool
    dispatcher_observation_age_s: Optional[float]
    actual_dispatch_rps: Optional[float]
    applied_max_admission_rps: Optional[float]


@dataclass(frozen=True)
class BatchSchedulingPlan:
    """Pure policy result; callers merge the floor with other scale opinions."""

    replica_floor: int
    drain_limit: BatchDrainLimitDecision
    diagnostics: BatchSchedulingDiagnostics


def plan_batch_schedule(
    tick_input: TickInput,
    config: BatchSchedulingPolicyConfig,
    *,
    decision_id: str,
) -> BatchSchedulingPlan:
    """Compute a deterministic batch plan from one input snapshot.

    ``ready_num_decode`` is the aggregate POC's ready-worker count. A present
    worker snapshot is considered current for ``tick_input.now_s`` because the
    normal planner adapter gathers it for that tick. Missing or invalid worker
    state is treated as stale capacity.
    """

    if not decision_id:
        raise ValueError("decision_id must not be empty")
    if not math.isfinite(tick_input.now_s):
        raise ValueError("tick_input.now_s must be finite")

    now_s = tick_input.now_s
    batch = tick_input.batch
    ready_replicas, capacity_stale = _ready_replicas(tick_input)
    hold_floor = _hold_floor(ready_replicas, config)

    pool_traffic = (
        _latest_pool_traffic(batch.pool_traffic, config.pool_id) if batch else None
    )
    online_age_s, online_stale = _observation_freshness(
        pool_traffic.observed_at_s if pool_traffic else None,
        now_s,
        config.max_observation_age_s,
    )
    online_offered_rps = pool_traffic.online_offered_rps if pool_traffic else None
    if online_offered_rps is not None and (
        not math.isfinite(online_offered_rps) or online_offered_rps < 0
    ):
        online_stale = True

    active_jobs = _active_jobs(batch.job_demands, config.pool_id) if batch else []
    oldest_job_age_s: Optional[float] = None
    job_stale = False
    mismatched_jobs: list[BatchJobDemand] = []
    invalid_jobs: list[BatchJobDemand] = []
    for job in active_jobs:
        age_s, stale = _observation_freshness(
            job.observed_at_s,
            now_s,
            config.max_observation_age_s,
        )
        if age_s is not None:
            oldest_job_age_s = (
                age_s if oldest_job_age_s is None else max(oldest_job_age_s, age_s)
            )
        job_stale = job_stale or stale
        if job.work_class != config.work_class:
            mismatched_jobs.append(job)
        if job.remaining_requests < 0 or (
            job.deadline_at_s is not None and not math.isfinite(job.deadline_at_s)
        ):
            invalid_jobs.append(job)

    dispatcher_feedback = (
        _latest_dispatcher_feedback(batch.dispatcher_feedback, config.pool_id)
        if batch
        else None
    )
    dispatcher_age_s, dispatcher_stale = _observation_freshness(
        dispatcher_feedback.observed_at_s if dispatcher_feedback else None,
        now_s,
        config.max_observation_age_s,
    )

    safety_reasons: list[str] = []
    if batch is None:
        safety_reasons.append("batch_observation_missing")
    if capacity_stale:
        safety_reasons.append("capacity_observation_missing_or_invalid")
    if pool_traffic is None:
        safety_reasons.append("online_traffic_observation_missing")
    elif online_stale:
        safety_reasons.append("online_traffic_observation_stale_or_invalid")
    if job_stale:
        safety_reasons.append("active_job_observation_stale_or_invalid")
    if invalid_jobs:
        safety_reasons.extend(
            f"active_job_invalid:{job.job_id}" for job in invalid_jobs
        )
    if mismatched_jobs:
        safety_reasons.extend(
            f"work_class_mismatch:{job.job_id}:{job.work_class}"
            for job in mismatched_jobs
        )

    # A fresh durable job snapshot is enough to break the scale-from-zero
    # observation loop: frontend traffic may remain unavailable until a worker
    # is serving, but admission remains fail-closed until that sample exists.
    # Stale or malformed demand must never trigger capacity.
    scale_from_zero_floor = (
        config.scale_from_zero_replicas
        if ready_replicas == 0
        and active_jobs
        and not job_stale
        and not invalid_jobs
        and not mismatched_jobs
        else 0
    )

    if safety_reasons:
        return _safety_pause_plan(
            now_s=now_s,
            decision_id=decision_id,
            config=config,
            replica_floor=max(hold_floor, scale_from_zero_floor),
            required_replica_floor=(
                scale_from_zero_floor if scale_from_zero_floor > 0 else None
            ),
            reasons=tuple(safety_reasons),
            active_jobs=active_jobs,
            ready_replicas=ready_replicas,
            online_offered_rps=online_offered_rps,
            online_age_s=online_age_s,
            oldest_job_age_s=oldest_job_age_s,
            capacity_stale=capacity_stale,
            online_stale=online_stale,
            job_stale=job_stale,
            dispatcher_feedback=dispatcher_feedback,
            dispatcher_age_s=dispatcher_age_s,
            dispatcher_stale=dispatcher_stale,
        )

    # These values are proven present by the safety gate above.
    assert ready_replicas is not None
    assert online_offered_rps is not None

    required_batch_rps, demand_bounds, infeasible_reasons = _edf_demand_bound(
        active_jobs,
        now_s,
        config,
    )
    required_replica_floor: Optional[int]
    if math.isfinite(required_batch_rps):
        required_replica_floor = max(
            config.min_replicas,
            scale_from_zero_floor,
            math.ceil(
                (online_offered_rps + required_batch_rps)
                / config.safe_rps_per_ready_replica
            ),
        )
        replica_floor = min(required_replica_floor, config.max_replicas)
        if required_replica_floor > config.max_replicas:
            infeasible_reasons.append("required_replica_floor_exceeds_max_replicas")
    else:
        required_replica_floor = None
        replica_floor = config.max_replicas

    if (
        config.max_batch_admission_rps is not None
        and required_batch_rps > config.max_batch_admission_rps
    ):
        infeasible_reasons.append("batch_cap_below_required_rps")

    current_safe_headroom_rps = max(
        0.0,
        ready_replicas * config.safe_rps_per_ready_replica - online_offered_rps,
    )
    candidate_drain_cap_rps = current_safe_headroom_rps if active_jobs else 0.0
    if config.max_batch_admission_rps is not None:
        candidate_drain_cap_rps = min(
            candidate_drain_cap_rps,
            config.max_batch_admission_rps,
        )

    # Scaling and drain decisions are applied independently downstream, so the
    # advertised replica floor must be able to sustain the advertised drain.
    # Without this coupling, a best-effort job could consume today's headroom
    # while another scaling opinion simultaneously reduced the fleet to the
    # deadline-only floor.
    if candidate_drain_cap_rps > 0:
        drain_sustaining_floor = math.ceil(
            (online_offered_rps + candidate_drain_cap_rps)
            / config.safe_rps_per_ready_replica
        )
        replica_floor = min(
            config.max_replicas,
            max(replica_floor, drain_sustaining_floor),
        )

    # Predict against the advertised lower bound, not today's possibly larger
    # fleet. This keeps the deadline/slack claim valid if another scaling
    # opinion elects to scale down to exactly this floor.
    planned_replicas = replica_floor
    planned_batch_capacity_rps = max(
        0.0,
        planned_replicas * config.safe_rps_per_ready_replica - online_offered_rps,
    )
    if config.max_batch_admission_rps is not None:
        planned_batch_capacity_rps = min(
            planned_batch_capacity_rps,
            config.max_batch_admission_rps,
        )

    # A configured replica ceiling can prevent the floor from preserving all
    # current headroom. Clip the drain in that case so the joint decision still
    # satisfies online + batch <= floor * safe per-replica capacity.
    drain_cap_rps = min(
        candidate_drain_cap_rps,
        planned_batch_capacity_rps,
    )

    predicted_finish_at_s, minimum_slack_s = _finish_diagnostics(
        active_jobs=active_jobs,
        demand_bounds=demand_bounds,
        now_s=now_s,
        planned_batch_capacity_rps=planned_batch_capacity_rps,
        scaling_up=replica_floor > ready_replicas,
        config=config,
    )
    if demand_bounds and planned_batch_capacity_rps <= 0:
        infeasible_reasons.append("no_planned_batch_capacity")
    elif minimum_slack_s is not None and minimum_slack_s < -1e-9:
        infeasible_reasons.append("negative_deadline_slack")

    infeasible_reasons = list(dict.fromkeys(infeasible_reasons))
    diagnostics = BatchSchedulingDiagnostics(
        required_batch_rps=required_batch_rps,
        current_safe_headroom_rps=current_safe_headroom_rps,
        planned_batch_capacity_rps=planned_batch_capacity_rps,
        predicted_finish_at_s=predicted_finish_at_s,
        minimum_deadline_slack_s=minimum_slack_s,
        infeasible=bool(infeasible_reasons),
        infeasible_reasons=tuple(infeasible_reasons),
        safety_paused=False,
        active_job_count=len(active_jobs),
        remaining_requests=sum(job.remaining_requests for job in active_jobs),
        ready_replicas=ready_replicas,
        required_replica_floor=required_replica_floor,
        online_offered_rps=online_offered_rps,
        online_observation_age_s=online_age_s,
        oldest_job_observation_age_s=oldest_job_age_s,
        capacity_observation_stale=False,
        online_observation_stale=False,
        job_observation_stale=False,
        dispatcher_feedback_stale=dispatcher_stale,
        dispatcher_observation_age_s=dispatcher_age_s,
        actual_dispatch_rps=(
            dispatcher_feedback.actual_dispatch_rps if dispatcher_feedback else None
        ),
        applied_max_admission_rps=(
            dispatcher_feedback.applied_max_admission_rps
            if dispatcher_feedback
            else None
        ),
    )
    return BatchSchedulingPlan(
        replica_floor=replica_floor,
        drain_limit=BatchDrainLimitDecision(
            pool_id=config.pool_id,
            max_admission_rps=drain_cap_rps,
            valid_until_s=now_s + config.drain_lease_duration_s,
            decision_id=decision_id,
        ),
        diagnostics=diagnostics,
    )


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_non_negative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative and finite")


def _ready_replicas(tick_input: TickInput) -> tuple[Optional[int], bool]:
    worker_counts = tick_input.worker_counts
    if worker_counts is None or worker_counts.ready_num_decode is None:
        return None, True
    ready_replicas = worker_counts.ready_num_decode
    if ready_replicas < 0:
        return None, True
    return ready_replicas, False


def _hold_floor(
    ready_replicas: Optional[int], config: BatchSchedulingPolicyConfig
) -> int:
    if ready_replicas is None:
        return config.max_replicas
    return min(config.max_replicas, max(config.min_replicas, ready_replicas))


def _observation_freshness(
    observed_at_s: Optional[float],
    now_s: float,
    max_age_s: float,
) -> tuple[Optional[float], bool]:
    if observed_at_s is None or not math.isfinite(observed_at_s):
        return None, True
    age_s = now_s - observed_at_s
    return age_s, age_s < 0 or age_s > max_age_s


def _latest_pool_traffic(
    observations: list[PoolTrafficDemand], pool_id: str
) -> Optional[PoolTrafficDemand]:
    matches = [
        observation for observation in observations if observation.pool_id == pool_id
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda observation: (
            _sortable_timestamp(observation.observed_at_s),
            observation.online_offered_rps,
        ),
    )


def _latest_dispatcher_feedback(
    observations: list[BatchDispatcherFeedback], pool_id: str
) -> Optional[BatchDispatcherFeedback]:
    matches = [
        observation for observation in observations if observation.pool_id == pool_id
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda observation: (
            _sortable_timestamp(observation.observed_at_s),
            observation.queued_requests,
            observation.inflight_requests,
            observation.actual_dispatch_rps,
        ),
    )


def _active_jobs(
    observations: list[BatchJobDemand], pool_id: str
) -> list[BatchJobDemand]:
    latest_by_id: dict[str, BatchJobDemand] = {}
    for job in observations:
        if job.pool_id != pool_id:
            continue
        previous = latest_by_id.get(job.job_id)
        if previous is None or _job_order_key(job) > _job_order_key(previous):
            latest_by_id[job.job_id] = job
    return sorted(
        (
            job
            for job in latest_by_id.values()
            if job.remaining_requests != 0
            and job.status.strip().lower() not in _NON_DISPATCHABLE_STATUSES
        ),
        key=lambda job: job.job_id,
    )


def _job_order_key(job: BatchJobDemand) -> tuple[float, str, int, int, int, float, str]:
    return (
        _sortable_timestamp(job.observed_at_s),
        job.status,
        job.total_requests,
        job.completed_requests,
        job.failed_requests,
        (
            _sortable_timestamp(job.deadline_at_s)
            if job.deadline_at_s is not None
            else math.inf
        ),
        job.work_class,
    )


def _sortable_timestamp(timestamp_s: float) -> float:
    return timestamp_s if math.isfinite(timestamp_s) else math.inf


def _edf_demand_bound(
    active_jobs: list[BatchJobDemand],
    now_s: float,
    config: BatchSchedulingPolicyConfig,
) -> tuple[float, list[tuple[float, int]], list[str]]:
    deadline_jobs = sorted(
        (job for job in active_jobs if job.deadline_at_s is not None),
        key=lambda job: (job.deadline_at_s, job.job_id),
    )
    cumulative_requests = 0
    required_batch_rps = 0.0
    demand_bounds: list[tuple[float, int]] = []
    infeasible_reasons: list[str] = []
    for job in deadline_jobs:
        assert job.deadline_at_s is not None
        cumulative_requests += job.remaining_requests
        demand_bounds.append((job.deadline_at_s, cumulative_requests))
        available_s = (
            job.deadline_at_s
            - now_s
            - config.cold_start_margin_s
            - config.finalization_margin_s
        )
        if available_s <= 0:
            required_batch_rps = math.inf
            infeasible_reasons.append(f"deadline_window_exhausted:{job.job_id}")
            continue
        required_batch_rps = max(
            required_batch_rps,
            cumulative_requests / available_s,
        )
    return required_batch_rps, demand_bounds, infeasible_reasons


def _finish_diagnostics(
    *,
    active_jobs: list[BatchJobDemand],
    demand_bounds: list[tuple[float, int]],
    now_s: float,
    planned_batch_capacity_rps: float,
    scaling_up: bool,
    config: BatchSchedulingPolicyConfig,
) -> tuple[Optional[float], Optional[float]]:
    remaining_requests = sum(job.remaining_requests for job in active_jobs)
    if remaining_requests == 0:
        return now_s, None
    if planned_batch_capacity_rps <= 0:
        return None, -math.inf if demand_bounds else None

    start_at_s = now_s + (config.cold_start_margin_s if scaling_up else 0.0)
    predicted_finish_at_s = (
        start_at_s
        + remaining_requests / planned_batch_capacity_rps
        + config.finalization_margin_s
    )
    slacks = [
        deadline_at_s
        - (
            start_at_s
            + cumulative_requests / planned_batch_capacity_rps
            + config.finalization_margin_s
        )
        for deadline_at_s, cumulative_requests in demand_bounds
    ]
    return predicted_finish_at_s, min(slacks) if slacks else None


def _safety_pause_plan(
    *,
    now_s: float,
    decision_id: str,
    config: BatchSchedulingPolicyConfig,
    replica_floor: int,
    required_replica_floor: Optional[int],
    reasons: tuple[str, ...],
    active_jobs: list[BatchJobDemand],
    ready_replicas: Optional[int],
    online_offered_rps: Optional[float],
    online_age_s: Optional[float],
    oldest_job_age_s: Optional[float],
    capacity_stale: bool,
    online_stale: bool,
    job_stale: bool,
    dispatcher_feedback: Optional[BatchDispatcherFeedback],
    dispatcher_age_s: Optional[float],
    dispatcher_stale: bool,
) -> BatchSchedulingPlan:
    diagnostics = BatchSchedulingDiagnostics(
        required_batch_rps=None,
        current_safe_headroom_rps=0.0,
        planned_batch_capacity_rps=0.0,
        predicted_finish_at_s=None,
        minimum_deadline_slack_s=None,
        infeasible=True,
        infeasible_reasons=reasons,
        safety_paused=True,
        active_job_count=len(active_jobs),
        remaining_requests=sum(max(0, job.remaining_requests) for job in active_jobs),
        ready_replicas=ready_replicas,
        required_replica_floor=required_replica_floor,
        online_offered_rps=online_offered_rps,
        online_observation_age_s=online_age_s,
        oldest_job_observation_age_s=oldest_job_age_s,
        capacity_observation_stale=capacity_stale,
        online_observation_stale=online_stale,
        job_observation_stale=job_stale,
        dispatcher_feedback_stale=dispatcher_stale,
        dispatcher_observation_age_s=dispatcher_age_s,
        actual_dispatch_rps=(
            dispatcher_feedback.actual_dispatch_rps if dispatcher_feedback else None
        ),
        applied_max_admission_rps=(
            dispatcher_feedback.applied_max_admission_rps
            if dispatcher_feedback
            else None
        ),
    )
    return BatchSchedulingPlan(
        replica_floor=replica_floor,
        drain_limit=BatchDrainLimitDecision(
            pool_id=config.pool_id,
            max_admission_rps=0.0,
            valid_until_s=now_s + config.drain_lease_duration_s,
            decision_id=decision_id,
        ),
        diagnostics=diagnostics,
    )


__all__ = [
    "BatchSchedulingDiagnostics",
    "BatchSchedulingPlan",
    "BatchSchedulingPolicyConfig",
    "plan_batch_schedule",
]
