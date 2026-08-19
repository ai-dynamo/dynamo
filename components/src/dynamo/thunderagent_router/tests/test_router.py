# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ThunderAgentScheduler that don't need a Dynamo runtime."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus
from dynamo.thunderagent_router.router import ThunderAgentConfig, ThunderAgentScheduler

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@dataclass
class FakeCapacity:
    """Stand-in for WorkerCapacityProvider with configurable worker state."""

    workers: dict[int, int] = field(default_factory=dict)
    live_workers: Optional[set[int]] = None

    def snapshot(self) -> dict[int, int]:
        return dict(self.workers)

    def live_worker_ids(self) -> set[int]:
        if self.live_workers is not None:
            return set(self.live_workers)
        return set(self.workers)


def make_router(
    capacity_workers: Optional[dict[int, int]] = None,
    config: Optional[ThunderAgentConfig] = None,
) -> tuple[ThunderAgentScheduler, FakeCapacity]:
    capacity = FakeCapacity(workers=capacity_workers or {})
    cfg = config or ThunderAgentConfig(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
        pause_threshold=0.95,
        soft_demote_threshold=0.80,
    )
    return ThunderAgentScheduler(capacity, cfg), capacity  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_first_turn_no_admission_block():
    router, _ = make_router()
    decision = await router.before_request("p1")
    assert decision.was_paused is False
    assert decision.priority_jump == 0.0


@pytest.mark.asyncio
async def test_after_request_records_real_tokens():
    router, _ = make_router()
    await router.before_request("p1")
    await router.after_request("p1", prompt_tokens=120, completion_tokens=30)
    program = router._table.programs["p1"]
    assert program.token_total == 150
    assert program.status == ProgramStatus.ACTING


@pytest.mark.asyncio
async def test_status_snapshot_reports_programs_and_worker_utilization():
    workers = {
        1: 1000,
        2: 500,
    }
    router, _ = make_router(capacity_workers=workers)

    await router.before_request("p1", estimated_prompt_tokens=100)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=25)
    await router.before_request("p2", estimated_prompt_tokens=50)

    snapshot = await router.status_snapshot()

    assert snapshot["programs_total"] == 2
    assert snapshot["paused_total"] == 0
    assert snapshot["lifecycle_counts"]["active"] == 2
    assert snapshot["status_counts"]["acting"] == 1
    assert snapshot["status_counts"]["reasoning"] == 1
    assert snapshot["workers"]["1"]["capacity"] == 1000
    assert snapshot["workers"]["1"]["used"] == 225
    assert snapshot["workers"]["1"]["active_programs"] == 1
    assert {
        (program["program_id"], program["assigned_worker_id"])
        for program in snapshot["programs"]
    } == {("p1", 1), ("p2", 2)}


@pytest.mark.asyncio
async def test_metrics_snapshot_reports_lifecycle_counters_and_gauges():
    router, _ = make_router(capacity_workers={1: 1000})

    await router.before_request("p1", estimated_prompt_tokens=100)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=20)
    assert await router.end_program("p1") is True

    async def fail_status_snapshot() -> dict:
        raise AssertionError("metrics_snapshot must not build detailed status rows")

    router.status_snapshot = fail_status_snapshot  # type: ignore[method-assign]

    metrics = await router.metrics_snapshot()

    assert metrics["counters"]["programs_created_total"] == 1
    assert metrics["counters"]["programs_ended_total"] == 1
    assert metrics["counters"]["requests_admitted_total"] == 1
    assert metrics["counters"]["worker_assignments_total"] == 1
    assert metrics["gauges"]["programs_total"] == 0
    assert metrics["gauges"]["paused_total"] == 0
    assert metrics["gauges"]["workers_total"] == 1


@pytest.mark.asyncio
async def test_before_request_records_exact_prompt_estimate_before_admission():
    router, _ = make_router()
    await router.before_request("p1", estimated_prompt_tokens=1234)
    program = router._table.programs["p1"]
    assert program.token_total == 1234
    assert program.status == ProgramStatus.REASONING


@pytest.mark.asyncio
async def test_assigned_worker_hint_reflects_sticky_assignment():
    router, _ = make_router()
    await router.before_request("p1", estimated_prompt_tokens=100)
    await router.assign_worker("p1", 3)
    decision = await router.before_request("p1", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 3


@pytest.mark.asyncio
async def test_stale_worker_assignment_moves_to_replacement():
    router, capacity = make_router(capacity_workers={1: 1000})
    decision = await router.before_request("p1", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 1

    capacity.workers = {}
    decision = await router.before_request("p1", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 1

    capacity.workers = {2: 1000}
    capacity.live_workers = {1, 2}
    decision = await router.before_request("p1", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 1

    capacity.live_workers = {2}
    decision = await router.before_request("p1", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 2
    assert router._stat_worker_assignments == 2


@pytest.mark.asyncio
async def test_stale_replacement_bypasses_new_program_fairness_gate():
    router, capacity = make_router(capacity_workers={1: 300})
    decision = await router.before_request("active", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 1

    waiter = asyncio.create_task(
        router.before_request("waiting", estimated_prompt_tokens=100)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)
    assert router._table.programs["waiting"].lifecycle == ProgramLifecycle.PAUSED

    capacity.workers = {2: 1000}
    capacity.live_workers = {2}
    decision = await router.before_request("active", estimated_prompt_tokens=100)
    assert decision.assigned_worker_hint == 2

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


@pytest.mark.asyncio
async def test_pause_acting_then_before_request_blocks_until_resume():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
    )
    router, _ = make_router(config=cfg)

    await router.before_request("p1")
    await router.assign_worker("p1", 0)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)
    await router._pause_acting("p1")
    assert router._table.programs["p1"].lifecycle == ProgramLifecycle.PAUSED

    waiter = asyncio.create_task(router.before_request("p1"))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)

    async with router._lock:
        router._resume_program(router._table.programs["p1"], target_worker_id=1)

    decision = await asyncio.wait_for(waiter, timeout=1.0)
    assert decision.was_paused is True
    assert decision.priority_jump == cfg.resume_priority_boost
    assert decision.assigned_worker_hint == 1
    metrics = await router.metrics_snapshot()
    assert metrics["counters"]["worker_assignments_total"] == 2


@pytest.mark.asyncio
async def test_forced_resume_after_timeout():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        resume_timeout_seconds=0.05,
    )
    router, _ = make_router(config=cfg)
    await router.before_request("p1")
    await router.assign_worker("p1", 0)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)
    await router._pause_acting("p1")
    decision = await router.before_request("p1")
    assert decision.was_paused is True
    assert router._stat_forced_resumes >= 1
    assert router._table.programs["p1"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_new_program_queues_before_first_request_when_capacity_full():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        resume_timeout_seconds=2.0,
        pause_threshold=1.0,
        resume_hysteresis=0.0,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    await router.before_request("existing", estimated_prompt_tokens=950)
    await router.assign_worker("existing", 1)

    waiter = asyncio.create_task(
        router.before_request("new", estimated_prompt_tokens=100)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)
    assert router._table.programs["new"].lifecycle == ProgramLifecycle.PAUSED

    async with router._lock:
        router._resume_program(router._table.programs["new"], target_worker_id=1)
    decision = await asyncio.wait_for(waiter, timeout=1.0)
    assert decision.was_paused is True


@pytest.mark.asyncio
async def test_cold_start_admits_without_sticky_pin():
    """No MDC visible yet: don't park, let the request through; the
    chunk-loop callback will populate ``assigned_worker_id`` once the
    engine picks a worker."""
    router, _ = make_router(capacity_workers={})
    decision = await router.before_request("cold_start")
    assert decision.was_paused is False
    assert decision.assigned_worker_hint is None
    program = router._table.programs["cold_start"]
    assert program.lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_soft_demote_marks_borderline_workers():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        soft_demote_threshold=0.80,
        pause_threshold=0.95,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    await router.before_request("p1")
    await router.assign_worker("p1", 1)
    await router.after_request("p1", prompt_tokens=750, completion_tokens=0)
    await router.before_request("p1")
    await router.assign_worker("p1", 1)

    router._apply_soft_demotes(router._capacity.snapshot())
    program = router._table.programs["p1"]
    assert program.soft_demoted_until > time.monotonic()

    await router.after_request("p1", prompt_tokens=860, completion_tokens=2)
    decision = await router.before_request("p1")
    assert decision.priority_jump == cfg.soft_demote_priority_jump
    assert decision.was_soft_demoted is True


@pytest.mark.asyncio
async def test_pause_until_safe_pauses_smallest_acting_first():
    cfg = ThunderAgentConfig(
        pause_threshold=0.80,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)

    # Used = 600 + 100 + 2*100 = 900; pausing small leaves 700 <= target.
    for pid, prompt_tokens in [("big", 600), ("small", 100)]:
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["big"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_until_safe_is_scoped_to_overloaded_worker():
    cfg = ThunderAgentConfig(
        pause_threshold=0.95,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
        2: 1000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)

    for pid, worker_id, prompt_tokens in [
        ("hot_big", 1, 700),
        ("hot_small", 1, 200),
        ("cold", 2, 700),
    ]:
        await router.before_request(pid)
        await router.assign_worker(pid, worker_id)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["hot_small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["hot_big"].lifecycle == ProgramLifecycle.ACTIVE
    assert router._table.programs["cold"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_drives_util_to_pause_target_not_threshold():
    """Each pause cycle drains util down to pause_target, not just below threshold."""
    cfg = ThunderAgentConfig(
        pause_threshold=0.95,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1_000_000,
    }
    router, _ = make_router(capacity_workers=workers, config=cfg)
    for i in range(10):
        pid = f"p{i}"
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(pid, prompt_tokens=100_000, completion_tokens=0)

    await router._pause_until_safe(router._capacity.snapshot())

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    # 10 programs * (100k tokens + 100 buffer) = 1.0010M; target 0.80M.
    # Each pause releases (100k + 100). Pause 2 -> 0.8008M (still over),
    # pause 3 -> 0.7007M (under). Anything else means over- or under-shoot.
    assert paused == 3, f"paused={paused}"


@pytest.mark.asyncio
async def test_scheduler_tick_resumes_before_pausing_new_overload():
    """Upstream TA ordering: resume old paused work, then pause overload."""
    cfg = ThunderAgentConfig(
        pause_threshold=1.0,
        pause_target=0.80,
        resume_hysteresis=0.0,
        acting_token_weight=1.0,
        acting_decay_tau_seconds=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        1: 1000,
    }
    router, capacity = make_router(config=cfg)

    # Capacity is attached after setup so first-turn admission gating does not
    # queue the synthetic programs before the scheduler tick.
    for i in range(10):
        pid = f"p{i}"
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(pid, prompt_tokens=100, completion_tokens=0)
        router._table.programs[pid].acting_since = time.monotonic() - 10.0

    capacity.workers = workers
    await router._scheduler_tick()

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    assert paused == 6


@pytest.mark.fault_tolerance
@pytest.mark.asyncio
async def test_cancelled_admission_of_new_program_leaves_no_trace():
    """A cancelled first-turn admission must not outlive the request.

    The parked program is small and the program holding the worker is large, so
    a tick after the blocker is released has room to resume the phantom -- which
    is what makes the phantom's absence, rather than a lucky capacity shortage,
    the reason nothing is assigned to it.
    """
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=1000.0,
        resume_timeout_seconds=0.5,
        pause_threshold=1.0,
        pause_target=1.0,
        resume_hysteresis=0.0,
    )
    router, _ = make_router(capacity_workers={1: 1000}, config=cfg)

    decision = await router.before_request("existing", estimated_prompt_tokens=850)
    assert decision.assigned_worker_hint == 1
    assert router._worker_used(1) == 950

    waiter = asyncio.create_task(
        router.before_request("new", estimated_prompt_tokens=50)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)
    assert "new" in router._table.paused

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert "new" not in router._table.programs
    assert "new" not in router._table.paused

    assert await router.end_program("existing") is True
    decision = await asyncio.wait_for(
        router.before_request("other", estimated_prompt_tokens=50), timeout=1.5
    )
    assert decision.was_paused is False
    assert decision.assigned_worker_hint == 1

    assignments_before_tick = router._stat_worker_assignments
    await router._scheduler_tick()

    assert "new" not in router._table.programs
    assert router._worker_used(1) == 150
    assert router._stat_worker_assignments == assignments_before_tick


@pytest.mark.fault_tolerance
@pytest.mark.asyncio
async def test_cancelled_admission_of_existing_program_restores_prior_turn():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=1000.0,
        resume_timeout_seconds=2.0,
    )
    router, _ = make_router(config=cfg)

    await router.before_request("p1")
    await router.assign_worker("p1", 1)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)
    await router._pause_acting("p1")

    program = router._table.programs["p1"]
    assert program.lifecycle == ProgramLifecycle.PAUSED
    assert program.status == ProgramStatus.ACTING
    assert program.step_count == 1
    assert program.token_total == 110
    assert program.assigned_worker_id is None
    assert "p1" in router._table.paused
    waiting_before = program.waiting
    acting_since_before = program.acting_since
    assert waiting_before is not None
    assert not waiting_before.is_set()

    waiter = asyncio.create_task(
        router.before_request("p1", estimated_prompt_tokens=500)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    # Session history survives: the cancelled turn is undone, not the program.
    assert router._table.programs["p1"] is program
    assert program.lifecycle == ProgramLifecycle.PAUSED
    assert program.status == ProgramStatus.ACTING
    assert program.step_count == 1
    assert program.token_total == 110
    assert program.assigned_worker_id is None
    assert program.acting_since == acting_since_before
    assert "p1" in router._table.paused
    assert program.waiting is waiting_before
    assert not program.waiting.is_set()


@pytest.mark.fault_tolerance
@pytest.mark.asyncio
async def test_cancelled_admission_does_not_strand_a_concurrent_waiter():
    """Two turns of one session share a program, so one cancelling must not
    delete it out from under the other: the survivor is parked on the program's
    Event and only a resume of that program can wake it."""
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=1000.0,
        resume_timeout_seconds=5.0,
        pause_threshold=1.0,
        pause_target=1.0,
        resume_hysteresis=0.0,
    )
    router, _ = make_router(capacity_workers={1: 1000}, config=cfg)

    decision = await router.before_request("existing", estimated_prompt_tokens=850)
    assert decision.assigned_worker_hint == 1

    first = asyncio.create_task(
        router.before_request("shared", estimated_prompt_tokens=50)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(first), timeout=0.05)
    assert "shared" in router._table.paused

    second = asyncio.create_task(
        router.before_request("shared", estimated_prompt_tokens=50)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(second), timeout=0.05)
    wait_event = router._table.programs["shared"].waiting

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    assert "shared" in router._table.programs
    assert "shared" in router._table.paused
    assert router._table.programs["shared"].waiting is wait_event

    assert await router.end_program("existing") is True
    await router._scheduler_tick()

    decision = await asyncio.wait_for(second, timeout=1.5)
    assert decision.was_paused is True
    assert decision.assigned_worker_hint == 1


@pytest.mark.fault_tolerance
@pytest.mark.asyncio
async def test_cancelled_admission_does_not_undo_a_later_turn():
    """Both turns mutate the same Program object, so object identity alone
    cannot tell the cancelled turn's mutations from the live turn's."""
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=1000.0,
        resume_timeout_seconds=5.0,
    )
    router, _ = make_router(config=cfg)

    await router.before_request("p1")
    await router.assign_worker("p1", 1)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)
    await router._pause_acting("p1")
    program = router._table.programs["p1"]

    cancelled = asyncio.create_task(
        router.before_request("p1", estimated_prompt_tokens=500)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(cancelled), timeout=0.05)

    later = asyncio.create_task(
        router.before_request("p1", estimated_prompt_tokens=777)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(later), timeout=0.05)
    assert program.token_total == 777
    assert program.step_count == 3

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    # The later turn's admission is still in force; restoring the cancelled
    # turn's snapshot would have put token_total back to 110 and step_count to 1.
    assert router._table.programs["p1"] is program
    assert program.token_total == 777
    assert program.step_count == 3
    assert program.lifecycle == ProgramLifecycle.PAUSED

    later.cancel()
    with pytest.raises(asyncio.CancelledError):
        await later


@pytest.mark.fault_tolerance
@pytest.mark.asyncio
async def test_rollback_completes_when_cancellation_is_redelivered():
    """A scheduler tick holds the lock across awaits, so the rollback's
    acquisition can block long enough to be cancelled a second time."""
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=1000.0,
        resume_timeout_seconds=5.0,
        pause_threshold=1.0,
        pause_target=1.0,
        resume_hysteresis=0.0,
    )
    router, _ = make_router(capacity_workers={1: 1000}, config=cfg)

    decision = await router.before_request("existing", estimated_prompt_tokens=850)
    assert decision.assigned_worker_hint == 1

    waiter = asyncio.create_task(
        router.before_request("new", estimated_prompt_tokens=50)
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)
    assert "new" in router._table.paused

    await router._lock.acquire()
    try:
        waiter.cancel()
        await asyncio.sleep(0.01)
        assert "new" in router._table.programs, "rollback should be parked on the lock"
        waiter.cancel()
        await asyncio.sleep(0.01)
        assert "new" in router._table.programs
    finally:
        router._lock.release()

    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert "new" not in router._table.programs
    assert "new" not in router._table.paused
