# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ThunderAgentScheduler that don't need a Dynamo runtime."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from dynamo.thunderagent_router.capacity import WorkerCapacity
from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus
from dynamo.thunderagent_router.router import ThunderAgentConfig, ThunderAgentScheduler

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@dataclass
class FakeCapacity:
    """Stand-in for WorkerCapacityProvider that returns a fixed snapshot."""

    workers: dict[int, int] = field(default_factory=dict)
    block_sizes: dict[int, int] = field(default_factory=dict)

    def snapshot(self) -> dict[int, WorkerCapacity]:
        return {
            worker_id: WorkerCapacity(
                retention_tokens=tokens,
                block_size=self.block_sizes.get(worker_id, 1),
            )
            for worker_id, tokens in self.workers.items()
        }


def make_router(
    capacity_workers: Optional[dict[int, int]] = None,
    config: Optional[ThunderAgentConfig] = None,
    block_size: int = 1,
) -> tuple[ThunderAgentScheduler, FakeCapacity]:
    workers = capacity_workers or {}
    capacity = FakeCapacity(
        workers=workers,
        block_sizes={worker_id: block_size for worker_id in workers},
    )
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
    await router.before_request("existing", estimated_prompt_tokens=850)
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

    # Used = 700 + 200 = 900; pausing small leaves 700 <= target.
    for pid, prompt_tokens in [("big", 700), ("small", 200)]:
        await router.before_request(pid)
        await router.assign_worker(pid, 1)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["big"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_until_safe_skips_zero_footprint_acting_program():
    block_size = 4160
    cfg = ThunderAgentConfig(
        buffer_per_program=0,
        pause_threshold=0.95,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    router, capacity = make_router(config=cfg)

    for program_id, prompt_tokens in [
        ("sub-block", block_size - 1),
        ("full-block", block_size),
    ]:
        await router.before_request(program_id)
        await router.assign_worker(program_id, 1)
        await router.after_request(
            program_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
        )

    await router.before_request("reasoning", estimated_prompt_tokens=2 * block_size)
    await router.assign_worker("reasoning", 1)
    capacity.workers = {1: 3 * block_size}
    capacity.block_sizes = {1: block_size}

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["sub-block"].lifecycle == ProgramLifecycle.ACTIVE
    assert router._table.programs["full-block"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["reasoning"].marked_for_pause is False


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
        ("hot_small", 1, 300),
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
    # ACTING programs retain complete reusable blocks without an in-flight
    # buffer. Ten 100k-token programs use 1M; pausing two reaches 0.80M.
    assert paused == 2, f"paused={paused}"


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
        await router.after_request(pid, prompt_tokens=200, completion_tokens=0)
        router._table.programs[pid].acting_since = time.monotonic() - 10.0

    capacity.workers = workers
    await router._scheduler_tick()

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    assert paused == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("block_size", [1, 64, 4160])
async def test_concurrent_partial_reasoning_requests_reserve_whole_blocks(block_size):
    buffer_per_program = 1
    required = ((1 + buffer_per_program + block_size - 1) // block_size) * block_size
    cfg = ThunderAgentConfig(
        buffer_per_program=buffer_per_program,
        scheduler_interval_seconds=10.0,
        resume_timeout_seconds=2.0,
    )
    router, _ = make_router(
        capacity_workers={1: 2 * required},
        config=cfg,
        block_size=block_size,
    )

    await router.before_request("p1", estimated_prompt_tokens=1)
    await router.before_request("p2", estimated_prompt_tokens=1)

    async with router._lock:
        wait_event, was_paused = router._admit_locked("p3", estimated_prompt_tokens=1)

    assert router._worker_used(1, block_size) == 2 * required
    assert wait_event is not None
    assert was_paused is True
    assert router._table.programs["p3"].lifecycle == ProgramLifecycle.PAUSED


@pytest.mark.asyncio
@pytest.mark.parametrize("block_size", [1, 64, 4160])
async def test_acting_transition_counts_only_complete_reusable_blocks(block_size):
    cfg = ThunderAgentConfig(
        buffer_per_program=0,
        scheduler_interval_seconds=10.0,
    )
    tokens = block_size + 1
    router, _ = make_router(
        capacity_workers={1: 3 * block_size},
        config=cfg,
        block_size=block_size,
    )

    await router.before_request("p1", estimated_prompt_tokens=tokens)
    assert router._worker_used(1, block_size) == 2 * block_size

    await router.after_request("p1", prompt_tokens=tokens, completion_tokens=0)
    complete_tokens = (tokens // block_size) * block_size
    assert router._worker_used(1, block_size) == complete_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("block_size", [1, 64, 4160])
async def test_acting_weight_applies_after_complete_block_accounting(block_size):
    cfg = ThunderAgentConfig(
        acting_token_weight=0.5,
        buffer_per_program=0,
        scheduler_interval_seconds=10.0,
    )
    router, _ = make_router(
        capacity_workers={1: 3 * block_size},
        config=cfg,
        block_size=block_size,
    )

    await router.before_request("p1", estimated_prompt_tokens=block_size + 1)
    await router.after_request("p1", prompt_tokens=block_size + 1, completion_tokens=0)

    complete_tokens = ((block_size + 1) // block_size) * block_size
    assert router._worker_used(1, block_size) == int(complete_tokens * 0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("block_size", [1, 64, 4160])
async def test_resume_placement_reserves_partial_program_as_whole_block(block_size):
    buffer_per_program = 1
    required = ((1 + buffer_per_program + block_size - 1) // block_size) * block_size
    cfg = ThunderAgentConfig(
        buffer_per_program=buffer_per_program,
        pause_threshold=1.0,
        resume_hysteresis=0.0,
        scheduler_interval_seconds=10.0,
    )
    router, _ = make_router(
        capacity_workers={1: 2 * required},
        config=cfg,
        block_size=block_size,
    )
    await router.before_request("active", estimated_prompt_tokens=1)

    for program_id in ("paused-1", "paused-2"):
        program = router._table.begin_request(program_id, estimated_prompt_tokens=1)
        program.lifecycle = ProgramLifecycle.PAUSED
        router._table.paused[program_id] = None

    await router._greedy_resume(router._capacity.snapshot())

    assert router._table.programs["paused-1"].lifecycle == ProgramLifecycle.ACTIVE
    assert router._table.programs["paused-2"].lifecycle == ProgramLifecycle.PAUSED
    assert router._worker_used(1, block_size) == 2 * required


@pytest.mark.asyncio
async def test_resume_preserves_selection_and_placement_with_shared_block_size():
    block_size = 64
    cfg = ThunderAgentConfig(
        buffer_per_program=0,
        pause_threshold=1.0,
        resume_hysteresis=0.0,
        scheduler_interval_seconds=10.0,
    )
    router, _ = make_router(
        capacity_workers={1: 2 * block_size, 2: 2 * block_size},
        config=cfg,
        block_size=block_size,
    )

    for program_id, tokens in [
        ("small-1", 1),
        ("large", block_size + 1),
        ("small-2", 1),
    ]:
        program = router._table.begin_request(
            program_id,
            estimated_prompt_tokens=tokens,
        )
        program.lifecycle = ProgramLifecycle.PAUSED
        router._table.paused[program_id] = None

    await router._greedy_resume(router._capacity.snapshot())

    assert router._table.programs["large"].assigned_worker_id == 1
    assert router._table.programs["small-1"].assigned_worker_id == 2
    assert router._table.programs["small-2"].assigned_worker_id == 2
    assert not router._table.paused
