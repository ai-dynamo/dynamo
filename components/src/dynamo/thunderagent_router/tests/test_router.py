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
    """Stand-in for WorkerCapacityProvider that returns a fixed snapshot."""

    workers: dict[int, int] = field(default_factory=dict)

    def snapshot(self) -> dict[int, int]:
        return dict(self.workers)


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
