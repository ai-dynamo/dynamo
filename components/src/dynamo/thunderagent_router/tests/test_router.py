# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KvThunderAgentRouter that don't need a Dynamo runtime.

These mock out KvRouter and FpmCapacityProvider so we can validate the v0
scheduler primitives (pause/resume gating, soft demotion, sticky worker
placement) deterministically.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from dynamo.thunderagent_router.capacity import WorkerCapacity, WorkerKey
from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus
from dynamo.thunderagent_router.router import KvThunderAgentRouter, ThunderAgentConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@dataclass
class FakeCapacity:
    """Stand-in for FpmCapacityProvider that returns a fixed snapshot."""

    workers: dict[WorkerKey, WorkerCapacity] = field(default_factory=dict)

    def snapshot(self) -> dict[WorkerKey, WorkerCapacity]:
        return dict(self.workers)


class FakeKvRouter:
    """Records calls to ``best_worker``; the router class only invokes that method."""

    def __init__(self, response: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.response = response
        self.calls: list[list[int]] = []

    async def best_worker(self, token_ids):
        self.calls.append(list(token_ids))
        return self.response


def make_router(
    capacity_workers: Optional[dict[WorkerKey, WorkerCapacity]] = None,
    kv_response: tuple[int, int, int] = (7, 0, 0),
    config: Optional[ThunderAgentConfig] = None,
) -> tuple[KvThunderAgentRouter, FakeCapacity, FakeKvRouter]:
    capacity = FakeCapacity(workers=capacity_workers or {})
    kv = FakeKvRouter(response=kv_response)
    cfg = config or ThunderAgentConfig(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
        pause_threshold=0.95,
        soft_demote_threshold=0.80,
    )
    return KvThunderAgentRouter(kv, capacity, cfg), capacity, kv  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_first_turn_no_admission_block():
    router, _, _ = make_router()
    decision = await router.before_request("p1")
    assert decision.was_paused is False
    assert decision.priority_jump == 0.0


@pytest.mark.asyncio
async def test_after_request_records_real_tokens():
    router, _, _ = make_router()
    await router.before_request("p1")
    await router.after_request(
        "p1",
        prompt_tokens=120,
        completion_tokens=30,
    )
    program = router._table.programs["p1"]
    assert program.token_total == 150
    assert program.status == ProgramStatus.ACTING


@pytest.mark.asyncio
async def test_before_request_records_exact_prompt_estimate_before_admission():
    router, _, _ = make_router()
    await router.before_request("p1", estimated_prompt_tokens=1234)
    program = router._table.programs["p1"]
    assert program.token_total == 1234
    assert program.status == ProgramStatus.REASONING


@pytest.mark.asyncio
async def test_select_worker_returns_none_when_no_sticky_assignment():
    """Without a prior sticky assignment, select_worker returns None so the
    KvRouter picks freely from current token_ids."""
    router, _, kv = make_router(kv_response=(42, 0, 7))
    await router.before_request("p1")
    await router.after_request(
        "p1",
        prompt_tokens=100,
        completion_tokens=10,
    )
    worker_id = await router.select_worker(
        "p1", token_ids=[10, 11, 12, 13, 14], was_paused=True
    )
    assert worker_id is None
    assert kv.calls == []


@pytest.mark.asyncio
async def test_select_worker_falls_through_when_not_paused():
    router, _, kv = make_router()
    await router.before_request("p1")
    await router.after_request(
        "p1",
        prompt_tokens=100,
        completion_tokens=10,
    )
    worker_id = await router.select_worker(
        "p1", token_ids=[10, 11, 12, 13], was_paused=False
    )
    assert worker_id is None
    assert kv.calls == []


@pytest.mark.asyncio
async def test_select_worker_uses_sticky_assignment_when_known():
    router, _, kv = make_router()
    await router.before_request("p1", estimated_prompt_tokens=100)
    router.assign_worker("p1", 3)
    worker_id = await router.select_worker("p1", token_ids=[1, 2, 3], was_paused=False)
    assert worker_id == 3
    assert kv.calls == []


@pytest.mark.asyncio
async def test_pause_acting_then_before_request_blocks_until_resume():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=0.05,
        resume_timeout_seconds=2.0,
    )
    router, _, _ = make_router(config=cfg)

    # Turn 1: place on worker 0 then transition to ACTING.
    await router.before_request("p1")
    router.assign_worker("p1", 0)
    await router.after_request("p1", prompt_tokens=100, completion_tokens=10)

    # Manually pause via the public-but-internal method (the scheduler tick
    # would use the same method given a FPM snapshot; we just bypass the tick
    # for determinism).
    await router._pause_acting("p1")
    assert router._table.programs["p1"].lifecycle == ProgramLifecycle.PAUSED

    # Turn 2 should block until resume signals the waiter.
    waiter = asyncio.create_task(router.before_request("p1"))
    await asyncio.sleep(0.02)
    assert not waiter.done()

    async with router._lock:
        program = router._table.programs["p1"]
        router._resume_program(program, target_worker_id=1)

    decision = await asyncio.wait_for(waiter, timeout=1.0)
    assert decision.was_paused is True
    assert decision.priority_jump == cfg.resume_priority_boost
    assert decision.assigned_worker_hint == 1
    stats = router.stats()
    assert stats["waits"] == 1
    assert stats["wait_seconds"] >= 0.02
    assert stats["wait_gt1s"] == 0
    assert stats["max_wait_seconds"] >= 0.02


@pytest.mark.asyncio
async def test_forced_resume_after_timeout():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        resume_timeout_seconds=0.05,
    )
    router, _, _ = make_router(config=cfg)
    await router.before_request("p1")
    router.assign_worker("p1", 0)
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
        WorkerKey(worker_id=1, dp_rank=0): WorkerCapacity(
            worker_id=1,
            dp_rank=0,
            kv_pool_tokens=1000,
        ),
    }
    router, _, _ = make_router(capacity_workers=workers, config=cfg)
    await router.before_request("existing", estimated_prompt_tokens=950)
    router.assign_worker("existing", 1)

    waiter = asyncio.create_task(
        router.before_request("new", estimated_prompt_tokens=100)
    )
    await asyncio.sleep(0.02)
    assert not waiter.done()
    assert router._table.programs["new"].lifecycle == ProgramLifecycle.PAUSED
    assert "new" in router._table.paused

    async with router._lock:
        router._resume_program(router._table.programs["new"], target_worker_id=1)
    decision = await asyncio.wait_for(waiter, timeout=1.0)
    assert decision.was_paused is True


@pytest.mark.asyncio
async def test_soft_demote_marks_borderline_workers():
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        soft_demote_threshold=0.80,
        pause_threshold=0.95,
    )
    workers = {
        WorkerKey(worker_id=1, dp_rank=0): WorkerCapacity(
            worker_id=1,
            dp_rank=0,
            kv_pool_tokens=1000,
        ),
    }
    router, _, _ = make_router(capacity_workers=workers, config=cfg)
    # Working-set semantics: program token_total + ThunderAgent's per-program
    # buffer drives util, not FPM-current active_decode. Make p1 take ~85% of
    # the 1000-token pool: 750 prompt tokens + 100 buffer.
    await router.before_request("p1")
    router.assign_worker("p1", 1)
    await router.after_request("p1", prompt_tokens=750, completion_tokens=0)
    # Transition back to REASONING for the next turn so soft-demote can fire.
    await router.before_request("p1")
    router.assign_worker("p1", 1)

    # Run scheduler tick directly.
    snapshot = router._capacity.snapshot()
    router._apply_soft_demotes(snapshot)
    program = router._table.programs["p1"]
    assert program.soft_demoted_until > time.time()
    # Now next before_request should fold the negative jump.
    await router.after_request("p1", prompt_tokens=860, completion_tokens=2)
    decision = await router.before_request("p1")
    assert decision.priority_jump == cfg.soft_demote_priority_jump
    assert decision.was_soft_demoted is True


@pytest.mark.asyncio
async def test_scheduling_disabled_passthrough_records_lifecycle():
    """Arm B equivalent: lifecycle still recorded but no pause/admission."""
    cfg = ThunderAgentConfig(
        scheduler_interval_seconds=10.0,
        scheduling_disabled=True,
        pause_threshold=0.50,  # would pause aggressively if scheduling were on
    )
    workers = {
        WorkerKey(worker_id=1, dp_rank=0): WorkerCapacity(
            worker_id=1,
            dp_rank=0,
            kv_pool_tokens=1000,
        ),
    }
    router, _, _ = make_router(capacity_workers=workers, config=cfg)

    # before_request must not block; priority_jump must be zero.
    decision = await router.before_request("p1")
    assert decision.was_paused is False
    assert decision.priority_jump == 0.0
    assert decision.was_soft_demoted is False

    # Lifecycle is still recorded so analysis stays usable.
    program = router._table.programs["p1"]
    assert program.step_count == 1
    assert program.lifecycle == ProgramLifecycle.ACTIVE

    # after_request records tokens but does not transition to paused even if
    # marked_for_pause was set externally (it shouldn't be; defense-in-depth).
    program.marked_for_pause = True
    await router.after_request(
        "p1",
        prompt_tokens=120,
        completion_tokens=30,
    )
    assert program.token_total == 150
    assert program.lifecycle == ProgramLifecycle.ACTIVE  # not PAUSED

    # Soft-demote tick still safe to call; with scheduling_disabled the start()
    # method skips spawning the loop, so we just exercise the no-op pathway
    # here by not calling start() at all.
    assert router._scheduler_task is None


@pytest.mark.asyncio
async def test_utilization_uses_kv_pool_and_working_set():
    """Realistic vLLM TP8 numbers: KV pool is 1M, working set is 1.05M
    (program contexts summing past pool capacity). Threshold trips on the
    program-table sum vs kv_pool, not on any engine-runtime signal.
    """
    # Pin acting_token_weight=1.0 + pause_target=0.95 to isolate the
    # "uses kv pool, not batch budget" assertion from later behavior knobs.
    cfg = ThunderAgentConfig(
        pause_threshold=0.95,
        pause_target=0.95,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        WorkerKey(worker_id=1, dp_rank=0): WorkerCapacity(
            worker_id=1,
            dp_rank=0,
            kv_pool_tokens=1_000_000,
        ),
    }
    router, _, _ = make_router(capacity_workers=workers, config=cfg)
    # Build a working set with large + small programs. Token sum is exactly
    # 950k, but ThunderAgent's 100-token/program buffer pushes it over the
    # threshold. Pausing the small program is sufficient.
    for pid, prompt_tokens in [("big", 940_000), ("small", 10_000)]:
        await router.before_request(pid)
        router.assign_worker(pid, 1)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    snapshot = router._capacity.snapshot()
    await router._pause_until_safe(snapshot)

    # Working set = 950_000 + 2*100 buffer, threshold = 950_000 → over →
    # pause smallest first. If util were not derived from the program-table
    # sum vs the static kv_pool, no
    # pause would fire and this assert would fail.
    assert router._table.programs["small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["big"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_until_safe_pauses_smallest_acting_first():
    cfg = ThunderAgentConfig(
        pause_threshold=0.80,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        WorkerKey(worker_id=1, dp_rank=0): WorkerCapacity(
            worker_id=1,
            dp_rank=0,
            kv_pool_tokens=1000,
        ),
    }
    router, _, _ = make_router(capacity_workers=workers, config=cfg)

    # Two ACTING programs of different sizes on the over-threshold worker.
    # Used = 600 + 100 + 2*100 = 900; pausing small leaves 700 <= target.
    for pid, prompt_tokens in [("big", 600), ("small", 100)]:
        await router.before_request(pid)
        router.assign_worker(pid, 1)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    snapshot = router._capacity.snapshot()
    await router._pause_until_safe(snapshot)

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
        WorkerKey(worker_id=1, dp_rank=0): WorkerCapacity(
            worker_id=1,
            dp_rank=0,
            kv_pool_tokens=1000,
        ),
        WorkerKey(worker_id=2, dp_rank=0): WorkerCapacity(
            worker_id=2,
            dp_rank=0,
            kv_pool_tokens=1000,
        ),
    }
    router, _, _ = make_router(capacity_workers=workers, config=cfg)

    for pid, worker_id, prompt_tokens in [
        ("hot_big", 1, 700),
        ("hot_small", 1, 200),
        ("cold", 2, 700),
    ]:
        await router.before_request(pid)
        router.assign_worker(pid, worker_id)
        await router.after_request(
            pid, prompt_tokens=prompt_tokens, completion_tokens=0
        )

    await router._pause_until_safe(router._capacity.snapshot())

    assert router._table.programs["hot_small"].lifecycle == ProgramLifecycle.PAUSED
    assert router._table.programs["hot_big"].lifecycle == ProgramLifecycle.ACTIVE
    assert router._table.programs["cold"].lifecycle == ProgramLifecycle.ACTIVE


@pytest.mark.asyncio
async def test_pause_drives_util_to_pause_target_not_threshold():
    """Fix 1: each pause cycle drains util DOWN to pause_target, leaving real
    headroom. Without this, B8 stalled at util=0.948 indefinitely because
    pauses stopped firing the moment util slipped under pause_threshold.
    """
    cfg = ThunderAgentConfig(
        pause_threshold=0.95,
        pause_target=0.80,
        acting_token_weight=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        WorkerKey(worker_id=1, dp_rank=0): WorkerCapacity(
            worker_id=1,
            dp_rank=0,
            kv_pool_tokens=1_000_000,
        ),
    }
    router, _, _ = make_router(capacity_workers=workers, config=cfg)
    # Build a working set of 10 equal-size programs summing to 1.0M (100% util).
    for i in range(10):
        pid = f"p{i}"
        await router.before_request(pid)
        router.assign_worker(pid, 1)
        await router.after_request(pid, prompt_tokens=100_000, completion_tokens=0)

    snapshot = router._capacity.snapshot()
    await router._pause_until_safe(snapshot)

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    # After pause cycle, util should be ~0.80, not ~0.95.
    # 10 programs * 100k = 1.0M total. To reach 0.80M we need 2 paused.
    # Old behavior (pause_target ignored) would have paused 1 (1.0M -> 0.9M).
    assert paused >= 2, f"pause cycle stopped too early: paused={paused}, expected >= 2"
    # Verify we didn't over-shoot wildly either.
    assert paused <= 3, f"pause cycle ran too long: paused={paused}, expected <= 3"


@pytest.mark.asyncio
async def test_scheduler_tick_resumes_before_pausing_new_overload():
    """Match upstream TA ordering: resume old paused work, then pause overload.

    This prevents a newly-paused program from being immediately reconsidered
    by greedy resume on the same scheduler tick.
    """
    cfg = ThunderAgentConfig(
        pause_threshold=1.0,
        pause_target=0.80,
        resume_hysteresis=0.0,
        acting_token_weight=1.0,
        acting_decay_tau_seconds=1.0,
        scheduler_interval_seconds=10.0,
    )
    workers = {
        WorkerKey(worker_id=1, dp_rank=0): WorkerCapacity(
            worker_id=1,
            dp_rank=0,
            kv_pool_tokens=1000,
        ),
    }
    router, capacity, _ = make_router(config=cfg)

    # Ten idle ACTING programs occupy 2000 pause-side tokens including
    # ThunderAgent's 100-token/program buffer. The pause pass must drain that
    # to the 0.80 target, which leaves four active and six paused.
    # Capacity is attached after setup so first-turn admission gating does not
    # queue the synthetic programs before this test reaches the scheduler tick.
    for i in range(10):
        pid = f"p{i}"
        await router.before_request(pid)
        router.assign_worker(pid, 1)
        await router.after_request(pid, prompt_tokens=100, completion_tokens=0)
        router._table.programs[pid].acting_since = time.time() - 10.0

    capacity.workers = workers
    await router._scheduler_tick()

    paused = sum(
        1
        for p in router._table.programs.values()
        if p.lifecycle == ProgramLifecycle.PAUSED
    )
    assert paused == 6
    assert router._stat_pauses == 6
    assert router._stat_resumes == 0
