# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThunderAgent program scheduler: native port of upstream TA's algorithm.

Pause-smallest-ACTING-first; BFD restore; exponential decay on the resume
side. v0 reads real token counts from chat-completions ``usage`` instead of
upstream's ``chars / 5`` proxy estimator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from dynamo.thunderagent_router.capacity import WorkerCapacityProvider
from dynamo.thunderagent_router.program_state import (
    Program,
    ProgramLifecycle,
    ProgramStatus,
    ProgramTable,
)

logger = logging.getLogger(__name__)


@dataclass
class PauseDecision:
    program_id: str
    priority_jump: float = 0.0
    waited_seconds: float = 0.0
    was_paused: bool = False
    was_soft_demoted: bool = False
    assigned_worker_hint: Optional[int] = None


@dataclass
class ThunderAgentConfig:
    pause_threshold: float = 0.95
    soft_demote_threshold: float = 0.80
    soft_demote_priority_jump: float = -2.0
    resume_priority_boost: float = 1.0
    resume_timeout_seconds: float = 1800.0
    scheduler_interval_seconds: float = 5.0
    resume_hysteresis: float = 0.10
    pause_target: float = 0.80
    acting_token_weight: float = 1.0
    acting_decay_tau_seconds: float = 1.0
    buffer_per_program: int = 100


class ThunderAgentScheduler:
    def __init__(
        self,
        capacity: WorkerCapacityProvider,
        config: ThunderAgentConfig,
    ) -> None:
        self._capacity = capacity
        self._cfg = config
        self._table = ProgramTable()
        self._lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stat_forced_resumes = 0

    def start(self) -> None:
        if self._scheduler_task is not None:
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            "ThunderAgent scheduler started (interval=%ss, pause=%.2f, soft=%.2f)",
            self._cfg.scheduler_interval_seconds,
            self._cfg.pause_threshold,
            self._cfg.soft_demote_threshold,
        )

    async def stop(self) -> None:
        if self._scheduler_task is None:
            return
        self._scheduler_task.cancel()
        try:
            await self._scheduler_task
        except asyncio.CancelledError:
            pass
        self._scheduler_task = None

    async def before_request(
        self,
        program_id: str,
        estimated_prompt_tokens: int = 0,
    ) -> PauseDecision:
        wait_started = time.monotonic()
        async with self._lock:
            wait_event, was_paused = self._admit_locked(
                program_id, estimated_prompt_tokens
            )

        if wait_event is not None:
            try:
                await asyncio.wait_for(
                    wait_event.wait(), timeout=self._cfg.resume_timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Forced resume for %s after %.1fs",
                    program_id,
                    self._cfg.resume_timeout_seconds,
                )
                async with self._lock:
                    program = self._table.programs.get(program_id)
                    if (
                        program is not None
                        and program.lifecycle == ProgramLifecycle.PAUSED
                    ):
                        worker_id = self._least_loaded_worker_locked(
                            self._capacity.snapshot()
                        )
                        self._resume_program(program, worker_id)
                        self._stat_forced_resumes += 1

        waited = time.monotonic() - wait_started

        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                return PauseDecision(program_id=program_id, waited_seconds=waited)

            priority_jump = self._cfg.resume_priority_boost if was_paused else 0.0
            soft_demoted = program.soft_demoted_until > time.monotonic()
            if soft_demoted:
                priority_jump += self._cfg.soft_demote_priority_jump

            return PauseDecision(
                program_id=program_id,
                priority_jump=priority_jump,
                waited_seconds=waited,
                was_paused=was_paused,
                was_soft_demoted=soft_demoted,
                assigned_worker_hint=program.assigned_worker_id,
            )

    def _admit_locked(
        self,
        program_id: str,
        estimated_prompt_tokens: int,
    ) -> tuple[Optional[asyncio.Event], bool]:
        # Caller holds self._lock.
        was_new = program_id not in self._table.programs
        program = self._table.begin_request(program_id, estimated_prompt_tokens)
        if program.lifecycle == ProgramLifecycle.PAUSED:
            program.waiting = program.waiting or asyncio.Event()
            return program.waiting, True

        if not (was_new and program.assigned_worker_id is None):
            return None, False

        capacities = self._capacity.snapshot()
        if not capacities:
            # Cold start: MDC hasn't published yet. Let the request flow
            # through with no pin; the chunk-loop callback will populate
            # ``assigned_worker_id`` once the engine picks a worker, and
            # subsequent turns get the sticky pin.
            return None, False
        worker_id = self._select_worker_for_new_program_locked(
            capacities, program.token_total
        )
        if worker_id is not None:
            program.assigned_worker_id = worker_id
            return None, False

        # All workers full: queue until the scheduler tick resumes us.
        program.waiting = program.waiting or asyncio.Event()
        program.lifecycle = ProgramLifecycle.PAUSED
        self._table.paused[program_id] = None
        logger.debug(
            "Queued new program %s (tokens=%d)",
            program_id,
            program.token_total,
        )
        return program.waiting, True

    def record_output_tokens(self, program_id: str, delta_tokens: int) -> None:
        # No-await fast path on the streaming chunk loop. Safe because the
        # event loop is single-task; the scheduler tick tolerates a stale
        # token_total by one tick.
        program = self._table.programs.get(program_id)
        if program is not None and program.status == ProgramStatus.REASONING:
            program.token_total += delta_tokens

    async def after_request(
        self,
        program_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        do_pause = False
        async with self._lock:
            program = self._table.end_request(
                program_id, prompt_tokens, completion_tokens
            )
            if program is None:
                return
            if program.marked_for_pause:
                program.marked_for_pause = False
                do_pause = True

        if do_pause:
            await self._pause_acting(program_id)

    async def assign_worker(self, program_id: str, worker_id: int) -> None:
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is not None:
                program.assigned_worker_id = worker_id

    async def _scheduler_loop(self) -> None:
        consecutive_failures = 0
        try:
            while True:
                await asyncio.sleep(self._cfg.scheduler_interval_seconds)
                try:
                    await self._scheduler_tick()
                    consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                    logger.exception("ThunderAgent scheduler tick error")
                    if consecutive_failures >= 10:
                        logger.error(
                            "Scheduler tick failed %d times in a row; halting loop",
                            consecutive_failures,
                        )
                        return
        except asyncio.CancelledError:
            return

    async def _scheduler_tick(self) -> None:
        capacities = self._capacity.snapshot()
        if not capacities:
            return
        # Upstream TA ordering: resume first, then pause -- a program paused
        # this tick can't resume until the next.
        self._apply_soft_demotes(capacities)
        await self._greedy_resume(capacities)
        await self._pause_until_safe(capacities)

    def _program_tokens(self, program: Program, *, decayed: bool = False) -> int:
        if program.status != ProgramStatus.ACTING:
            return program.token_total
        if not decayed:
            return int(program.token_total * self._cfg.acting_token_weight)
        tau = max(self._cfg.acting_decay_tau_seconds, 1e-3)
        idle = (
            max(0.0, time.monotonic() - program.acting_since)
            if program.acting_since > 0
            else 0.0
        )
        return int(program.token_total * (2.0 ** (-(idle / tau))))

    def _active_programs_for_worker(self, worker_id: int) -> list[Program]:
        return [
            p
            for p in self._table.programs.values()
            if p.lifecycle == ProgramLifecycle.ACTIVE
            and p.assigned_worker_id == worker_id
        ]

    def _worker_used(self, worker_id: int, *, decayed: bool = False) -> int:
        programs = self._active_programs_for_worker(worker_id)
        tokens = sum(self._program_tokens(p, decayed=decayed) for p in programs)
        return tokens + len(programs) * self._cfg.buffer_per_program

    def _least_loaded_worker_locked(self, capacities: dict[int, int]) -> Optional[int]:
        if not capacities:
            return None
        return max(
            capacities,
            key=lambda w: capacities[w] - self._worker_used(w, decayed=True),
        )

    def _select_worker_for_new_program_locked(
        self,
        capacities: dict[int, int],
        estimated_tokens: int,
    ) -> Optional[int]:
        # Fairness: new programs queue behind any existing paused program.
        if self._table.paused:
            return None
        buffer = self._cfg.buffer_per_program
        required = estimated_tokens + buffer
        candidates = [
            (w, self._worker_used(w))
            for w, c in capacities.items()
            if c - self._worker_used(w) >= required
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])[0]

    def _apply_soft_demotes(self, capacities: dict[int, int]) -> None:
        soft_until = time.monotonic() + self._cfg.scheduler_interval_seconds * 1.5
        for worker_id, capacity in capacities.items():
            util = self._worker_used(worker_id) / capacity
            if not (
                self._cfg.soft_demote_threshold <= util < self._cfg.pause_threshold
            ):
                continue
            for program in self._active_programs_for_worker(worker_id):
                if (
                    not program.marked_for_pause
                    and program.soft_demoted_until < soft_until
                ):
                    program.soft_demoted_until = soft_until

    async def _pause_until_safe(self, capacities: dict[int, int]) -> None:
        threshold = self._cfg.pause_threshold
        pause_target = min(self._cfg.pause_target, threshold)

        for worker_id, capacity in capacities.items():
            # Hold the lock for the entire per-worker decision so the snapshot
            # of program state used by _smallest_candidates / _worker_used
            # cannot race with concurrent before_request admissions.
            async with self._lock:
                base_used = self._worker_used(worker_id)
                if base_used <= capacity * threshold:
                    continue

                target_limit = capacity * pause_target
                paused_this_tick = 0
                marked_this_tick = 0
                # Bound the inner loop by total program count so a candidate
                # transitioning out from under us can't spin the tick.
                for _ in range(len(self._table.programs) + 1):
                    if self._worker_used(worker_id) <= target_limit:
                        break
                    acting, reasoning = self._smallest_candidates(worker_id)
                    if acting is not None:
                        if self._pause_acting_locked(acting.program_id):
                            paused_this_tick += 1
                        continue
                    if reasoning is not None:
                        if (
                            not reasoning.marked_for_pause
                            and reasoning.lifecycle == ProgramLifecycle.ACTIVE
                            and reasoning.status == ProgramStatus.REASONING
                        ):
                            reasoning.marked_for_pause = True
                            marked_this_tick += 1
                        continue
                    break

                final_used = self._worker_used(worker_id)

            if paused_this_tick or marked_this_tick:
                logger.info(
                    "scheduler.tick worker=%s paused=%d marked=%d util=%.4f -> %.4f",
                    worker_id,
                    paused_this_tick,
                    marked_this_tick,
                    base_used / capacity,
                    final_used / capacity,
                )

    def _smallest_candidates(
        self, worker_id: int
    ) -> tuple[Optional[Program], Optional[Program]]:
        smallest_acting: Optional[Program] = None
        smallest_reasoning: Optional[Program] = None
        for program in self._table.programs.values():
            if program.assigned_worker_id != worker_id:
                continue
            if program.lifecycle != ProgramLifecycle.ACTIVE:
                continue
            if program.marked_for_pause:
                continue
            if program.status == ProgramStatus.ACTING:
                if (
                    smallest_acting is None
                    or program.token_total < smallest_acting.token_total
                ):
                    smallest_acting = program
            elif program.status == ProgramStatus.REASONING:
                if (
                    smallest_reasoning is None
                    or program.token_total < smallest_reasoning.token_total
                ):
                    smallest_reasoning = program
        return smallest_acting, smallest_reasoning

    async def _pause_acting(self, program_id: str) -> bool:
        async with self._lock:
            return self._pause_acting_locked(program_id)

    def _pause_acting_locked(self, program_id: str) -> bool:
        # Caller holds self._lock.
        program = self._table.programs.get(program_id)
        if program is None:
            return False
        if program.lifecycle == ProgramLifecycle.PAUSED:
            return False
        if program.status != ProgramStatus.ACTING:
            return False
        program.lifecycle = ProgramLifecycle.PAUSED
        program.assigned_worker_id = None
        if program.waiting is None:
            program.waiting = asyncio.Event()
        else:
            program.waiting.clear()
        self._table.paused[program_id] = None
        logger.debug("Paused program %s (tokens=%d)", program_id, program.token_total)
        return True

    async def _greedy_resume(self, capacities: dict[int, int]) -> None:
        if not self._table.paused:
            return

        async with self._lock:
            paused_programs = [
                self._table.programs[pid]
                for pid in self._table.paused
                if pid in self._table.programs
            ]
            if not paused_programs:
                return

            def group_key(program: Program) -> int:
                if program.step_count <= 1:
                    return 1
                if program.status == ProgramStatus.REASONING:
                    return 0
                return 2

            paused_programs.sort(key=lambda p: (group_key(p), p.token_total))

            resume_ceiling = max(
                0.0, self._cfg.pause_threshold - self._cfg.resume_hysteresis
            )
            backend_caps = [
                (w, int(c * resume_ceiling) - self._worker_used(w, decayed=False))
                for w, c in capacities.items()
            ]
            backend_caps = [
                (w, r) for w, r in backend_caps if r > self._cfg.buffer_per_program
            ]
            if not backend_caps:
                return

            backend_caps.sort(key=lambda x: -x[1])

            total_capacity = sum(r for _, r in backend_caps)
            resumable_programs: list[Program] = []
            cumulative = 0
            for program in paused_programs:
                required = program.token_total + self._cfg.buffer_per_program
                if cumulative + required <= total_capacity:
                    resumable_programs.append(program)
                    cumulative += required

            if not resumable_programs:
                return

            resumable_programs.sort(key=lambda p: -p.token_total)
            min_required = (
                min(p.token_total for p in resumable_programs)
                + self._cfg.buffer_per_program
            )

            for program in resumable_programs:
                if not backend_caps:
                    break
                worker_id, remaining = backend_caps[0]
                if min_required > remaining:
                    break
                required = program.token_total + self._cfg.buffer_per_program
                if required > remaining:
                    continue
                self._resume_program(program, worker_id)
                updated_remaining = remaining - required
                if updated_remaining > self._cfg.buffer_per_program:
                    backend_caps[0] = (worker_id, updated_remaining)
                    backend_caps.sort(key=lambda x: -x[1])
                else:
                    backend_caps.pop(0)

    def _resume_program(
        self, program: Program, target_worker_id: Optional[int]
    ) -> None:
        # Caller holds self._lock.
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return
        program.lifecycle = ProgramLifecycle.ACTIVE
        program.assigned_worker_id = target_worker_id
        notify = program.waiting
        program.waiting = None
        self._table.paused.pop(program.program_id, None)
        if notify is not None:
            notify.set()
        logger.debug(
            "Resumed program %s -> worker=%s (tokens=%d)",
            program.program_id,
            target_worker_id,
            program.token_total,
        )
