# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ThunderAgent program scheduler.

A native re-implementation of upstream ThunderAgent's algorithm: program
lifecycle (REASONING / ACTING; ACTIVE / PAUSED), pause-smallest-ACTING-first,
BFD restore, exponential decay (``2^(-t/tau)``) applied only on the resume
side. v0's one mechanical change is real-token accounting from
chat-completions ``usage`` instead of upstream's ``chars / 5`` proxy
estimator -- available to us because the router runs in-path.

Status: experimental. The substantial deviations from upstream (blended
cost-function for worker selection, workflow-profile-aware pause, KV
demote/prefetch) are explicitly future work.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from dynamo.thunderagent_router.capacity import FpmCapacityProvider
from dynamo.thunderagent_router.program_state import (
    Program,
    ProgramLifecycle,
    ProgramStatus,
    ProgramTable,
)

logger = logging.getLogger(__name__)


@dataclass
class PauseDecision:
    """Diagnostics returned by :meth:`ThunderAgentScheduler.before_request`."""

    program_id: str
    priority_jump: float = 0.0
    waited_seconds: float = 0.0
    was_paused: bool = False
    was_soft_demoted: bool = False
    assigned_worker_hint: Optional[int] = None


@dataclass
class ThunderAgentConfig:
    """Tunable parameters for the program scheduler."""

    pause_threshold: float = 0.95
    """Pause workers whose utilization >= this fraction of capacity."""

    soft_demote_threshold: float = 0.80
    """Soft-demote priority in [soft, pause); below pause_threshold."""

    soft_demote_priority_jump: float = -2.0
    """priority_jump (seconds) for soft-demoted programs. Negative = later."""

    resume_priority_boost: float = 1.0
    """priority_jump (seconds) added on resume from hard pause."""

    resume_timeout_seconds: float = 1800.0
    """Forced-resume after this many seconds in PAUSED."""

    scheduler_interval_seconds: float = 5.0
    """Period of the background scheduler tick."""

    scheduling_enabled: bool = True
    """When False, record lifecycle but skip pause/resume/soft-demote."""

    resume_hysteresis: float = 0.10
    """Util drop below pause_threshold required before any resume."""

    pause_target: float = 0.80
    """Setpoint pause cycles drain util down to (must be <= pause_threshold)."""

    acting_token_weight: float = 1.0
    """Weight on token_total for ACTING programs in the pause-side working set."""

    acting_decay_tau_seconds: float = 1.0
    """Tau (s) for ``2^(-idle/tau)`` decay of ACTING tokens on the resume side."""

    buffer_per_program: int = 100
    """Headroom reserved per program when BFD-packing during resume."""

    cache_aware_admission: bool = False
    """When True, the handler precomputes per-worker prefix-cache overlap via
    KvRouter.get_potential_loads(token_ids) and passes it to before_request.
    New-program admission prefers workers where the prompt is already cached,
    and treats cached tokens as a deduction from required capacity. Off by
    default so we can A/B this against the no-cache baseline."""

    shared_tokens_enabled: bool = False
    """When True, maintain a per-worker EMA of prefix-cache hit rate from
    chat-completions ``usage.prompt_tokens_details.cached_tokens`` and
    deduct ``cache_hit_rate * worker_used`` from the pause-side denominator.
    Mirrors upstream TA's ``shared_tokens`` which it polls from vLLM
    ``/metrics``. Off by default for A/B testing."""

    cache_hit_rate_alpha: float = 0.2
    """EMA smoothing factor for shared_tokens cache hit rate."""


class ThunderAgentScheduler:
    """Outer-loop program scheduler.

    The handler in ``__main__`` calls :meth:`before_request` -> dispatch ->
    :meth:`after_request` for each LLM turn. The scheduler is purely
    coordination state; it does not own the request stream.
    """

    def __init__(
        self,
        capacity: FpmCapacityProvider,
        config: ThunderAgentConfig,
    ) -> None:
        self._capacity = capacity
        self._cfg = config
        self._table = ProgramTable()
        self._lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stat_forced_resumes = 0
        # Per-worker EMA of (cached_tokens / prompt_tokens), updated from
        # response usage when shared_tokens_enabled.
        self._worker_cache_hit_rate: dict[int, float] = {}

    def start(self) -> None:
        if self._scheduler_task is not None:
            return
        if not self._cfg.scheduling_enabled:
            logger.info("ThunderAgent scheduling disabled: passthrough mode")
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
        worker_cached_tokens: Optional[dict[int, int]] = None,
    ) -> PauseDecision:
        """Admission gate. Blocks if the program is currently PAUSED.

        ``worker_cached_tokens`` (when cache_aware_admission is on) maps
        worker_id -> tokens of this prompt already cached on that worker.
        Used by new-program admission to prefer prefix-local workers.
        """
        if not self._cfg.scheduling_enabled:
            async with self._lock:
                self._table.begin_request(program_id, estimated_prompt_tokens)
            return PauseDecision(program_id=program_id)

        wait_started = time.monotonic()
        async with self._lock:
            wait_event, was_paused = self._admit_locked(
                program_id, estimated_prompt_tokens, worker_cached_tokens
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
        worker_cached_tokens: Optional[dict[int, int]] = None,
    ) -> tuple[Optional[asyncio.Event], bool]:
        """Resolve the admission state for a turn. Returns (wait_event, was_paused).

        Caller already holds ``self._lock``.
        """
        was_new = program_id not in self._table.programs
        program = self._table.begin_request(program_id, estimated_prompt_tokens)
        if program.lifecycle == ProgramLifecycle.PAUSED:
            program.waiting = program.waiting or asyncio.Event()
            return program.waiting, True

        if not (was_new and program.assigned_worker_id is None):
            return None, False

        # New program: assign a worker that has room, otherwise queue.
        capacities = self._capacity.snapshot()
        if not capacities:
            return None, False
        worker_id = self._select_worker_for_new_program_locked(
            capacities, program.token_total, worker_cached_tokens
        )
        if worker_id is not None:
            program.assigned_worker_id = worker_id
            return None, False
        program.waiting = program.waiting or asyncio.Event()
        program.lifecycle = ProgramLifecycle.PAUSED
        self._table.paused[program_id] = None
        logger.info(
            "Queued new program %s before first request (tokens=%d)",
            program_id,
            program.token_total,
        )
        return program.waiting, True

    def record_output_tokens(self, program_id: str, delta_tokens: int) -> None:
        """Streaming token-progress accounting. ``after_request`` overwrites
        with authoritative usage at end of turn.

        Lock-free on the hot path: GIL covers the int add, and concurrent
        scheduler reads tolerate a stale token_total by one tick.
        """
        program = self._table.programs.get(program_id)
        if program is not None and program.status == ProgramStatus.REASONING:
            program.token_total += delta_tokens

    async def after_request(
        self,
        program_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Transition program -> ACTING and record real token accounting."""
        do_pause = False
        async with self._lock:
            program = self._table.end_request(
                program_id, prompt_tokens, completion_tokens
            )
            if program is None:
                return
            if self._cfg.scheduling_enabled and program.marked_for_pause:
                program.marked_for_pause = False
                do_pause = True

        if do_pause:
            await self._pause_acting(program_id)

    def assign_worker(self, program_id: str, worker_id: int) -> None:
        """Record the dispatched worker. Lock-free; Program mutation is
        single-threaded under asyncio."""
        program = self._table.programs.get(program_id)
        if program is not None:
            program.assigned_worker_id = worker_id

    def update_cache_hit_rate(
        self, worker_id: int, cached_tokens: int, prompt_tokens: int
    ) -> None:
        """EMA update of per-worker prefix-cache hit rate from response usage.

        Only meaningful when shared_tokens_enabled; the deduction is applied
        in _worker_used. Lock-free per the same GIL contract as record_output_tokens.
        """
        if not self._cfg.shared_tokens_enabled or prompt_tokens <= 0:
            return
        sample = max(0.0, min(1.0, cached_tokens / prompt_tokens))
        prev = self._worker_cache_hit_rate.get(worker_id, 0.0)
        alpha = self._cfg.cache_hit_rate_alpha
        self._worker_cache_hit_rate[worker_id] = alpha * sample + (1 - alpha) * prev

    async def _scheduler_loop(self) -> None:
        consecutive_failures = 0
        try:
            while True:
                await asyncio.sleep(self._cfg.scheduler_interval_seconds)
                try:
                    await self._scheduler_tick()
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    logger.exception("ThunderAgent scheduler tick error: %s", exc)
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
        # Upstream TA ordering: resume first, then pause. A program paused
        # this tick can't resume until the next one.
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
        used = tokens + len(programs) * self._cfg.buffer_per_program
        if self._cfg.shared_tokens_enabled:
            # Upstream-style shared_tokens deduction: a fraction of "active"
            # tokens is actually shared with the prefix cache and not real
            # capacity pressure.
            hit_rate = self._worker_cache_hit_rate.get(worker_id, 0.0)
            shared = int(tokens * hit_rate)
            used = max(0, used - shared)
        return used

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
        worker_cached_tokens: Optional[dict[int, int]] = None,
    ) -> Optional[int]:
        # Upstream TA queues new programs behind any existing paused program for
        # fairness; don't let a fresh trajectory jump the global waiting queue.
        if self._table.paused:
            return None
        buffer = self._cfg.buffer_per_program
        cache_aware = worker_cached_tokens is not None
        candidates: list[tuple[int, int]] = []
        for w, c in capacities.items():
            cached = worker_cached_tokens.get(w, 0) if cache_aware else 0
            required = max(buffer, estimated_tokens - cached + buffer)
            used = self._worker_used(w)
            if c - used >= required:
                # Effective load = used minus prefix cached for this prompt.
                # Workers with the prefix already resident look less loaded,
                # so we naturally prefer them.
                candidates.append((w, used - cached))
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
        """Hard-pause smallest ACTING + mark smallest REASONING per worker.

        ACTING programs are paused immediately. REASONING programs are marked;
        they remain in the capacity math until they finish their current turn,
        matching upstream TA's request-boundary enforcement.
        """
        threshold = self._cfg.pause_threshold
        pause_target = min(self._cfg.pause_target, threshold)

        for worker_id, capacity in capacities.items():
            base_used = self._worker_used(worker_id)
            if base_used <= capacity * threshold:
                continue

            target_limit = capacity * pause_target
            paused_this_tick = 0
            marked_this_tick = 0
            while self._worker_used(worker_id) > target_limit:
                acting, reasoning = self._smallest_candidates(worker_id)
                if acting is not None:
                    await self._pause_acting(acting.program_id)
                    paused_this_tick += 1
                    continue
                if reasoning is not None:
                    async with self._lock:
                        target = self._table.programs.get(reasoning.program_id)
                        if (
                            target is not None
                            and not target.marked_for_pause
                            and target.lifecycle == ProgramLifecycle.ACTIVE
                            and target.status == ProgramStatus.REASONING
                        ):
                            target.marked_for_pause = True
                            marked_this_tick += 1
                    continue
                break

            if paused_this_tick or marked_this_tick:
                logger.info(
                    "tick worker=%s paused=%d marked=%d util=%.4f -> %.4f",
                    worker_id,
                    paused_this_tick,
                    marked_this_tick,
                    base_used / capacity,
                    self._worker_used(worker_id) / capacity,
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

    async def _pause_acting(self, program_id: str) -> None:
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                return
            if program.lifecycle == ProgramLifecycle.PAUSED:
                return
            if program.status != ProgramStatus.ACTING:
                return
            program.lifecycle = ProgramLifecycle.PAUSED
            program.assigned_worker_id = None
            if program.waiting is None:
                program.waiting = asyncio.Event()
            else:
                program.waiting.clear()
            self._table.paused[program_id] = None
            logger.info(
                "Paused program %s (tokens=%d)", program_id, program.token_total
            )

    async def _greedy_resume(self, capacities: dict[int, int]) -> None:
        """Resume paused programs with ThunderAgent-style BFD packing.

        Resume gate uses non-decayed used-tokens so paused programs only
        re-admit when other ACTING programs have actually released capacity.
        Priority order: REASONING (continuation) -> NEW (step==1) -> ACTING.
        """
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
        """Caller already holds ``self._lock``."""
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return
        program.lifecycle = ProgramLifecycle.ACTIVE
        program.assigned_worker_id = target_worker_id
        notify = program.waiting
        program.waiting = None
        self._table.paused.pop(program.program_id, None)
        if notify is not None:
            notify.set()
        logger.info(
            "Resumed program %s -> worker=%s (tokens=%d)",
            program.program_id,
            target_worker_id,
            program.token_total,
        )
