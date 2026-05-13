# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KvThunderAgentRouter -- ThunderAgent program scheduler inside a Dynamo router.

A native re-implementation of upstream ThunderAgent's algorithm:
program lifecycle (REASONING / ACTING; ACTIVE / PAUSED),
pause-smallest-ACTING-first, BFD restore, exponential decay (2^(-t/tau))
applied only on the resume side. v0's one mechanical change is
real-token accounting from chat-completions ``usage`` instead of
upstream's ``chars / 5`` proxy estimator -- available to us because the
router runs in-path.

Status: experimental. The substantial deviations from upstream
(blended cost-function for worker selection, workflow-profile-aware
pause, KV demote/prefetch) are explicitly future work.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from dynamo.llm import KvRouter
from dynamo.thunderagent_router.capacity import FpmCapacityProvider
from dynamo.thunderagent_router.program_state import (
    Program,
    ProgramLifecycle,
    ProgramStatus,
    ProgramTable,
)

logger = logging.getLogger(__name__)


# ThunderAgent's BUFFER_PER_PROGRAM equivalent. Used to leave headroom when
# packing programs onto a worker via BFD-style resume.
DEFAULT_BUFFER_PER_PROGRAM = 100


@dataclass
class PauseDecision:
    """Diagnostics returned by :meth:`KvThunderAgentRouter.before_request`.

    ``priority_jump`` is summed with the client-supplied ``nvext.agent_hints``
    priority before the request enters the KV router queue. Negative values
    soft-demote borderline candidates; positive values boost resumes.
    """

    program_id: str
    priority_jump: float = 0.0
    waited_seconds: float = 0.0
    was_paused: bool = False
    was_soft_demoted: bool = False
    assigned_worker_hint: Optional[int] = None


@dataclass
class ThunderAgentConfig:
    """Tunable parameters for the program scheduler.

    Defaults pick conservative values; tune via CLI flags / env vars.
    """

    pause_threshold: float = 0.95
    """Hard-pause kick-in: pause programs on workers whose utilization >= this."""

    soft_demote_threshold: float = 0.80
    """Soft-pause: demote priority for programs whose worker utilization is
    above this but below ``pause_threshold``. Borderline cases bias toward
    losing queue position rather than being pulled out entirely."""

    soft_demote_priority_jump: float = -2.0
    """Negative ``priority_jump`` (seconds) applied to soft-demoted programs.
    The router queue treats higher ``priority_jump`` as earlier-arrival, so a
    negative value pushes the request later."""

    resume_priority_boost: float = 1.0
    """Boost (seconds) applied to a request that was just resumed from hard
    pause. Mirrors ThunderAgent's REASONING > NEW > ACTING priority."""

    resume_timeout_seconds: float = 1800.0
    """Forced-resume after this many seconds in PAUSED. Mirrors ThunderAgent
    public-repo ``_wait_for_resume`` ceiling."""

    scheduler_interval_seconds: float = 5.0
    """Period of the background scheduler tick (capacity check, pause/resume).

    ThunderAgent's default is 5s; we keep the same so behavior matches the
    reference proxy when v0 ablations replicate prior numbers."""

    scheduling_disabled: bool = False
    """When true, the router records lifecycle state (begin/end_request,
    last_prefix capture) but does NOT pause, resume, or soft-demote. The
    scheduler tick is also skipped. Used as the 'TR off' arm to isolate
    scheduling value from program-aware passthrough."""

    resume_hysteresis: float = 0.10
    """How far below ``pause_threshold`` the working set must drop before
    we resume any paused program. 0.0 means immediate resume the moment
    a slot opens (oscillates); 0.10 means resume only when working_set is
    <= (pause_threshold - 0.10) * pool. Empirically, with very fast
    request cadence (mini-SWE @ 128 workers), 0.0 thrashes; 0.10 stabilises."""

    pause_target: float = 0.80
    """Setpoint that ``_pause_until_safe`` drives util DOWN to during a
    pause cycle. Trigger fires at ``pause_threshold`` (e.g. 0.95) but we
    keep pausing programs until projected util reaches ``pause_target``
    (e.g. 0.80). Without this, B8 stalled at util=0.948 indefinitely:
    pauses stopped firing the moment util slipped under pause_threshold,
    leaving the engine permanently saturated at the threshold instead of
    actually below it. Upstream TA achieves the same by pausing on
    overflow (remaining_capacity < 0) and pausing many in one shot."""

    acting_token_weight: float = 1.0
    """Flat multiplier applied to ``token_total`` for programs in
    ProgramStatus.ACTING when summing the **pause-decision** working set.
    Mirrors upstream TA's ``tool_coefficient``: default 1.0 = "treat acting
    programs as if their KV is still resident". Lower values admit more
    programs before the pause threshold trips; raise above 1.0 to be even
    more conservative."""

    acting_decay_tau_seconds: float = 1.0
    """Time constant for the exponential decay applied to ACTING programs
    in the **resume-decision** capacity. Weight is
    ``2 ** (-(now - acting_since) / tau)``. With tau=1.0s, a program idle
    10s contributes 2^-10 ≈ 0.001 of its tokens; idle 60s contributes
    essentially zero. Mirrors upstream TA's ``remaining_capacity_with_decay``:
    optimistic on the resume side without ever deleting a program -- a
    zombie that returns 20 minutes later still has its token_total intact."""

    buffer_per_program: int = DEFAULT_BUFFER_PER_PROGRAM
    """Headroom reserved per program when BFD-packing during resume."""

    kv_aware_resume_enabled: bool = False
    """Experimental override on resume worker selection. When True,
    ``select_worker`` uses ``KvRouter.best_worker(last_prefix)`` instead of
    BFD's load assignment. Default False; kept for ablation
    reproducibility."""


class KvThunderAgentRouter:
    """Outer-loop program scheduler. Wraps a :class:`KvRouter` instance.

    The handler in ``__main__`` is responsible for calling
    :meth:`before_request` -> :meth:`select_worker` -> ``kv_router.generate``
    -> :meth:`after_request` for each LLM turn. The router itself is purely
    coordination state; it does not own the request stream.
    """

    def __init__(
        self,
        kv_router: KvRouter,
        capacity: FpmCapacityProvider,
        config: ThunderAgentConfig,
    ) -> None:
        self._kv_router = kv_router
        self._capacity = capacity
        self._cfg = config
        self._table = ProgramTable()
        # Coarse-grained lock for pause/resume bookkeeping. Inner state can
        # afford to serialize across tens-of-Hz scheduler ticks; per-request
        # ``before_request`` only acquires it briefly.
        self._lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None

        self._stat_pauses = 0
        self._stat_resumes = 0
        self._stat_soft_demotes = 0
        self._stat_forced_resumes = 0
        self._stat_waits = 0
        self._stat_wait_seconds = 0.0
        self._stat_wait_gt1s = 0
        self._stat_max_wait_seconds = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._scheduler_task is not None:
            return
        if self._cfg.scheduling_disabled:
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

    def stats(self) -> dict:
        counts = self._table.counts()
        return {
            **counts,
            "total_programs": len(self._table.programs),
            "pauses": self._stat_pauses,
            "resumes": self._stat_resumes,
            "soft_demotes": self._stat_soft_demotes,
            "forced_resumes": self._stat_forced_resumes,
            "waits": self._stat_waits,
            "wait_seconds": self._stat_wait_seconds,
            "wait_gt1s": self._stat_wait_gt1s,
            "max_wait_seconds": self._stat_max_wait_seconds,
        }

    # ------------------------------------------------------------------ #
    # Per-request hooks
    # ------------------------------------------------------------------ #

    async def before_request(
        self, program_id: str, estimated_prompt_tokens: int = 0
    ) -> PauseDecision:
        """Admission gate. Blocks if the program is currently PAUSED.

        Returns the priority adjustment the caller should fold into the request
        before forwarding to the KV router queue.
        """
        if self._cfg.scheduling_disabled:
            # Passthrough: still record lifecycle so KV-aware resume + analysis
            # have program_id-aware data, but skip the admission gate and
            # priority adjustment.
            async with self._lock:
                self._table.begin_request(program_id, estimated_prompt_tokens)
            return PauseDecision(program_id=program_id)

        wait_started = time.monotonic()
        was_paused: bool
        wait_event: Optional[asyncio.Event]
        async with self._lock:
            was_new = program_id not in self._table.programs
            program = self._table.begin_request(program_id, estimated_prompt_tokens)
            if program.lifecycle == ProgramLifecycle.PAUSED:
                wait_event = program.waiting or asyncio.Event()
                program.waiting = wait_event
                was_paused = True
            elif was_new and program.assigned_worker_id is None:
                snapshot = self._capacity.snapshot()
                # If discovery/FPM has not populated a capacity row yet, fall
                # through and let the KvRouter pick. Once capacity is visible,
                # mirror upstream TA: new programs either claim a backend with
                # enough remaining capacity or enter the paused queue before
                # their first request reaches the engine.
                if self._worker_capacities(snapshot):
                    worker_id = self._select_worker_for_new_program_locked(
                        snapshot, program.token_total
                    )
                    if worker_id is None:
                        wait_event = program.waiting or asyncio.Event()
                        program.waiting = wait_event
                        program.lifecycle = ProgramLifecycle.PAUSED
                        program.origin_worker_id = None
                        self._table.paused[program_id] = True
                        was_paused = True
                        logger.info(
                            "Queued new program %s before first request (tokens=%d)",
                            program_id,
                            program.token_total,
                        )
                    else:
                        program.assigned_worker_id = worker_id
                        wait_event = None
                        was_paused = False
                else:
                    wait_event = None
                    was_paused = False
            else:
                wait_event = None
                was_paused = False

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
                        snapshot = self._capacity.snapshot()
                        worker_id = self._least_loaded_worker_locked(snapshot)
                        self._resume_program(program, worker_id)
                        self._stat_forced_resumes += 1

        waited = time.monotonic() - wait_started
        if wait_event is not None:
            self._stat_waits += 1
            self._stat_wait_seconds += waited
            if waited > 1.0:
                self._stat_wait_gt1s += 1
            self._stat_max_wait_seconds = max(self._stat_max_wait_seconds, waited)

        # Compute priority contribution + worker hint after admission resolves.
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                # Released out from under us.
                return PauseDecision(program_id=program_id, waited_seconds=waited)

            priority_jump = 0.0
            soft_demoted = False
            if was_paused:
                priority_jump = self._cfg.resume_priority_boost
            now = time.time()
            if program.soft_demoted_until > now:
                priority_jump += self._cfg.soft_demote_priority_jump
                soft_demoted = True

            return PauseDecision(
                program_id=program_id,
                priority_jump=priority_jump,
                waited_seconds=waited,
                was_paused=was_paused,
                was_soft_demoted=soft_demoted,
                assigned_worker_hint=program.assigned_worker_id,
            )

    async def select_worker(
        self,
        program_id: str,
        token_ids: list[int],
        was_paused: bool,
    ) -> Optional[int]:
        """Resolve a worker for this turn.

        Item 2 (KV-aware resume): if the program was paused, prefer the worker
        with the warmest KV for the program's last-turn prefix. Otherwise fall
        through to the KV router's normal best_worker for the current
        ``token_ids`` (which is what the frontend would have done anyway).

        Returns ``None`` to indicate "let the KV router pick" -- the handler
        forwards the request without a pinned ``worker_id``.
        """
        async with self._lock:
            program = self._table.programs.get(program_id)
            prefix = program.last_prefix_token_ids if program else None
            assigned_worker_id = program.assigned_worker_id if program else None

        if was_paused and prefix and self._cfg.kv_aware_resume_enabled:
            try:
                worker_id, _dp_rank, _overlap = await self._kv_router.best_worker(
                    prefix
                )
                logger.debug(
                    "ThunderAgent KV-aware resume: program=%s prefix_len=%d -> worker=%d",
                    program_id,
                    len(prefix),
                    worker_id,
                )
                return worker_id
            except Exception as exc:
                logger.warning(
                    "best_worker failed during KV-aware resume for %s: %s; falling back",
                    program_id,
                    exc,
                )

        # ThunderAgent keeps a program sticky to its assigned backend until it
        # is paused/resumed. Pin to the assigned worker when known; otherwise
        # let the KvRouter pick from current token_ids.
        if assigned_worker_id is not None:
            return assigned_worker_id
        return None

    def record_output_tokens(self, program_id: str, delta_tokens: int) -> None:
        """Mirror upstream streaming token progress accounting.

        During a long REASONING turn, the scheduler should see the output KV
        growth before the terminal usage chunk arrives. ``after_request``
        overwrites this with authoritative usage at the end of the turn.
        """
        if delta_tokens <= 0:
            return
        program = self._table.programs.get(program_id)
        if program is not None and program.status == ProgramStatus.REASONING:
            program.token_total += delta_tokens

    async def after_request(
        self,
        program_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        last_prefix_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Transition program -> ACTING, record real token accounting (item 1)
        and the prefix used for KV-aware resume placement (item 2)."""
        do_pause = False
        async with self._lock:
            program = self._table.end_request(
                program_id,
                prompt_tokens,
                completion_tokens,
                last_prefix_token_ids,
            )
            if program is None:
                return
            if not self._cfg.scheduling_disabled and program.marked_for_pause:
                program.marked_for_pause = False
                do_pause = True

        if do_pause:
            await self._pause_acting(program_id)

    async def release(self, program_id: str) -> None:
        async with self._lock:
            self._table.release(program_id)

    def assign_worker(self, program_id: str, worker_id: int) -> None:
        """Record the worker the request actually got dispatched to.
        Called from the handler after :meth:`select_worker` resolves."""
        # This runs on the request-completion path so we keep it lock-free
        # via direct dict access; ``Program`` mutation is single-threaded
        # within asyncio.
        program = self._table.programs.get(program_id)
        if program is not None:
            program.assigned_worker_id = worker_id

    # ------------------------------------------------------------------ #
    # Scheduler tick
    # ------------------------------------------------------------------ #

    async def _scheduler_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._cfg.scheduler_interval_seconds)
                try:
                    await self._scheduler_tick()
                except Exception as exc:
                    logger.exception("ThunderAgent scheduler tick error: %s", exc)
        except asyncio.CancelledError:
            return

    async def _scheduler_tick(self) -> None:
        snapshot = self._capacity.snapshot()
        if not snapshot:
            return
        # Step 1: surface soft demotes for borderline workers.
        # Step 2: greedy resume programs that were already paused before this
        #         tick, using the optimistic decay-side capacity.
        # Step 3: hard-pause / mark newly overloaded workers, using the
        #         conservative pause-side working set.
        #
        # This matches upstream ThunderAgent's scheduler ordering:
        # _greedy_resume() runs before _pause_until_safe(). A program paused in
        # this tick is therefore not eligible to resume until the next scheduler
        # tick, without adding a separate minimum-dwell constant.
        #
        # Note: we deliberately do NOT have a TTL/GC step. Upstream TA solves
        # the "zombie programs inflate capacity" problem via the
        # remaining_capacity_with_decay() decay function: idle programs
        # naturally contribute zero to the resume decision after ~10s but
        # keep their ``token_total`` history for when they return. Hard GC
        # destroys that history and causes pathological recovery (see B9).
        self._apply_soft_demotes(snapshot)
        await self._greedy_resume(snapshot)
        await self._pause_until_safe(snapshot)

    def _program_tokens(self, program: Program, *, decayed: bool = False) -> int:
        if program.status == ProgramStatus.ACTING:
            if decayed:
                tau = max(self._cfg.acting_decay_tau_seconds, 1e-3)
                idle = (
                    max(0.0, time.time() - program.acting_since)
                    if program.acting_since > 0
                    else 0.0
                )
                return int(program.token_total * (2.0 ** (-(idle / tau))))
            return int(program.token_total * self._cfg.acting_token_weight)
        return program.token_total

    @staticmethod
    def _worker_capacities(snapshot: dict) -> dict[int, int]:
        capacities: dict[int, int] = {}
        for cap in snapshot.values():
            capacity = cap.capacity_tokens
            if capacity > 0:
                capacities[cap.worker_id] = capacities.get(cap.worker_id, 0) + capacity
        return capacities

    def _active_programs_for_worker(self, worker_id: int) -> list[Program]:
        return [
            program
            for program in self._table.programs.values()
            if program.lifecycle == ProgramLifecycle.ACTIVE
            and program.assigned_worker_id == worker_id
        ]

    def _worker_used(self, worker_id: int, *, decayed: bool = False) -> int:
        programs = self._active_programs_for_worker(worker_id)
        tokens = sum(
            self._program_tokens(program, decayed=decayed) for program in programs
        )
        return tokens + len(programs) * self._cfg.buffer_per_program

    def _worker_remaining(
        self, worker_id: int, capacity_tokens: int, *, decayed: bool = False
    ) -> int:
        return capacity_tokens - self._worker_used(worker_id, decayed=decayed)

    def _least_loaded_worker_locked(self, snapshot: dict) -> Optional[int]:
        capacities = self._worker_capacities(snapshot)
        if not capacities:
            return None
        return max(
            capacities,
            key=lambda worker_id: self._worker_remaining(
                worker_id, capacities[worker_id], decayed=True
            ),
        )

    def _select_worker_for_new_program_locked(
        self, snapshot: dict, estimated_tokens: int
    ) -> Optional[int]:
        # Upstream TA queues new programs behind any existing paused program for
        # fairness; don't let a fresh trajectory jump the global waiting queue.
        if self._table.paused:
            return None
        capacities = self._worker_capacities(snapshot)
        if not capacities:
            return None
        required = estimated_tokens + self._cfg.buffer_per_program
        candidates: list[tuple[int, int]] = []
        for worker_id, capacity in capacities.items():
            remaining = self._worker_remaining(worker_id, capacity, decayed=False)
            if remaining >= required:
                candidates.append((worker_id, self._worker_used(worker_id)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1])
        return candidates[0][0]

    def _apply_soft_demotes(self, snapshot: dict) -> None:
        """Borderline soft-demote per worker.

        Upstream capacity is per backend. Keep the soft demote band scoped to
        the worker whose program set is actually near capacity instead of using
        a global pool that can hide a hot worker behind cold ones.
        """
        capacities = self._worker_capacities(snapshot)
        if not capacities:
            return
        soft_until = time.time() + self._cfg.scheduler_interval_seconds * 1.5
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
                    self._stat_soft_demotes += 1

    async def _pause_until_safe(self, snapshot: dict) -> None:
        """Hard-pause smallest ACTING + mark smallest REASONING per worker.

        Working set: Σ program.token_total across active programs assigned to
        the worker, plus ThunderAgent's per-program buffer.
        Worker capacity: kv_cache_block_size × total_kv_blocks.

        ACTING programs are paused immediately. REASONING programs are marked;
        they remain in the capacity math until they finish their current turn,
        matching upstream TA's request-boundary enforcement.
        """
        capacities = self._worker_capacities(snapshot)
        if not capacities:
            return
        threshold = self._cfg.pause_threshold
        pause_target = min(self._cfg.pause_target, threshold)

        for worker_id, capacity in capacities.items():
            base_used = self._worker_used(worker_id)
            util = base_used / capacity if capacity > 0 else 0.0
            active_count = len(self._active_programs_for_worker(worker_id))
            logger.info(
                "tick: worker=%s used=%d pool=%d util=%.4f thresh=%.2f target=%.2f over=%s active_progs=%d",
                worker_id,
                base_used,
                capacity,
                util,
                threshold,
                pause_target,
                util > threshold,
                active_count,
            )

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
                new_used = self._worker_used(worker_id)
                logger.info(
                    "scheduler tick worker=%s paused=%d marked=%d (util %.4f -> ~%.4f, target=%.2f)",
                    worker_id,
                    paused_this_tick,
                    marked_this_tick,
                    util,
                    new_used / capacity if capacity > 0 else 0.0,
                    pause_target,
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
            program.origin_worker_id = program.assigned_worker_id
            program.assigned_worker_id = None
            if program.waiting is None:
                program.waiting = asyncio.Event()
            else:
                program.waiting.clear()
            self._table.paused[program_id] = True
            self._stat_pauses += 1
            logger.info(
                "Paused program %s (tokens=%d, origin_worker=%s)",
                program_id,
                program.token_total,
                program.origin_worker_id,
            )

    async def _greedy_resume(self, snapshot: dict) -> None:
        """Resume paused programs with ThunderAgent-style BFD packing.

        Remaining capacity is computed per worker. ACTING programs use the
        exponential decay term on the resume side, mirroring upstream TA's
        ``remaining_capacity_with_decay``.

        Why decay (and not GC + flat weight): a program idle for tool
        execution is not pinning KV anywhere near its ``token_total`` --
        the engine pages most of it out. Counting it at full weight on
        the resume side stalls forever (B8/B9: "stuck at util=0.95");
        deleting it on a TTL destroys its history and causes pathological
        recovery (B9 collapse). Continuous decay never deletes, just
        stops counting after ~10s idle.

        Priority order matches ThunderAgent: REASONING (continuation) -> NEW
        (step==1) -> ACTING. Within group, ascending by token total.
        """
        if not self._table.paused:
            return

        capacities = self._worker_capacities(snapshot)
        if not capacities:
            return

        async with self._lock:
            paused_programs: list[Program] = [
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

            # Match upstream TA default (use_acting_token_decay=False): the
            # resume gate uses NON-decayed used-tokens so paused programs only
            # re-admit when other ACTING programs have actually released
            # capacity. Decayed-on-resume re-admitted within one tick because
            # acting weight collapses to ~0 after a few seconds, producing
            # 6000x-shorter pause dwell than upstream (B13: pause_s_sum=1.9s
            # vs A_v2 8800s).
            backend_caps = [
                (
                    worker_id,
                    int(
                        capacity
                        * max(
                            0.0, self._cfg.pause_threshold - self._cfg.resume_hysteresis
                        )
                    )
                    - self._worker_used(worker_id, decayed=False),
                )
                for worker_id, capacity in capacities.items()
            ]
            backend_caps = [
                (worker_id, remaining)
                for worker_id, remaining in backend_caps
                if remaining > self._cfg.buffer_per_program
            ]
            if not backend_caps:
                return

            backend_caps.sort(key=lambda x: -x[1])

            # Upstream selects the largest priority-eligible set that fits in
            # aggregate remaining capacity, then places largest-first on the
            # worker with the most remaining capacity.
            total_capacity = sum(remaining for _worker_id, remaining in backend_caps)
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
                min(program.token_total for program in resumable_programs)
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
        """Note: caller already holds ``self._lock``."""
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return
        program.lifecycle = ProgramLifecycle.ACTIVE
        program.assigned_worker_id = target_worker_id
        program.origin_worker_id = None
        notify = program.waiting
        program.waiting = None
        self._table.paused.pop(program.program_id, None)
        self._stat_resumes += 1
        if notify is not None:
            notify.set()
        logger.info(
            "Resumed program %s -> worker=%s (tokens=%d)",
            program.program_id,
            target_worker_id,
            program.token_total,
        )
