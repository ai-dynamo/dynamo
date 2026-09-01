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
from typing import Any, Optional

from dynamo.thunderagent_router.capacity import WorkerCapacityProvider
from dynamo.thunderagent_router.program_state import (
    Program,
    ProgramLifecycle,
    ProgramStatus,
    ProgramTable,
    ReplicaKey,
    RequestSnapshot,
)

logger = logging.getLogger(__name__)


@dataclass
class PauseDecision:
    program_id: str
    priority_jump: float = 0.0
    waited_seconds: float = 0.0
    was_paused: bool = False
    was_soft_demoted: bool = False
    # Keep the worker-only field for callers of the original scheduler API;
    # new routing code consumes the complete replica hint below.
    assigned_worker_hint: Optional[int] = None
    assigned_dp_rank: Optional[int] = None
    assigned_replica_hint: Optional[ReplicaKey] = None

    def __post_init__(self) -> None:
        if self.assigned_replica_hint is None:
            if (
                self.assigned_worker_hint is not None
                and self.assigned_dp_rank is not None
            ):
                self.assigned_replica_hint = (
                    self.assigned_worker_hint,
                    self.assigned_dp_rank,
                )
            return
        worker_id, dp_rank = self.assigned_replica_hint
        if self.assigned_worker_hint is None:
            self.assigned_worker_hint = worker_id
        if self.assigned_dp_rank is None:
            self.assigned_dp_rank = dp_rank


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


@dataclass
class _AdmissionGate:
    """Serialize admission transactions for one program."""

    lock: asyncio.Lock
    users: int = 0


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
        self._admission_gates: dict[str, _AdmissionGate] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stat_forced_resumes = 0
        self._stat_programs_created = 0
        self._stat_programs_ended = 0
        self._stat_requests_admitted = 0
        self._stat_requests_paused = 0
        self._stat_pauses = 0
        self._stat_resumes = 0
        self._stat_marked_for_pause = 0
        self._stat_worker_assignments = 0
        self._stat_admissions_cancelled = 0

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
        gate = self._admission_gates.get(program_id)
        if gate is None:
            gate = _AdmissionGate(lock=asyncio.Lock())
            self._admission_gates[program_id] = gate
        gate.users += 1
        try:
            async with gate.lock:
                return await self._admit_request(program_id, estimated_prompt_tokens)
        finally:
            gate.users -= 1
            if gate.users == 0 and self._admission_gates.get(program_id) is gate:
                self._admission_gates.pop(program_id)

    async def _admit_request(
        self,
        program_id: str,
        estimated_prompt_tokens: int,
    ) -> PauseDecision:
        wait_started = time.monotonic()
        async with self._lock:
            snapshot = self._table.snapshot_request(program_id)
            wait_event, was_paused = self._admit_locked(
                program_id, estimated_prompt_tokens
            )
            # No await can replace this program before its identity is captured.
            admitted = self._table.programs.get(program_id)
            admitted_epoch = admitted.admission_epoch if admitted is not None else None

        try:
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
                            replica = self._least_loaded_replica_locked(
                                self._capacity.snapshot()
                            )
                            self._resume_program(program, replica)
                            self._stat_forced_resumes += 1

            waited = time.monotonic() - wait_started

            async with self._lock:
                program = self._table.programs.get(program_id)
                if program is None:
                    return PauseDecision(program_id=program_id, waited_seconds=waited)

                self._stat_requests_admitted += 1
                if was_paused:
                    self._stat_requests_paused += 1

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
                    assigned_replica_hint=self._program_replica(program),
                )
        except asyncio.CancelledError:
            # Admission mutates shared state before the first cancellable wait.
            await self._rollback_admission_shielded(
                program_id, snapshot, admitted, admitted_epoch
            )
            raise

    async def _rollback_admission_shielded(
        self,
        program_id: str,
        snapshot: Optional[RequestSnapshot],
        admitted: Optional[Program],
        admitted_epoch: Optional[int],
    ) -> None:
        """Finish rollback even if request cancellation is delivered again."""
        rollback = asyncio.ensure_future(
            self._rollback_admission(program_id, snapshot, admitted, admitted_epoch)
        )
        while not rollback.done():
            try:
                await asyncio.shield(rollback)
            except asyncio.CancelledError:
                if rollback.cancelled():
                    break
        if not rollback.cancelled():
            rollback.result()

    async def _rollback_admission(
        self,
        program_id: str,
        snapshot: Optional[RequestSnapshot],
        admitted: Optional[Program],
        admitted_epoch: Optional[int],
    ) -> None:
        if admitted is None:
            return
        async with self._lock:
            if self._table.programs.get(program_id) is not admitted:
                return
            if admitted.admission_epoch != admitted_epoch:
                # A later admission owns the shared Program state.
                return
            self._table.rollback_request(program_id, snapshot)
            self._stat_admissions_cancelled += 1
            logger.info(
                "thunderagent.program admission_cancelled program=%s "
                "retained=%s active=%d paused=%d",
                program_id,
                snapshot is not None,
                len(self._table.programs),
                len(self._table.paused),
            )

    def _admit_locked(
        self,
        program_id: str,
        estimated_prompt_tokens: int,
    ) -> tuple[Optional[asyncio.Event], bool]:
        # Caller holds self._lock.
        was_new = program_id not in self._table.programs
        program = self._table.begin_request(program_id, estimated_prompt_tokens)
        if was_new:
            self._stat_programs_created += 1
            logger.info(
                "thunderagent.program created program=%s "
                "estimated_prompt_tokens=%d active=%d",
                program_id,
                estimated_prompt_tokens,
                len(self._table.programs),
            )
        if program.lifecycle == ProgramLifecycle.PAUSED:
            program.waiting = program.waiting or asyncio.Event()
            return program.waiting, True

        # Keep one capacity view for both sticky-pin validation and admission.
        # A live worker can change its advertised DP range without changing
        # its instance ID; an empty or worker-missing snapshot still means
        # the MDC view is temporarily incomplete, so preserve the existing pin.
        capacities = self._normalize_capacities(self._capacity.snapshot())
        needs_assignment = not self._has_replica_assignment(program)
        stale_replacement = False
        live_worker_ids: set[int] = set()
        if program.assigned_worker_id is not None:
            live_worker_ids = self._capacity.live_worker_ids()
            assigned_replica = self._program_replica(program)
            worker_has_advertised_replica = any(
                replica[0] == program.assigned_worker_id for replica in capacities
            )
            if (
                (not live_worker_ids or program.assigned_worker_id in live_worker_ids)
                and assigned_replica is not None
                and (
                    not worker_has_advertised_replica or assigned_replica in capacities
                )
            ):
                return None, False
            stale_worker_id = program.assigned_worker_id
            program.assigned_worker_id = None
            program.assigned_dp_rank = None
            needs_assignment = True
            stale_replacement = bool(live_worker_ids) and (
                stale_worker_id not in live_worker_ids
            )
            logger.info(
                "thunderagent.worker stale_pin program=%s old_worker=%s "
                "available_workers=%s",
                program_id,
                stale_worker_id,
                sorted(live_worker_ids),
            )

        if not needs_assignment:
            return None, False

        if stale_replacement:
            capacities = {
                replica: capacity
                for replica, capacity in capacities.items()
                if replica[0] in live_worker_ids
            }

        if not capacities:
            # Cold start: MDC hasn't published yet. Let the request flow
            # through with no pin; the chunk-loop callback will populate the
            # worker and DP rank once the engine picks a replica, and subsequent
            # turns get the sticky pin.
            return None, False
        replica = self._select_replica_for_admission_locked(
            capacities,
            program.token_total,
            queue_behind_paused=not stale_replacement,
        )
        if replica is not None:
            program.assigned_worker_id, program.assigned_dp_rank = replica
            self._stat_worker_assignments += 1
            return None, False

        # All workers full: queue until the scheduler tick resumes us.
        program.waiting = program.waiting or asyncio.Event()
        program.lifecycle = ProgramLifecycle.PAUSED
        self._table.paused[program_id] = None
        self._stat_pauses += 1
        logger.info(
            "thunderagent.program paused program=%s reason=admission_full "
            "tokens=%d paused=%d",
            program_id,
            program.token_total,
            len(self._table.paused),
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

    @staticmethod
    def _normalize_capacities(
        capacities: dict[Any, int],
    ) -> dict[ReplicaKey, int]:
        """Normalize legacy worker-only capacity maps to replica keys."""
        normalized: dict[ReplicaKey, int] = {}
        for key, capacity in capacities.items():
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and all(
                    isinstance(part, int) and not isinstance(part, bool) for part in key
                )
            ):
                normalized[(key[0], key[1])] = capacity
            elif isinstance(key, int) and not isinstance(key, bool):
                normalized[(key, 0)] = capacity
        return normalized

    @staticmethod
    def _program_replica(program: Program) -> Optional[ReplicaKey]:
        if program.assigned_worker_id is None or program.assigned_dp_rank is None:
            return None
        return (program.assigned_worker_id, program.assigned_dp_rank)

    @classmethod
    def _has_replica_assignment(cls, program: Program) -> bool:
        return cls._program_replica(program) is not None

    async def assign_replica(self, program_id: str, replica: ReplicaKey) -> None:
        """Record the worker and DP rank selected by the backend."""
        worker_id, dp_rank = replica
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is not None:
                program.assigned_worker_id = worker_id
                program.assigned_dp_rank = dp_rank
                self._stat_worker_assignments += 1

    async def assign_worker(
        self,
        program_id: str,
        worker_id: int,
        dp_rank: Optional[int] = None,
    ) -> Optional[ReplicaKey]:
        """Compatibility wrapper for callers that only know a worker ID.

        A worker-only assignment is accepted only when its capacity snapshot
        identifies exactly one replica. Multi-rank and unknown workers must
        provide the rank so a request is never pinned ambiguously.
        """
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                return None
            if dp_rank is None:
                capacities = self._normalize_capacities(self._capacity.snapshot())
                replicas = [key for key in capacities if key[0] == worker_id]
                if len(replicas) == 1:
                    dp_rank = replicas[0][1]
                else:
                    logger.warning(
                        "Ignoring ambiguous worker-only assignment for worker=%s",
                        worker_id,
                    )
                    return None
            program.assigned_worker_id = worker_id
            program.assigned_dp_rank = dp_rank
            self._stat_worker_assignments += 1
            return (worker_id, dp_rank)

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
        capacities = self._normalize_capacities(self._capacity.snapshot())
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

    def _active_programs_for_replica(self, replica: ReplicaKey | int) -> list[Program]:
        if isinstance(replica, tuple):
            worker_id, dp_rank = replica
        else:
            worker_id, dp_rank = replica, 0
        return [
            p
            for p in self._table.programs.values()
            if p.lifecycle == ProgramLifecycle.ACTIVE
            and p.assigned_worker_id == worker_id
            and p.assigned_dp_rank == dp_rank
        ]

    def _replica_used(self, replica: ReplicaKey | int, *, decayed: bool = False) -> int:
        programs = self._active_programs_for_replica(replica)
        tokens = sum(self._program_tokens(p, decayed=decayed) for p in programs)
        return tokens + len(programs) * self._cfg.buffer_per_program

    def _least_loaded_replica_locked(
        self, capacities: dict[Any, int]
    ) -> Optional[ReplicaKey]:
        capacities = self._normalize_capacities(capacities)
        if not capacities:
            return None
        return max(
            capacities,
            key=lambda replica: (
                capacities[replica] - self._replica_used(replica, decayed=True)
            ),
        )

    def _select_replica_for_admission_locked(
        self,
        capacities: dict[Any, int],
        estimated_tokens: int,
        *,
        queue_behind_paused: bool,
    ) -> Optional[ReplicaKey]:
        capacities = self._normalize_capacities(capacities)
        # Fairness: new programs queue behind any existing paused program.
        if queue_behind_paused and self._table.paused:
            return None
        buffer = self._cfg.buffer_per_program
        required = estimated_tokens + buffer
        candidates = [
            (replica, self._replica_used(replica))
            for replica, capacity in capacities.items()
            if capacity - self._replica_used(replica) >= required
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])[0]

    def _active_programs_for_worker(self, worker_id: int) -> list[Program]:
        """Return all active programs on a worker (legacy aggregate view)."""
        return [
            p
            for p in self._table.programs.values()
            if p.lifecycle == ProgramLifecycle.ACTIVE
            and p.assigned_worker_id == worker_id
        ]

    def _worker_used(self, worker_id: int, *, decayed: bool = False) -> int:
        """Return aggregate usage across a worker's replicas.

        Admission and pressure decisions use ``_replica_used``; this helper is
        retained for compatibility with existing operational callers.
        """
        programs = self._active_programs_for_worker(worker_id)
        tokens = sum(self._program_tokens(p, decayed=decayed) for p in programs)
        return tokens + len(programs) * self._cfg.buffer_per_program

    def _least_loaded_worker_locked(self, capacities: dict[Any, int]) -> Optional[int]:
        replica = self._least_loaded_replica_locked(capacities)
        return replica[0] if replica is not None else None

    def _select_worker_for_admission_locked(
        self,
        capacities: dict[Any, int],
        estimated_tokens: int,
        *,
        queue_behind_paused: bool,
    ) -> Optional[int]:
        replica = self._select_replica_for_admission_locked(
            capacities,
            estimated_tokens,
            queue_behind_paused=queue_behind_paused,
        )
        return replica[0] if replica is not None else None

    def _apply_soft_demotes(self, capacities: dict[Any, int]) -> None:
        capacities = self._normalize_capacities(capacities)
        soft_until = time.monotonic() + self._cfg.scheduler_interval_seconds * 1.5
        for replica, capacity in capacities.items():
            util = self._replica_used(replica) / capacity
            if not (
                self._cfg.soft_demote_threshold <= util < self._cfg.pause_threshold
            ):
                continue
            for program in self._active_programs_for_replica(replica):
                if (
                    not program.marked_for_pause
                    and program.soft_demoted_until < soft_until
                ):
                    program.soft_demoted_until = soft_until

    async def _pause_until_safe(self, capacities: dict[Any, int]) -> None:
        capacities = self._normalize_capacities(capacities)
        threshold = self._cfg.pause_threshold
        pause_target = min(self._cfg.pause_target, threshold)

        for replica, capacity in capacities.items():
            # Hold the lock for the entire per-replica decision so the snapshot
            # of program state used by _smallest_candidates / _replica_used
            # cannot race with concurrent before_request admissions.
            async with self._lock:
                base_used = self._replica_used(replica)
                if base_used <= capacity * threshold:
                    continue

                target_limit = capacity * pause_target
                paused_this_tick = 0
                marked_this_tick = 0
                # Bound the inner loop by total program count so a candidate
                # transitioning out from under us can't spin the tick.
                for _ in range(len(self._table.programs) + 1):
                    if self._replica_used(replica) <= target_limit:
                        break
                    acting, reasoning = self._smallest_candidates(replica)
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
                            self._stat_marked_for_pause += 1
                            marked_this_tick += 1
                        continue
                    break

                final_used = self._replica_used(replica)

            if paused_this_tick or marked_this_tick:
                logger.info(
                    "scheduler.tick worker=%s dp_rank=%s paused=%d marked=%d "
                    "util=%.4f -> %.4f",
                    replica[0],
                    replica[1],
                    paused_this_tick,
                    marked_this_tick,
                    base_used / capacity,
                    final_used / capacity,
                )

    def _smallest_candidates(
        self, replica: ReplicaKey | int
    ) -> tuple[Optional[Program], Optional[Program]]:
        if isinstance(replica, tuple):
            worker_id, dp_rank = replica
        else:
            worker_id, dp_rank = replica, 0
        smallest_acting: Optional[Program] = None
        smallest_reasoning: Optional[Program] = None
        for program in self._table.programs.values():
            if (
                program.assigned_worker_id != worker_id
                or program.assigned_dp_rank != dp_rank
            ):
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
            elif program.status == ProgramStatus.REASONING and (
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
        program.assigned_dp_rank = None
        if program.waiting is None:
            program.waiting = asyncio.Event()
        else:
            program.waiting.clear()
        self._table.paused[program_id] = None
        self._stat_pauses += 1
        logger.info(
            "thunderagent.program paused program=%s reason=pressure "
            "tokens=%d paused=%d",
            program_id,
            program.token_total,
            len(self._table.paused),
        )
        return True

    async def end_program(self, program_id: str) -> bool:
        """Release a finished program.

        Deletes it from the program table + paused set and wakes any waiter,
        so its tokens stop counting against worker utilization. Mirrors
        upstream TA's ``release_program``. Idempotent: returns False if unknown.
        """
        async with self._lock:
            program = self._table.programs.get(program_id)
            if program is None:
                return False
            program.lifecycle = ProgramLifecycle.TERMINATED
            if program.waiting is not None:
                program.waiting.set()  # unblock any coroutine paused on this program
                program.waiting = None
            self._table.release(program_id)
            self._stat_programs_ended += 1
            logger.info(
                "thunderagent.program terminated program=%s remaining=%d",
                program_id,
                len(self._table.programs),
            )
            return True

    async def _greedy_resume(self, capacities: dict[Any, int]) -> None:
        capacities = self._normalize_capacities(capacities)
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
                (
                    replica,
                    int(capacity * resume_ceiling)
                    - self._replica_used(replica, decayed=False),
                )
                for replica, capacity in capacities.items()
            ]
            backend_caps = [
                (replica, remaining)
                for replica, remaining in backend_caps
                if remaining > self._cfg.buffer_per_program
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

            resumed_this_tick = 0
            for program in resumable_programs:
                if not backend_caps:
                    break
                replica, remaining = backend_caps[0]
                if min_required > remaining:
                    break
                required = program.token_total + self._cfg.buffer_per_program
                if required > remaining:
                    continue
                self._resume_program(program, replica)
                resumed_this_tick += 1
                updated_remaining = remaining - required
                if updated_remaining > self._cfg.buffer_per_program:
                    backend_caps[0] = (replica, updated_remaining)
                    backend_caps.sort(key=lambda x: -x[1])
                else:
                    backend_caps.pop(0)

            if resumed_this_tick:
                logger.info(
                    "scheduler.tick resumed=%d still_paused=%d",
                    resumed_this_tick,
                    len(self._table.paused),
                )

    def _resume_program(
        self,
        program: Program,
        target_replica: Optional[ReplicaKey | int] = None,
        *,
        target_worker_id: Optional[int] = None,
        target_dp_rank: Optional[int] = None,
    ) -> None:
        # Caller holds self._lock.
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return
        if isinstance(target_replica, int):
            target_replica = (
                target_replica,
                0 if target_dp_rank is None else target_dp_rank,
            )
        if target_replica is None and target_worker_id is not None:
            target_replica = (
                target_worker_id,
                0 if target_dp_rank is None else target_dp_rank,
            )
        program.lifecycle = ProgramLifecycle.ACTIVE
        if target_replica is None:
            program.assigned_worker_id = None
            program.assigned_dp_rank = None
        else:
            program.assigned_worker_id, program.assigned_dp_rank = target_replica
            self._stat_worker_assignments += 1
        notify = program.waiting
        program.waiting = None
        self._table.paused.pop(program.program_id, None)
        if notify is not None:
            notify.set()
        self._stat_resumes += 1
        logger.info(
            "thunderagent.program resumed program=%s worker=%s dp_rank=%s "
            "tokens=%d paused=%d",
            program.program_id,
            target_replica[0] if target_replica is not None else None,
            target_replica[1] if target_replica is not None else None,
            program.token_total,
            len(self._table.paused),
        )

    def _worker_snapshot_locked(
        self, capacities: dict[Any, int]
    ) -> dict[str, dict[str, Any]]:
        """Build worker metrics from per-replica accounting while holding the lock."""
        capacities = self._normalize_capacities(capacities)
        replica_rows: dict[ReplicaKey, dict[str, Any]] = {}
        for replica, capacity in capacities.items():
            programs = self._active_programs_for_replica(replica)
            active_count = len(programs)
            buffer_tokens = active_count * self._cfg.buffer_per_program
            used = sum(self._program_tokens(p) for p in programs) + buffer_tokens
            used_decayed = (
                sum(self._program_tokens(p, decayed=True) for p in programs)
                + buffer_tokens
            )
            replica_rows[replica] = {
                "worker_id": replica[0],
                "dp_rank": replica[1],
                "capacity": capacity,
                "used": used,
                "used_decayed": used_decayed,
                "utilization": used / capacity if capacity else None,
                "utilization_decayed": used_decayed / capacity if capacity else None,
                "active_programs": active_count,
            }

        workers: dict[str, dict[str, Any]] = {}
        for worker_id in dict.fromkeys(replica[0] for replica in capacities):
            replicas = {
                str(replica[1]): row
                for replica, row in replica_rows.items()
                if replica[0] == worker_id
            }
            capacity = sum(row["capacity"] for row in replicas.values())
            worker_used = sum(row["used"] for row in replicas.values())
            worker_used_decayed = sum(row["used_decayed"] for row in replicas.values())
            active_count = sum(row["active_programs"] for row in replicas.values())
            workers[str(worker_id)] = {
                "capacity": capacity,
                "used": worker_used,
                "used_decayed": worker_used_decayed,
                "utilization": worker_used / capacity if capacity else None,
                "utilization_decayed": worker_used_decayed / capacity
                if capacity
                else None,
                "active_programs": active_count,
            }
        return workers

    async def status_snapshot(self) -> dict:
        async with self._lock:
            capacities = self._capacity.snapshot()
            lifecycle_counts = {lifecycle.value: 0 for lifecycle in ProgramLifecycle}
            status_counts = {status.value: 0 for status in ProgramStatus}
            programs = []

            for program in self._table.programs.values():
                lifecycle_counts[program.lifecycle.value] += 1
                status_counts[program.status.value] += 1
                programs.append(
                    {
                        "program_id": program.program_id,
                        "lifecycle": program.lifecycle.value,
                        "status": program.status.value,
                        "assigned_worker_id": program.assigned_worker_id,
                        "assigned_dp_rank": program.assigned_dp_rank,
                        "token_total": program.token_total,
                        "step_count": program.step_count,
                        "marked_for_pause": program.marked_for_pause,
                        "soft_demoted": program.soft_demoted_until > time.monotonic(),
                    }
                )

            workers = self._worker_snapshot_locked(capacities)

            return {
                "programs_total": len(self._table.programs),
                "paused_total": len(self._table.paused),
                "lifecycle_counts": lifecycle_counts,
                "status_counts": status_counts,
                "workers": workers,
                "programs": programs,
            }

    async def metrics_snapshot(self) -> dict:
        async with self._lock:
            workers = self._worker_snapshot_locked(self._capacity.snapshot())
            return {
                "counters": {
                    "programs_created_total": self._stat_programs_created,
                    "programs_ended_total": self._stat_programs_ended,
                    "requests_admitted_total": self._stat_requests_admitted,
                    "requests_paused_total": self._stat_requests_paused,
                    "program_pauses_total": self._stat_pauses,
                    "program_resumes_total": self._stat_resumes,
                    "programs_marked_for_pause_total": self._stat_marked_for_pause,
                    "forced_resumes_total": self._stat_forced_resumes,
                    "admissions_cancelled_total": self._stat_admissions_cancelled,
                    "worker_assignments_total": self._stat_worker_assignments,
                },
                "gauges": {
                    "programs_total": len(self._table.programs),
                    "paused_total": len(self._table.paused),
                    "workers_total": len(workers),
                },
                "workers": workers,
            }
