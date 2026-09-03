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
from dataclasses import dataclass, field
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
    # Replica this program is accounted against, and therefore the pin to send. None
    # means the program is not placed yet, so the request goes out unpinned.
    assigned_replica_hint: Optional[ReplicaKey] = None
    # Identifies the admission this decision belongs to, so a late first-chunk back-fill
    # cannot write into a placement a subsequent turn already made.
    admission_epoch: int = 0


@dataclass
class _ReplicaUsage:
    """One replica's occupancy, from a single pass over the program table.

    Kept live for a whole tick: pause and resume adjust it as they mutate the table. No
    ``used_decayed`` field -- it moves with the clock, so it is derived on demand.
    """

    capacity: int
    used: int = 0
    programs: list[Program] = field(default_factory=list)


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
                            usage = self._replica_usage_locked(
                                self._capacity.snapshot()
                            )
                            replica = self._least_loaded_replica_locked(usage)
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
                    assigned_replica_hint=program.assigned_replica,
                    admission_epoch=program.admission_epoch,
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

        # Any unplaced program, not just a new one: the back-fill records only a complete
        # replica, so admission is what repairs a program whose response named no rank.
        needs_assignment = program.assigned_replica is None
        stale_replacement = False
        live_worker_ids: set[int] = set()
        if program.assigned_replica is not None:
            live_worker_ids = self._capacity.live_worker_ids()
            if not live_worker_ids or program.assigned_replica[0] in live_worker_ids:
                return None, False
            stale_worker_id = program.assigned_replica[0]
            program.assigned_replica = None
            needs_assignment = True
            stale_replacement = True
            logger.info(
                "thunderagent.worker stale_pin program=%s old_worker=%s "
                "available_workers=%s",
                program_id,
                stale_worker_id,
                sorted(live_worker_ids),
            )

        if not needs_assignment:
            return None, False

        capacities = self._capacity.snapshot()
        if stale_replacement:
            capacities = {
                key: capacity
                for key, capacity in capacities.items()
                if key[0] in live_worker_ids
            }

        if not capacities:
            # Cold start: no MDC yet, so no replica to reason about. Let the request
            # through unassigned; the first response chunk back-fills the pair.
            return None, False
        replica = self._select_replica_for_admission_locked(
            self._replica_usage_locked(capacities),
            program.token_total,
            queue_behind_paused=was_new and not stale_replacement,
        )
        if replica is not None:
            # Placing here is pre-existing (worker-keyed did the same argmin), and not
            # binding: _greedy_resume_locked re-picks the replica on every resume.
            program.assigned_replica = replica
            self._stat_worker_assignments += 1
            return None, False

        if not was_new:
            # Already running and unplaceable: let the turn through unpinned rather than
            # parking a live session. The next turn retries.
            return None, False

        # Every replica is full: queue until the scheduler tick resumes us.
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

    async def assign_worker(
        self,
        program_id: str,
        replica: Optional[ReplicaKey],
        *,
        admission_epoch: int,
    ) -> bool:
        """Back-fill the replica the engine chose. Cold start only.

        Refuses a rank-less answer, a stale epoch (a later turn owns this Program) and an
        already-placed program. Returns whether it recorded anything.
        """
        if replica is None:
            return False
        async with self._lock:
            program = self._table.programs.get(program_id)
            # Released, superseded by a later turn, not running, or already placed by
            # admission -- which is authoritative. A response only fills a gap it left.
            if (
                program is None
                or program.admission_epoch != admission_epoch
                or program.lifecycle != ProgramLifecycle.ACTIVE
                or program.assigned_replica is not None
            ):
                return False
            program.assigned_replica = replica
            self._stat_worker_assignments += 1
            return True

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
        # One lock acquisition and one table pass for the whole tick: per-phase rescans
        # cost O(replicas x programs), and replicas went from W to W x dp_size.
        async with self._lock:
            usage = self._replica_usage_locked(capacities)
            # Upstream TA ordering: resume first, then pause -- a program paused
            # this tick can't resume until the next.
            self._apply_soft_demotes(usage)
            self._greedy_resume_locked(usage)
            self._pause_until_safe_locked(usage)

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

    def _replica_usage_locked(
        self, capacities: dict[ReplicaKey, int]
    ) -> dict[ReplicaKey, _ReplicaUsage]:
        """Bucket ACTIVE programs by replica in one O(programs + replicas) pass.

        Caller holds ``self._lock``. Unplaced programs, and ones naming a replica no longer
        in ``capacities``, are counted against nothing -- as before.
        """
        usage = {
            key: _ReplicaUsage(capacity=capacity)
            for key, capacity in capacities.items()
        }
        buffer = self._cfg.buffer_per_program
        for program in self._table.programs.values():
            replica = program.assigned_replica
            if replica is None or program.lifecycle != ProgramLifecycle.ACTIVE:
                continue
            entry = usage.get(replica)
            if entry is None:
                continue
            entry.programs.append(program)
            entry.used += self._program_tokens(program) + buffer
        return usage

    def _decayed_used(self, entry: _ReplicaUsage) -> int:
        """``entry.used`` with the ACTING decay applied; derived on demand because it moves
        with the clock."""
        tokens = sum(self._program_tokens(p, decayed=True) for p in entry.programs)
        return tokens + len(entry.programs) * self._cfg.buffer_per_program

    def _least_loaded_replica_locked(
        self, usage: dict[ReplicaKey, _ReplicaUsage]
    ) -> Optional[ReplicaKey]:
        if not usage:
            return None
        return max(
            usage,
            key=lambda key: usage[key].capacity - self._decayed_used(usage[key]),
        )

    def _select_replica_for_admission_locked(
        self,
        usage: dict[ReplicaKey, _ReplicaUsage],
        estimated_tokens: int,
        *,
        queue_behind_paused: bool,
    ) -> Optional[ReplicaKey]:
        # Fairness: new programs queue behind any existing paused program.
        if queue_behind_paused and self._table.paused:
            return None
        required = estimated_tokens + self._cfg.buffer_per_program
        best_key: Optional[ReplicaKey] = None
        best_used: Optional[int] = None
        for key, entry in usage.items():
            if entry.capacity - entry.used < required:
                continue
            if best_used is None or entry.used < best_used:
                best_key, best_used = key, entry.used
        return best_key

    def _apply_soft_demotes(self, usage: dict[ReplicaKey, _ReplicaUsage]) -> None:
        soft_until = time.monotonic() + self._cfg.scheduler_interval_seconds * 1.5
        for entry in usage.values():
            util = entry.used / entry.capacity
            if not (
                self._cfg.soft_demote_threshold <= util < self._cfg.pause_threshold
            ):
                continue
            for program in entry.programs:
                if (
                    not program.marked_for_pause
                    and program.soft_demoted_until < soft_until
                ):
                    program.soft_demoted_until = soft_until

    def _pause_until_safe_locked(self, usage: dict[ReplicaKey, _ReplicaUsage]) -> None:
        """Shed load from every over-threshold replica. Caller holds ``self._lock``.

        Smallest-ACTING-first then mark REASONING, as upstream. Candidate order is taken
        once per replica instead of rescanning the table on every pause.
        """
        threshold = self._cfg.pause_threshold
        pause_target = min(self._cfg.pause_target, threshold)
        buffer = self._cfg.buffer_per_program

        for key, entry in usage.items():
            base_used = entry.used
            if base_used <= entry.capacity * threshold:
                continue

            target_limit = entry.capacity * pause_target
            paused_this_tick = 0
            marked_this_tick = 0

            acting = sorted(
                (p for p in entry.programs if p.status == ProgramStatus.ACTING),
                key=lambda p: p.token_total,
            )
            for program in acting:
                if entry.used <= target_limit:
                    break
                if program.marked_for_pause:
                    continue
                freed = self._program_tokens(program) + buffer
                if not self._pause_acting_locked(program.program_id):
                    continue
                entry.programs.remove(program)
                entry.used -= freed
                paused_this_tick += 1

            if entry.used > target_limit:
                # Marking does not free anything now; it defers the pause to
                # after_request. Upstream keeps marking until the candidates run out.
                reasoning = sorted(
                    (p for p in entry.programs if p.status == ProgramStatus.REASONING),
                    key=lambda p: p.token_total,
                )
                for program in reasoning:
                    if program.marked_for_pause:
                        continue
                    if program.lifecycle != ProgramLifecycle.ACTIVE:
                        continue
                    program.marked_for_pause = True
                    self._stat_marked_for_pause += 1
                    marked_this_tick += 1

            if paused_this_tick or marked_this_tick:
                logger.info(
                    "scheduler.tick worker=%s dp_rank=%s paused=%d marked=%d "
                    "util=%.4f -> %.4f",
                    key[0],
                    key[1],
                    paused_this_tick,
                    marked_this_tick,
                    base_used / entry.capacity,
                    entry.used / entry.capacity,
                )

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
        program.assigned_replica = None
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

    def _greedy_resume_locked(self, usage: dict[ReplicaKey, _ReplicaUsage]) -> None:
        """Best-fit-decreasing restore of paused programs. Caller holds ``self._lock``.

        Keeps ``usage`` in step as it resumes -- invariant maintenance, since a replica only
        accepts a resume while it stays under the resume ceiling.
        """
        if not self._table.paused:
            return

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
        buffer = self._cfg.buffer_per_program
        backend_caps = [
            (key, int(entry.capacity * resume_ceiling) - entry.used)
            for key, entry in usage.items()
        ]
        backend_caps = [(key, r) for key, r in backend_caps if r > buffer]
        if not backend_caps:
            return

        backend_caps.sort(key=lambda x: -x[1])

        total_capacity = sum(r for _, r in backend_caps)
        resumable_programs: list[Program] = []
        cumulative = 0
        for program in paused_programs:
            required = program.token_total + buffer
            if cumulative + required <= total_capacity:
                resumable_programs.append(program)
                cumulative += required

        if not resumable_programs:
            return

        resumable_programs.sort(key=lambda p: -p.token_total)
        min_required = min(p.token_total for p in resumable_programs) + buffer

        resumed_this_tick = 0
        for program in resumable_programs:
            if not backend_caps:
                break
            replica, remaining = backend_caps[0]
            if min_required > remaining:
                break
            required = program.token_total + buffer
            if required > remaining:
                continue
            self._resume_program(program, replica)
            entry = usage[replica]
            entry.programs.append(program)
            entry.used += self._program_tokens(program) + buffer
            resumed_this_tick += 1
            updated_remaining = remaining - required
            if updated_remaining > buffer:
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
        self, program: Program, target_replica: Optional[ReplicaKey]
    ) -> None:
        # Caller holds self._lock.
        if program.lifecycle != ProgramLifecycle.PAUSED:
            return
        program.lifecycle = ProgramLifecycle.ACTIVE
        program.assigned_replica = target_replica
        if target_replica is not None:
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
            None if target_replica is None else target_replica[0],
            None if target_replica is None else target_replica[1],
            program.token_total,
            len(self._table.paused),
        )

    def _worker_snapshot_locked(
        self, usage: dict[ReplicaKey, _ReplicaUsage]
    ) -> dict[str, dict[str, Any]]:
        """Per-replica metrics, keyed ``"<worker_id>:<dp_rank>"``, both parts also as fields.

        Reads the same usage map a tick does, so the two cannot disagree.
        """
        workers = {}
        for replica, entry in usage.items():
            used_decayed = self._decayed_used(entry)
            workers[f"{replica[0]}:{replica[1]}"] = {
                "worker_id": replica[0],
                "dp_rank": replica[1],
                "capacity": entry.capacity,
                "used": entry.used,
                "used_decayed": used_decayed,
                "utilization": entry.used / entry.capacity if entry.capacity else None,
                "utilization_decayed": (
                    used_decayed / entry.capacity if entry.capacity else None
                ),
                "active_programs": len(entry.programs),
            }
        return workers

    async def status_snapshot(self) -> dict:
        async with self._lock:
            usage = self._replica_usage_locked(self._capacity.snapshot())
            lifecycle_counts = {lifecycle.value: 0 for lifecycle in ProgramLifecycle}
            status_counts = {status.value: 0 for status in ProgramStatus}
            programs = []

            for program in self._table.programs.values():
                lifecycle_counts[program.lifecycle.value] += 1
                status_counts[program.status.value] += 1
                worker_id, dp_rank = program.assigned_replica or (None, None)
                programs.append(
                    {
                        "program_id": program.program_id,
                        "lifecycle": program.lifecycle.value,
                        "status": program.status.value,
                        "assigned_worker_id": worker_id,
                        "assigned_dp_rank": dp_rank,
                        "token_total": program.token_total,
                        "step_count": program.step_count,
                        "marked_for_pause": program.marked_for_pause,
                        "soft_demoted": program.soft_demoted_until > time.monotonic(),
                    }
                )

            workers = self._worker_snapshot_locked(usage)

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
            workers = self._worker_snapshot_locked(
                self._replica_usage_locked(self._capacity.snapshot())
            )
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
