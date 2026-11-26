# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Highest-U admission with capacity-checked backfill.

Keep ThunderAgent's worker reservations and tool-phase memory-pressure pausing.
Account for real token counts from request usage, not a character estimate.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial

from dynamo.duru_router.admission import Candidate, DuruPolicy
from dynamo.duru_router.diagnostics import emit as diagnostic
from dynamo.duru_router.diagnostics import enabled as diagnostics_enabled
from dynamo.duru_router.scoring import AccountingError, score_channel
from dynamo.thunderagent_router.policy import (
    ResumeChoice,
    ResumePlan,
    ThunderAgentPolicy,
)
from dynamo.thunderagent_router.program_state import Program as BaseProgram
from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus
from dynamo.thunderagent_router.program_state import ProgramTable as BaseProgramTable
from dynamo.thunderagent_router.router import ThunderAgentConfig as ThunderAgentConfig
from dynamo.thunderagent_router.router import ThunderAgentScheduler as BaseScheduler


@dataclass
class Program(BaseProgram):
    _u_new_admission: bool = False


class ProgramTable(BaseProgramTable):
    program_cls = Program


@dataclass(kw_only=True)
class DuruResumePlan(ResumePlan):
    candidates: dict[str, Candidate]
    policy: DuruPolicy
    established_waiters: bool
    snapshot: dict
    native_packing: bool = False
    diagnostics: bool = False
    native_rank: dict[str, int] | None = None

    def may_force_admit(self, program: Program) -> bool:
        return self.policy.may_force_admit(self.candidates[program.program_id])

    def record_admission(
        self, program: Program, choice: ResumeChoice, admission_rank: int
    ) -> None:
        program._u_new_admission = False
        if not self.diagnostics:
            return
        candidate = self.candidates[program.program_id]
        row = self.snapshot.get("programs", {}).get(program.program_id, {})
        diagnostic(
            "admission",
            time=time.monotonic(),
            program_id=program.program_id,
            request_id=row.get("request_id"),
            root=row.get("root"),
            U=candidate.measured_u,
            effective_score=candidate.effective_score,
            alpha=None,
            forced=choice.forced,
            worker=choice.replica,
            required=candidate.demand,
            remaining_before=choice.remaining,
            ceiling=choice.ceiling,
            first_admission=candidate.first_admission,
            established_waiters=self.established_waiters,
            native_queue_rank=self.native_rank[program.program_id],
            admission_rank=admission_rank,
            program_status=program.status.value,
            observation_time=self.snapshot.get("observation_time"),
            run_id=self.snapshot.get("run_id"),
        )


class DuruSchedulingPolicy(ThunderAgentPolicy):
    """Queue ordering and admission exceptions; ThunderAgent checks capacity."""

    retry_on_request_change = True
    force_resume_on_timeout = False
    mark_reasoning_for_pause = False

    def __init__(self, reader):
        score_channel()  # Validate before the host creates scheduler state/tasks.
        self.reader = reader
        self.decision_snapshot = {}
        self.last_defer = {}

    def begin_decision(self, program_id=None):
        self.decision_snapshot = self.reader.refresh()
        reason = self.reader.reason
        if self.decision_snapshot and program_id is not None:
            entry = self.decision_snapshot.get("programs", {}).get(program_id, {})
            if not entry or "request_id" not in entry:
                reason = "missing_program_score"
        ready = bool(self.decision_snapshot) and reason == "fresh"
        if not ready:
            now = time.monotonic()
            # A finite set of reasons, rather than a timestamp for every program ever seen.
            if diagnostics_enabled() and now - self.last_defer.get(reason, 0) >= 1:
                self.last_defer[reason] = now
                diagnostic(
                    "control_deferred",
                    reason=reason,
                    program_id=program_id,
                    snapshot_age_s=self.reader.age,
                    observation_time=self.reader.snapshot.get("observation_time"),
                    action="defer_admission_no_native_fallback",
                )
        return ready

    def score_row(self, program_id):
        return self.decision_snapshot.get("programs", {}).get(program_id, {})

    def admission_ready(self, program_id: str) -> bool:
        return self.begin_decision(program_id)

    def rank_initial_admission(self) -> bool:
        return True

    def admission_queued(self, program: Program) -> None:
        program._u_new_admission = True

    def check_admission_owner(self, program, admitted, admitted_epoch) -> None:
        if (
            program is not admitted
            or program is None
            or program.admission_epoch != admitted_epoch
        ):
            raise asyncio.CancelledError("Program terminated during admission")

    def prepare_resume(
        self, programs: list[Program], config: ThunderAgentConfig
    ) -> ResumePlan:
        if not self.begin_decision():
            return ResumePlan(programs=[], resume_ceiling=0.0)
        by_id = {p.program_id: p for p in programs}
        candidates = []
        for program in programs:
            row = self.score_row(program.program_id)
            candidates.append(
                Candidate(
                    identity=program.program_id,
                    demand=program.token_total + config.buffer_per_program,
                    score=row.get("score"),
                    measured_u=row.get("U"),
                    arrival=row.get("arrival", 0.0),
                    known=bool(row),
                    idle=program.status == ProgramStatus.ACTING,
                    pending_request=bool(row.get("request_id"))
                    and program.status == ProgramStatus.REASONING
                    and program.lifecycle == ProgramLifecycle.PAUSED,
                    first_admission=program._u_new_admission,
                )
            )
        policy = DuruPolicy(
            resume_ceiling=config.pause_threshold - config.resume_hysteresis,
        )
        established_waiters = any(not c.first_admission for c in candidates)
        # Initial admissions may fill the replica only when no established
        # program is waiting. Otherwise the whole queue shares resume headroom.
        ceiling = (
            max(0.0, min(1.0, policy.resume_ceiling)) if established_waiters else 1.0
        )
        diagnostics = diagnostics_enabled()
        return DuruResumePlan(
            programs=[by_id[c.identity] for c in policy.order(candidates)],
            resume_ceiling=ceiling,
            candidates={c.identity: c for c in candidates},
            policy=policy,
            established_waiters=established_waiters,
            diagnostics=diagnostics,
            snapshot=self.decision_snapshot,
            native_rank=(
                {p.program_id: i for i, p in enumerate(programs)}
                if diagnostics
                else None
            ),
        )


class DuruScheduler(BaseScheduler):
    """ThunderAgent scheduler configured with Duru's admission policy."""

    program_table_cls = ProgramTable
    policy_cls = DuruSchedulingPolicy

    def __init__(self, capacity, config, *, reader):
        self.policy_cls = partial(DuruSchedulingPolicy, reader=reader)
        super().__init__(capacity, config)

    async def _greedy_resume(self, capacities):
        try:
            await super()._greedy_resume(capacities)
        except AccountingError:
            # Score refresh precedes admissions. Abandon this optional retry;
            # the handler's health checks still expose the accounting failure.
            return
