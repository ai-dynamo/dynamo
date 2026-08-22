# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Program lifecycle data model. Mirrors ``ThunderAgent/program/state.py``.

v0 difference: ``token_total`` is real ``prompt_tokens + completion_tokens``
from chat-completions ``usage``, not upstream's ``chars / 5`` heuristic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProgramStatus(Enum):
    REASONING = "reasoning"
    ACTING = "acting"


class ProgramLifecycle(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class Program:
    program_id: str

    status: ProgramStatus = ProgramStatus.REASONING
    lifecycle: ProgramLifecycle = ProgramLifecycle.ACTIVE

    assigned_worker_id: Optional[int] = None

    token_total: int = 0

    step_count: int = 0
    marked_for_pause: bool = False
    # monotonic seconds; >0 means priority demotion active
    soft_demoted_until: float = 0.0
    waiting: Optional[asyncio.Event] = field(default=None, repr=False)

    # monotonic seconds; used to compute resume-side decay
    acting_since: float = 0.0

    # Distinguishes concurrent admissions that share a Program object.
    admission_epoch: int = 0


@dataclass(frozen=True)
class RequestSnapshot:
    """Program state captured before admission."""

    program: Program
    status: ProgramStatus
    lifecycle: ProgramLifecycle
    assigned_worker_id: Optional[int]
    token_total: int
    step_count: int
    marked_for_pause: bool
    soft_demoted_until: float
    waiting: Optional[asyncio.Event]
    acting_since: float
    was_paused: bool
    admission_epoch: int


@dataclass
class ProgramTable:
    programs: dict[str, Program] = field(default_factory=dict)
    # Insertion-ordered: ties in `_greedy_resume`'s sort resolve oldest-paused
    # first, mirroring upstream TA. Values are unused.
    paused: dict[str, None] = field(default_factory=dict)

    def begin_request(
        self, program_id: str, estimated_prompt_tokens: int = 0
    ) -> Program:
        program = self.programs.get(program_id)
        if program is None:
            program = Program(program_id=program_id)
            self.programs[program_id] = program
        program.step_count += 1
        program.admission_epoch += 1
        if estimated_prompt_tokens > 0:
            program.token_total = estimated_prompt_tokens
        program.status = ProgramStatus.REASONING
        program.acting_since = 0.0
        return program

    def snapshot_request(self, program_id: str) -> Optional[RequestSnapshot]:
        """Capture state before admission; None means admission creates it."""
        program = self.programs.get(program_id)
        if program is None:
            return None
        return RequestSnapshot(
            program=program,
            status=program.status,
            lifecycle=program.lifecycle,
            assigned_worker_id=program.assigned_worker_id,
            token_total=program.token_total,
            step_count=program.step_count,
            marked_for_pause=program.marked_for_pause,
            soft_demoted_until=program.soft_demoted_until,
            waiting=program.waiting,
            acting_since=program.acting_since,
            was_paused=program_id in self.paused,
            admission_epoch=program.admission_epoch,
        )

    def rollback_request(
        self, program_id: str, snapshot: Optional[RequestSnapshot]
    ) -> None:
        """Restore state only if the snapshot still names the current program."""
        if snapshot is None:
            self.paused.pop(program_id, None)
            self.programs.pop(program_id, None)
            return

        program = self.programs.get(program_id)
        if program is not snapshot.program:
            return

        program.status = snapshot.status
        program.lifecycle = snapshot.lifecycle
        program.assigned_worker_id = snapshot.assigned_worker_id
        program.token_total = snapshot.token_total
        program.step_count = snapshot.step_count
        program.marked_for_pause = snapshot.marked_for_pause
        program.soft_demoted_until = snapshot.soft_demoted_until
        program.acting_since = snapshot.acting_since
        program.admission_epoch = snapshot.admission_epoch
        # Do not replace an event installed by a concurrent admission.
        if program.waiting is None or program.waiting is snapshot.waiting:
            program.waiting = snapshot.waiting
        if program.lifecycle == ProgramLifecycle.PAUSED and program.waiting is not None:
            # A set event would admit the next paused turn without a capacity check.
            program.waiting.clear()
        if snapshot.was_paused:
            self.paused[program_id] = None
        else:
            self.paused.pop(program_id, None)

    def end_request(
        self, program_id: str, prompt_tokens: int, completion_tokens: int
    ) -> Optional[Program]:
        program = self.programs.get(program_id)
        if program is None:
            return None
        program.token_total = prompt_tokens + completion_tokens
        program.status = ProgramStatus.ACTING
        program.acting_since = time.monotonic()
        return program

    def release(self, program_id: str) -> Optional[Program]:
        """Remove a finished program from the table (and the paused set).

        Mirrors upstream TA's ``release_program`` deletion. Returns the removed
        Program (or None if it was already gone).
        """
        self.paused.pop(program_id, None)
        return self.programs.pop(program_id, None)
