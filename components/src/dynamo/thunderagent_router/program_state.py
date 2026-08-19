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

    # Bumped by every `begin_request`. Two concurrent requests for one session
    # share this object, so object identity alone cannot tell a rollback whether
    # the mutations it is about to undo are still its own.
    admission_epoch: int = 0


@dataclass(frozen=True)
class RequestSnapshot:
    """State of a Program immediately before ``begin_request`` mutates it.

    ``program`` is captured so a rollback can tell "the program I mutated" from
    "a program with the same id that a later request created": the two are
    different objects and only the first may be restored. ``admission_epoch``
    extends that to the case where the object is the same but a later request
    has admitted its own turn on it since.
    """

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
        """Record what a following ``begin_request``/admission attempt will change.

        Returns None when the program does not exist yet: that is the signal to
        ``rollback_request`` that the attempt is what created it, so undoing the
        attempt means removing it rather than restoring fields.
        """
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
        """Undo one abandoned request's mutations, given its pre-request snapshot.

        With ``snapshot`` None the program did not exist before the attempt, so
        it is dropped from both tables; dropping it from ``programs`` is what
        releases the capacity the scheduler had accounted to it. Otherwise the
        recorded fields — including ``paused`` membership — are put back, which
        preserves a live session's history that ``release`` would discard.

        Restoration is skipped when a different Program object now holds the id:
        the recorded one was released meanwhile and a newer request owns it.

        The caller is responsible for the matching ``admission_epoch`` check --
        the same object can be shared by a later request whose mutations must
        not be undone. See ``ThunderAgentScheduler._rollback_admission``.
        """
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
        # A concurrent admission for the same program may have installed its own
        # Event; leave that one in place rather than stranding its waiter.
        if program.waiting is None or program.waiting is snapshot.waiting:
            program.waiting = snapshot.waiting
        if program.lifecycle == ProgramLifecycle.PAUSED and program.waiting is not None:
            # The Event may have been set by the resume this rollback is undoing.
            # A paused program has to make its next waiter wait, and clearing
            # cannot un-wake anyone: `set` completes every waiter already parked
            # on the Event, and `clear` only affects `wait` calls made after it.
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
