# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Program lifecycle data model.

Mirrors ``ThunderAgent/program/state.py`` semantics with two intentional
differences for v0:

* ``last_prefix_token_ids`` is captured at request_end so resume can target the
  warmest-KV worker via ``KvRouter.best_worker``. ThunderAgent has no equivalent
  because its public repo does not do KV-aware resume.
* ``token_total`` comes from real ``prompt_tokens + completion_tokens`` reported
  by the response, not a ``chars / 5`` heuristic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProgramStatus(Enum):
    """What the program is currently doing.

    REASONING: a request is on GPU executing inference.
    ACTING: between LLM turns -- harness is running tools or waiting for the
    next user/agent input.
    """

    REASONING = "reasoning"
    ACTING = "acting"


class ProgramLifecycle(Enum):
    """Lifecycle state.

    ACTIVE: program is registered and admissible.
    PAUSED: program is held in the global waiting queue; the next request will
    block on ``waiting`` until the scheduler resumes it.
    TERMINATED: program is released; no further state kept.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class Program:
    """Per-``program_id`` scheduling state.

    ``program_id`` is the value of ``nvext.agent_context.trajectory_id``;
    we keep ThunderAgent's "program" terminology internal because the
    scheduling primitives are paper-aligned.
    """

    program_id: str

    status: ProgramStatus = ProgramStatus.REASONING
    lifecycle: ProgramLifecycle = ProgramLifecycle.ACTIVE

    # Soft worker affinity. Cleared on hard pause; restored on resume.
    assigned_worker_id: Optional[int] = None
    origin_worker_id: Optional[int] = None

    # Real token accounting. Updated from response usage at end of each turn.
    token_total: int = 0
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0

    # KV-aware resume hint. Captured at the end of each turn so the next-turn
    # before_request can ask KvRouter.best_worker(last_prefix) for a placement
    # that lands on the warmest worker.
    last_prefix_token_ids: Optional[list[int]] = None

    step_count: int = 0
    marked_for_pause: bool = False
    soft_demoted_until: float = 0.0  # epoch seconds; >0 means priority demotion active
    waiting: Optional[asyncio.Event] = field(default=None, repr=False)

    acting_since: float = 0.0


@dataclass
class ProgramTable:
    """In-memory registry of all known programs.

    Pure data: scheduling decisions live in ``router.py``. Methods here only
    update the state machine and surface aggregates the router needs.
    """

    programs: dict[str, Program] = field(default_factory=dict)
    # program_id -> presence sentinel; the global paused queue.
    paused: dict[str, bool] = field(default_factory=dict)

    def get_or_create(self, program_id: str) -> Program:
        program = self.programs.get(program_id)
        if program is None:
            program = Program(program_id=program_id)
            self.programs[program_id] = program
        return program

    def release(self, program_id: str) -> Optional[Program]:
        program = self.programs.pop(program_id, None)
        if program is None:
            return None
        program.lifecycle = ProgramLifecycle.TERMINATED
        self.paused.pop(program_id, None)
        if program.waiting is not None:
            program.waiting.set()
            program.waiting = None
        return program

    def begin_request(
        self, program_id: str, estimated_prompt_tokens: int = 0
    ) -> Program:
        """Transition program -> REASONING; bump step_count.

        ThunderAgent updates its token estimate before admission. Dynamo already
        has preprocessed token IDs at this point, so use that exact prompt
        length instead of the upstream chars/token heuristic.
        """
        program = self.get_or_create(program_id)
        program.step_count += 1
        if estimated_prompt_tokens > 0:
            program.token_total = estimated_prompt_tokens
        program.status = ProgramStatus.REASONING
        program.acting_since = 0.0
        return program

    def end_request(
        self,
        program_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        last_prefix_token_ids: Optional[list[int]] = None,
    ) -> Optional[Program]:
        """Transition program -> ACTING; record real token accounting and the
        last-turn prefix for KV-aware resume placement."""
        program = self.programs.get(program_id)
        if program is None:
            return None
        program.last_prompt_tokens = prompt_tokens
        program.last_completion_tokens = completion_tokens
        program.token_total = prompt_tokens + completion_tokens
        if last_prefix_token_ids is not None:
            program.last_prefix_token_ids = list(last_prefix_token_ids)
        program.status = ProgramStatus.ACTING
        program.acting_since = time.time()
        return program

    def counts(self) -> dict[str, int]:
        reasoning = acting = paused = 0
        for program in self.programs.values():
            if program.lifecycle == ProgramLifecycle.PAUSED:
                paused += 1
            elif program.status == ProgramStatus.REASONING:
                reasoning += 1
            else:
                acting += 1
        return {"reasoning": reasoning, "acting": acting, "paused": paused}
