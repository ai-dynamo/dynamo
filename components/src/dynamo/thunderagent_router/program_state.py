# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Program lifecycle data model.

Mirrors ``ThunderAgent/program/state.py`` semantics with one intentional
difference for v0: ``token_total`` comes from real
``prompt_tokens + completion_tokens`` reported on the chat-completions
response, not a ``chars / 5`` heuristic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProgramStatus(Enum):
    """REASONING: request on GPU. ACTING: between turns (tools/wait).

    ``acting_since`` carries a ``time.monotonic()`` stamp; never compare it
    against ``time.time()``.
    """

    REASONING = "reasoning"
    ACTING = "acting"


class ProgramLifecycle(Enum):
    """ACTIVE: admissible. PAUSED: blocked on ``waiting``. TERMINATED: cleaned."""

    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class Program:
    """Per-``program_id`` scheduling state. ``program_id`` is
    ``nvext.agent_context.trajectory_id``."""

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

    acting_since: float = 0.0


@dataclass
class ProgramTable:
    """In-memory registry of all known programs.

    Pure data: scheduling decisions live in ``router.py``.
    """

    programs: dict[str, Program] = field(default_factory=dict)
    # Used as an insertion-ordered set; values are never read.
    paused: dict[str, None] = field(default_factory=dict)

    def begin_request(
        self, program_id: str, estimated_prompt_tokens: int = 0
    ) -> Program:
        """Transition program -> REASONING; bump step_count."""
        program = self.programs.get(program_id)
        if program is None:
            program = Program(program_id=program_id)
            self.programs[program_id] = program
        program.step_count += 1
        if estimated_prompt_tokens > 0:
            program.token_total = estimated_prompt_tokens
        program.status = ProgramStatus.REASONING
        program.acting_since = 0.0
        return program

    def end_request(
        self, program_id: str, prompt_tokens: int, completion_tokens: int
    ) -> Optional[Program]:
        """Transition program -> ACTING and record real token accounting."""
        program = self.programs.get(program_id)
        if program is None:
            return None
        program.token_total = prompt_tokens + completion_tokens
        program.status = ProgramStatus.ACTING
        program.acting_since = time.monotonic()
        return program
