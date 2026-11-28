# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission policy interface and ThunderAgent's native packing policy.

Policies order candidates and authorize exceptions. The scheduler checks capacity,
selects replicas, maintains reservations, and applies lifecycle transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from dynamo.thunderagent_router.program_state import Program, ReplicaKey

if TYPE_CHECKING:
    from dynamo.thunderagent_router.router import ThunderAgentConfig


@dataclass
class ReplicaUsage:
    """Occupancy bucket maintained by the scheduler for one admission pass."""

    capacity: int
    used: int = 0
    programs: list[Program] = field(default_factory=list)


class ResumeChoice(NamedTuple):
    replica: ReplicaKey
    remaining: int
    ceiling: float
    forced: bool = False


@dataclass(kw_only=True)
class ResumePlan:
    """One ordered queue and its policy hooks; contains no placement decisions.

    All candidates share resume_ceiling. Native packing tries its capacity
    shortlist in decreasing footprint order,
    then considers excluded programs in policy order for forced admission only.
    """

    programs: list[Program]
    resume_ceiling: float
    native_packing: bool = True

    def before_attempt(self, program: Program) -> None:
        """Optional policy bookkeeping before the host checks capacity."""

    def may_force_admit(self, program: Program) -> bool:
        """Authorize an exception after failed fit, including shortlist exclusion."""
        return False

    def record_admission(
        self, program: Program, choice: ResumeChoice, admission_rank: int
    ) -> None:
        """Optional observation of a host decision; does not select placement."""


class ThunderAgentPolicy:
    # Native admission exceptions and retry points. A policy can replace these
    # without replacing the scheduler's state transitions.
    retry_on_request_change = False
    force_resume_on_timeout = True
    mark_reasoning_for_pause = True

    def admission_ready(self, program_id: str) -> bool:
        return True

    def rank_initial_admission(self) -> bool:
        """Whether unassigned programs must join the ranked resume queue."""
        return False

    def admission_queued(self, program: Program) -> None:
        """Optional policy bookkeeping when an unassigned program is queued."""

    def check_admission_owner(self, program, admitted, admitted_epoch) -> None:
        """A policy may reject an admission superseded while it was waiting."""

    def prepare_resume(
        self, programs: list[Program], config: ThunderAgentConfig
    ) -> ResumePlan:
        return ResumePlan(
            programs=programs,
            resume_ceiling=max(0.0, config.pause_threshold - config.resume_hysteresis),
        )
