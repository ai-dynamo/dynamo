# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission policy interface and ThunderAgent's native packing policy.

Policies order candidates and authorize exceptions. The scheduler checks capacity,
selects replicas, maintains reservations, and applies lifecycle transitions.

Subclass ThunderAgentPolicy and ResumePlan to extend scheduling. Hooks run
synchronously under the scheduler lock: they must not block or perform I/O.
Exceptions propagate to the caller (or the periodic scheduler's error handler).
Observers run after lifecycle changes; raising does not undo a placement. Keep
fallible diagnostics out of these hooks unless their failures are handled there.
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
    """Host-selected replica and pre-admission budget at the plan's shared ceiling.

    ``remaining`` excludes this candidate's reservation. ``forced`` identifies a
    policy-authorized admission that bypassed normal fit; it can overcommit the
    replica. Policies observe this value but do not select or mutate placement.
    """

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
    With native_packing=False, every candidate is considered in supplied order.
    may_force_admit works in either mode. The host still reserves the candidate's
    full footprint in packing mode, including ACTING programs whose usage is weighted.
    """

    programs: list[Program]
    resume_ceiling: float
    native_packing: bool = True

    def before_attempt(self, program: Program) -> None:
        """Bookkeeping before fit; this also runs for candidates that remain paused."""

    def may_force_admit(self, program: Program) -> bool:
        """Authorize an exception after failed fit, including shortlist exclusion."""
        return False

    def record_admission(
        self, program: Program, choice: ResumeChoice, admission_rank: int
    ) -> None:
        """Observe committed placement with a zero-based rank within this pass.

        The host has resumed the program when this runs. An exception does not
        roll back that transition and interrupts the remaining admission pass.
        """


class ThunderAgentPolicy:
    """Policy extension point; native behavior is the default implementation.

    rank_initial_admission=True puts unassigned programs into the resume queue
    and restricts resume choices to currently registered endpoint workers. An
    empty liveness result defers these admissions, including discovery failures.
    False retains native eager placement and cold-start response backfill.
    retry_on_request_change enables immediate resume passes as requests change.
    force_resume_on_timeout independently enables the host's timeout escape;
    disable it when every placement must satisfy the policy's resume plan.
    mark_reasoning_for_pause controls native overload pause marking.
    """

    # Native admission exceptions and retry points. A policy can replace these
    # without replacing the scheduler's state transitions.
    retry_on_request_change = False
    force_resume_on_timeout = True
    mark_reasoning_for_pause = True

    def admission_ready(self, program_id: str) -> bool:
        """Gate admission before mutating program state; False retries later."""
        return True

    def rank_initial_admission(self) -> bool:
        """Whether unassigned programs must join the ranked resume queue."""
        return False

    def admission_queued(self, program: Program) -> None:
        """Optional policy bookkeeping when an unassigned program is queued."""

    def snapshot_admission_state(self, program: Program) -> object:
        """Capture immutable/copied policy state before an existing program's turn."""
        return None

    def restore_admission_state(self, program: Program, state: object) -> None:
        """Restore policy state after owned cancellation restores the host snapshot.

        This runs only while the same program and admission epoch still own the
        state. Newly created programs are removed instead of being restored.
        """

    def check_admission_owner(self, program, admitted, admitted_epoch) -> None:
        """Reject superseded ownership after waiting, before returning a decision.

        Raise asyncio.CancelledError to use the host's guarded cancellation rollback.
        """

    def prepare_resume(
        self, programs: list[Program], config: ThunderAgentConfig
    ) -> ResumePlan:
        """Create one pass's candidate order and ceiling without changing placement."""
        return ResumePlan(
            programs=programs,
            resume_ceiling=max(0.0, config.pause_threshold - config.resume_hysteresis),
        )
