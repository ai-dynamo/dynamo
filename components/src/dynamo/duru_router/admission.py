# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duru decisions from explicit inputs, independent of routers and engine APIs.

Hosts own capacity checks, placement, freshness, observations and accounting.
Duru supplies queue order, admission ceilings and alpha-exception eligibility.
"""

from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple


def valid_score(value):
    return isinstance(value, (int, float)) and isfinite(value) and value >= 0


def priority_key(score, arrival, fallback_index):
    """Descending valid scores, then arrival; invalid scores keep input order."""
    if not valid_score(score):
        return (1, 0.0, 0.0, fallback_index)
    return (0, -score, arrival, fallback_index)


class Candidate(NamedTuple):
    identity: str
    demand: int
    score: float | None = None
    measured_u: float | None = None
    arrival: float = 0.0
    known: bool = True
    idle: bool = False
    pending_request: bool = False
    first_admission: bool = False

    @property
    def effective_score(self):
        # Idle agents have measured work, but no pending-request lookahead.
        return (
            self.measured_u if self.idle and not valid_score(self.score) else self.score
        )


@dataclass(frozen=True, slots=True)
class DuruPolicy:
    resume_ceiling: float = 1.0

    def order(self, candidates: list[Candidate]) -> list[Candidate]:
        return [
            candidate
            for _, candidate in sorted(
                ((i, c) for i, c in enumerate(candidates) if c.known),
                key=lambda pair: priority_key(
                    pair[1].effective_score, pair[1].arrival, pair[0]
                ),
            )
        ]

    def ceiling(self, candidate: Candidate, established_waiters: bool) -> float:
        if candidate.first_admission and not established_waiters:
            return 1.0
        return max(0.0, min(1.0, self.resume_ceiling))

    def may_force_admit(self, candidate: Candidate) -> bool:
        # This policy only admits requests that pass the host capacity check.
        return False
