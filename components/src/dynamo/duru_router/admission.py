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
    alpha: float | None = None
    resume_ceiling: float = 1.0

    def __post_init__(self):
        if self.alpha is not None and (not isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("Admission alpha must be finite and positive")

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

    def may_force_admit(self, candidate: Candidate) -> bool:
        return bool(
            self.alpha is not None
            and candidate.pending_request
            and valid_score(candidate.score)
            and candidate.score >= self.alpha
        )
