# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure reconcile-diff for DGDC lifecycle (tracking issue #13545, item 5).
Callers perform all I/O -- fetching current state and applying the
returned actions. Identity deliberately excludes Status.Rank, which is
mutable; only a status update is possible for an identity present in both
desired and current.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


class DiffInputError(ValueError):
    """The desired or current set was malformed in a way that makes the
    diff impossible to compute safely. Per #13200's taxonomy, this is a
    TERMINAL condition -- the caller should persist a failure condition
    rather than silently skip or retry, since retrying malformed input
    won't fix it.
    """


def compute_identity(spec: Mapping[str, Any], experimental: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"spec": spec, "experimental": experimental}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DesiredCandidate:
    """One candidate the publisher has selected for materialization.
    `spec`/`experimental` are exactly MaterializationResult.dgd["spec"] and
    MaterializationResult.experimental from dgd_output_adapter.py.
    """

    spec: Mapping[str, Any]
    experimental: Mapping[str, Any] = field(default_factory=dict)
    rank: int | None = None  # None for a scalar goal's single winner

    @property
    def identity(self) -> str:
        return compute_identity(self.spec, self.experimental)


@dataclass(frozen=True)
class CurrentDGDC:
    """One DGDC observed in the cluster right now. `identity` is read back
    from wherever the create action originally recorded it (e.g. a label
    or annotation) rather than recomputed.
    """

    name: str
    identity: str
    rank: int | None


@dataclass(frozen=True)
class ReconcileActions:
    creates: tuple[DesiredCandidate, ...] = ()
    deletes: tuple[str, ...] = ()  # names
    status_updates: tuple[tuple[str, int | None], ...] = ()  # (name, new_rank)


def _reject_duplicate_identities(items, *, label: str) -> dict[str, Any]:
    """Shared duplicate check for both desired and current -- a dict
    comprehension alone would silently collapse duplicates to whichever
    entry appears last, which for `current` specifically means a
    duplicate or racing DGDC is silently dropped from every downstream
    decision: not deleted when no longer desired, not reconciled when
    still desired. Both sides of this diff get the same protection.
    """
    by_identity: dict[str, Any] = {}
    for item in items:
        if item.identity in by_identity:
            raise DiffInputError(f"duplicate identity in {label} set: {item.identity}")
        by_identity[item.identity] = item
    return by_identity


def compute_actions(
    desired: list[DesiredCandidate], current: list[CurrentDGDC]
) -> ReconcileActions:
    """Empty `desired` (no feasible candidate this round) is expected-state
    per #13200's taxonomy, not an error -- it produces delete actions for
    every current DGDC. Whether "delete everything" is the right response
    to an empty round is a policy decision belonging to the caller, not
    this function.

    Duplicate identities within either `desired` or `current` raise
    DiffInputError -- a TERMINAL condition, since the diff cannot safely
    decide which of two identical-identity entries owns a given name.
    """
    desired_by_identity = _reject_duplicate_identities(desired, label="desired")
    current_by_identity = _reject_duplicate_identities(current, label="current")

    creates = tuple(
        candidate
        for identity, candidate in desired_by_identity.items()
        if identity not in current_by_identity
    )
    deletes = tuple(
        existing.name
        for identity, existing in current_by_identity.items()
        if identity not in desired_by_identity
    )
    status_updates = tuple(
        (existing.name, desired_by_identity[identity].rank)
        for identity, existing in current_by_identity.items()
        if identity in desired_by_identity
        and existing.rank != desired_by_identity[identity].rank
    )

    return ReconcileActions(creates=creates, deletes=deletes, status_updates=status_updates)
