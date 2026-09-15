# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile-diff algorithm for DGDC lifecycle (tracking issue #13545, item 5:
"materialize DGDCs, and reconcile their lifecycle").

Pure logic only. Deliberately has no Kubernetes client dependency -- per
#12625's reviewed guideline ("use concrete nested reconcilers with explicit
dependencies; programs must not retain or call an all-capable
...Reconciler"), this is a narrow, single-purpose component: given a desired
set and a current set, decide what to create/delete/update. Fetching the
current set and applying the decided actions are the caller's job, injected
as real Kubernetes clients once the v1beta2 CRDs are served (still unserved
as of #13603/#13744 -- both explicit: "Keep DGDR v1beta2 unserved until
controller support is added separately").

Identity design: a DesiredCandidate's identity is a content hash of its
rendered spec + relevant evaluation-context fields, NOT its rank. Rank
(DynamoGraphDeploymentCandidateStatus.Rank, confirmed via the real v1beta2
Go type: "one-based scalar ordering... absent for Pareto searches") changes
over time as better candidates are found and worse ones evicted, with the
candidate's own content often unchanged -- using rank as identity would
force a spurious delete+recreate on every reshuffle. Using a content hash
instead means "the content changed" and "the identity changed" are the same
event: a changed candidate naturally shows up as one identity disappearing
(-> delete) and a new one appearing (-> create), with no special "update
spec" case needed at all -- which matches DGDC.Spec's real, confirmed
immutability constraint (CEL rule: "spec is immutable") exactly, rather than
fighting it. Only Status fields (rank, conditions) can differ for the same
identity, matching what's actually mutable in the real schema.

Error taxonomy follows #13200's documented convention for this codebase's
reconcilers: ordinary (retryable, not persisted as a failure), terminal
(persist a failure condition), expected-state (not an error at all). See
each function's docstring for how it applies here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


class DiffInputError(ValueError):
    """The desired or current set was malformed in a way that makes the
    diff impossible to compute safely. Per #13200's taxonomy, this is a
    TERMINAL condition for whichever candidate/DGDC triggered it -- the
    caller should persist a failure condition on the owning DGDRRun rather
    than silently skip or retry, since retrying malformed input won't fix
    it. This is deliberately NOT raised for "no candidates yet" (that is
    expected-state -- see compute_actions' docstring), only for genuinely
    malformed entries.
    """


def compute_identity(spec: Mapping[str, Any], experimental: Mapping[str, Any]) -> str:
    """Stable content identity for one candidate: sha256 of its rendered
    spec + evaluation-context fields, canonicalized (sorted keys, no
    whitespace) so semantically-identical dicts always hash identically
    regardless of key ordering. Deliberately excludes rank -- see module
    docstring for why rank must never be part of identity.
    """
    canonical = json.dumps(
        {"spec": spec, "experimental": experimental}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DesiredCandidate:
    """One candidate the publisher has selected for materialization (the
    bounded top-K/Pareto-front output of item 5's selector, not built yet).
    `spec`/`experimental` are exactly MaterializationResult.dgd["spec"] and
    MaterializationResult.experimental from dgd_output_adapter.py -- this
    module consumes that output directly, not a redefinition of it.
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
    from wherever the create action originally recorded it (e.g. a label or
    annotation on the real object -- an real cluster client's job, not
    this module's) rather than recomputed, so a current object never needs
    its full spec re-fetched just to diff against desired state.
    """

    name: str
    identity: str
    rank: int | None


@dataclass(frozen=True)
class ReconcileActions:
    """What the caller should do. `creates` carry the full DesiredCandidate
    (a real client needs the spec to create the object). `deletes` and
    `status_updates` only need the name of an existing object -- the diff
    never needs a current object's full spec, only its recorded identity.
    """

    creates: tuple[DesiredCandidate, ...] = ()
    deletes: tuple[str, ...] = ()  # names
    status_updates: tuple[tuple[str, int | None], ...] = ()  # (name, new_rank)


def compute_actions(
    desired: list[DesiredCandidate], current: list[CurrentDGDC]
) -> ReconcileActions:
    """The diff itself. Pure function: same inputs always produce the same
    actions, no I/O, no side effects -- deliberately trivial to unit test
    without a Kubernetes client or any mocking beyond plain dataclasses.

    Empty `desired` (no feasible candidate this round) is NOT an error --
    per #13200's taxonomy this is expected-state, matching how the
    materializer/renderer layer already treats "no feasible candidate" as
    a normal outcome, not an exception. It produces delete actions for
    every current DGDC and nothing else; it is the caller's job to decide
    whether "delete everything" is actually the right response to an empty
    round (e.g. vs. "keep the last known-good state and just mark it
    stale") -- that is a policy decision belonging to the publisher, not
    this pure diff function.

    Duplicate identities within `desired` (the same content hash appearing
    twice, e.g. from a selector bug) raise DiffInputError -- a TERMINAL
    condition per the module docstring, since the diff cannot safely decide
    which of two identical-identity entries "owns" a given name.
    """
    seen: set[str] = set()
    for candidate in desired:
        if candidate.identity in seen:
            raise DiffInputError(
                f"duplicate candidate identity in desired set: {candidate.identity}"
            )
        seen.add(candidate.identity)

    desired_by_identity = {c.identity: c for c in desired}
    current_by_identity = {c.identity: c for c in current}

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
