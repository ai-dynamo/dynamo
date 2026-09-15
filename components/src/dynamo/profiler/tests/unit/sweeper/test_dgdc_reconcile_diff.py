# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DGDC reconcile-diff algorithm (tracking issue #13545,
item 5). Pure logic, no Kubernetes client needed -- every test constructs
plain dataclasses directly."""

from __future__ import annotations

import pytest

from dynamo.profiler.sweeper.dgdc_reconcile_diff import (
    CurrentDGDC,
    DesiredCandidate,
    DiffInputError,
    compute_actions,
    compute_identity,
)

_SPEC_A = {"components": [{"name": "worker", "replicas": 2}], "backendFramework": "trtllm"}
_SPEC_B = {"components": [{"name": "worker", "replicas": 4}], "backendFramework": "trtllm"}


def test_identity_is_stable_regardless_of_key_order() -> None:
    """Same content, different dict construction order, must hash
    identically -- otherwise two equivalent candidates from different code
    paths would never diff as the same identity."""
    spec_a = {"components": [{"name": "worker"}], "backendFramework": "trtllm"}
    spec_b = {"backendFramework": "trtllm", "components": [{"name": "worker"}]}
    assert compute_identity(spec_a, {}) == compute_identity(spec_b, {})


def test_identity_changes_when_experimental_context_differs() -> None:
    """Same spec, different evaluation context (e.g. kv_load_ratio) -- the
    real Pareto-collision case from the materializer work this identity
    scheme is modeled on. Must NOT collide."""
    id_a = compute_identity(_SPEC_A, {"kv_load_ratio": 0.25})
    id_b = compute_identity(_SPEC_A, {"kv_load_ratio": 1.0})
    assert id_a != id_b


def test_new_candidate_produces_a_create_and_nothing_else() -> None:
    desired = [DesiredCandidate(spec=_SPEC_A, rank=1)]
    actions = compute_actions(desired, current=[])

    assert len(actions.creates) == 1
    assert actions.creates[0].spec == _SPEC_A
    assert actions.deletes == ()
    assert actions.status_updates == ()


def test_candidate_no_longer_selected_produces_a_delete() -> None:
    identity = compute_identity(_SPEC_A, {})
    current = [CurrentDGDC(name="cand-000", identity=identity, rank=1)]

    actions = compute_actions(desired=[], current=current)

    assert actions.deletes == ("cand-000",)
    assert actions.creates == ()
    assert actions.status_updates == ()


def test_rank_only_change_produces_a_status_update_not_delete_and_recreate() -> None:
    """The core immutability-respecting case: same content (same identity),
    different rank -- e.g. this candidate moved from rank 2 to rank 1
    because a better one above it was evicted. Must be a status update
    only. A delete+create here would violate DGDC.Spec's real, confirmed
    CEL immutability rule for no reason, since the spec never changed."""
    identity = compute_identity(_SPEC_A, {})
    current = [CurrentDGDC(name="cand-000", identity=identity, rank=2)]
    desired = [DesiredCandidate(spec=_SPEC_A, rank=1)]

    actions = compute_actions(desired, current)

    assert actions.creates == ()
    assert actions.deletes == ()
    assert actions.status_updates == (("cand-000", 1),)


def test_unchanged_candidate_produces_no_actions_at_all() -> None:
    identity = compute_identity(_SPEC_A, {})
    current = [CurrentDGDC(name="cand-000", identity=identity, rank=1)]
    desired = [DesiredCandidate(spec=_SPEC_A, rank=1)]

    actions = compute_actions(desired, current)

    assert actions == compute_actions(desired, current)
    assert actions.creates == () and actions.deletes == () and actions.status_updates == ()


def test_changed_content_at_the_same_rank_is_delete_and_create_not_update() -> None:
    """A genuinely different candidate now holds rank 1 (e.g. a later round
    found a better shape). Since identity is a content hash, this MUST
    show up as delete-old + create-new, never as an in-place spec update --
    there is no code path in this module that could even attempt an
    in-place spec mutation, which is deliberate: it can't violate the
    immutability constraint if the operation doesn't exist."""
    old_identity = compute_identity(_SPEC_A, {})
    current = [CurrentDGDC(name="cand-000", identity=old_identity, rank=1)]
    desired = [DesiredCandidate(spec=_SPEC_B, rank=1)]

    actions = compute_actions(desired, current)

    assert actions.deletes == ("cand-000",)
    assert len(actions.creates) == 1
    assert actions.creates[0].spec == _SPEC_B
    assert actions.status_updates == ()


def test_empty_desired_set_is_not_an_error_and_deletes_everything_current() -> None:
    """No feasible candidate this round is expected-state per #13200's
    taxonomy, not an exception -- matches how the renderer layer already
    treats this outcome."""
    identity = compute_identity(_SPEC_A, {})
    current = [CurrentDGDC(name="cand-000", identity=identity, rank=1)]

    actions = compute_actions(desired=[], current=current)

    assert actions.deletes == ("cand-000",)
    assert actions.creates == ()


def test_duplicate_identity_in_desired_set_raises_terminal_error() -> None:
    desired = [
        DesiredCandidate(spec=_SPEC_A, rank=1),
        DesiredCandidate(spec=_SPEC_A, rank=2),
    ]
    with pytest.raises(DiffInputError, match="duplicate candidate identity"):
        compute_actions(desired, current=[])


def test_pareto_front_reordering_produces_only_status_updates() -> None:
    """Realistic Pareto scenario: three unchanged candidates get reshuffled
    ranks after a round finds a fourth candidate that displaces the
    previous rank 1. All three survivors should get status updates only;
    none should be recreated."""
    identities = [compute_identity({"components": [{"replicas": n}]}, {}) for n in (2, 4, 8)]
    current = [
        CurrentDGDC(name=f"cand-{i:03d}", identity=identity, rank=i + 1)
        for i, identity in enumerate(identities)
    ]
    # Same three candidates, ranks rotated: old rank 1 -> 2, 2 -> 3, 3 -> 1
    desired = [
        DesiredCandidate(spec={"components": [{"replicas": 8}]}, rank=1),
        DesiredCandidate(spec={"components": [{"replicas": 2}]}, rank=2),
        DesiredCandidate(spec={"components": [{"replicas": 4}]}, rank=3),
    ]

    actions = compute_actions(desired, current)

    assert actions.creates == ()
    assert actions.deletes == ()
    assert set(actions.status_updates) == {
        ("cand-000", 2),
        ("cand-001", 3),
        ("cand-002", 1),
    }
