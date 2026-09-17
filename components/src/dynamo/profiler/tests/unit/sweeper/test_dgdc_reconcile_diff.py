# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    spec_a = {"components": [{"name": "worker"}], "backendFramework": "trtllm"}
    spec_b = {"backendFramework": "trtllm", "components": [{"name": "worker"}]}
    assert compute_identity(spec_a, {}) == compute_identity(spec_b, {})


def test_identity_changes_when_experimental_context_differs() -> None:
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
    old_identity = compute_identity(_SPEC_A, {})
    current = [CurrentDGDC(name="cand-000", identity=old_identity, rank=1)]
    desired = [DesiredCandidate(spec=_SPEC_B, rank=1)]

    actions = compute_actions(desired, current)

    assert actions.deletes == ("cand-000",)
    assert len(actions.creates) == 1
    assert actions.creates[0].spec == _SPEC_B
    assert actions.status_updates == ()


def test_duplicate_identity_in_desired_set_raises_terminal_error() -> None:
    desired = [
        DesiredCandidate(spec=_SPEC_A, rank=1),
        DesiredCandidate(spec=_SPEC_A, rank=2),
    ]
    with pytest.raises(DiffInputError, match="duplicate identity in desired"):
        compute_actions(desired, current=[])


def test_duplicate_identity_in_current_with_empty_desired_raises_not_silently_drops() -> None:
    """Regression test: two current DGDCs sharing an identity, with an
    empty desired set, must not silently collapse to a single delete via
    dict-comprehension overwrite -- the surplus object would be dropped
    from every downstream decision and left orphaned in the cluster
    forever, never deleted."""
    identity = compute_identity(_SPEC_A, {})
    current = [
        CurrentDGDC(name="cand-000", identity=identity, rank=1),
        CurrentDGDC(name="cand-001", identity=identity, rank=1),
    ]
    with pytest.raises(DiffInputError, match="duplicate identity in current"):
        compute_actions(desired=[], current=current)


def test_duplicate_identity_in_current_still_desired_raises_not_silently_drops() -> None:
    """Regression test: same duplicate-current bug, the other affected
    path -- when the identity IS still desired, a silent dict-overwrite
    would leave one duplicate receiving a status update while the other
    is never reconciled at all, despite the reconciler claiming to
    reconcile the full current set."""
    identity = compute_identity(_SPEC_A, {})
    current = [
        CurrentDGDC(name="cand-000", identity=identity, rank=2),
        CurrentDGDC(name="cand-001", identity=identity, rank=3),
    ]
    desired = [DesiredCandidate(spec=_SPEC_A, rank=1)]
    with pytest.raises(DiffInputError, match="duplicate identity in current"):
        compute_actions(desired, current)


def test_rank_reordering_across_multiple_entries_produces_only_status_updates() -> None:
    """Generic multi-entry rank-change coverage for the diff algorithm
    itself, which is agnostic to what rank means -- it only detects
    whether the value differs. NOT framed as a "Pareto" scenario:
    Status.Rank's real, confirmed comment is "the one-based scalar
    ordering and is absent for Pareto searches" -- real Pareto candidates
    never carry a non-None, reshuffling rank like this at all. See
    test_pareto_candidates_never_produce_rank_status_updates below for
    what realistic Pareto data actually looks like.
    """
    identities = [compute_identity({"components": [{"replicas": n}]}, {}) for n in (2, 4, 8)]
    current = [
        CurrentDGDC(name=f"cand-{i:03d}", identity=identity, rank=i + 1)
        for i, identity in enumerate(identities)
    ]
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


def test_pareto_candidates_never_produce_rank_status_updates() -> None:
    """Confirmed against the real v1beta2 Go type: Status.Rank is "the
    one-based scalar ordering and is absent for Pareto searches" -- every
    real Pareto candidate leaves rank unset (None), regardless of front
    size or how membership changes. Since rank never varies on either
    side for realistic Pareto data, an unchanged front must produce zero
    status_updates, not the numbered reshuffling the test above exercises
    for the algorithm's generic case."""
    identities = [compute_identity({"components": [{"replicas": n}]}, {}) for n in (2, 4, 8)]
    current = [
        CurrentDGDC(name=f"cand-{i:03d}", identity=identity, rank=None)
        for i, identity in enumerate(identities)
    ]
    desired = [
        DesiredCandidate(spec={"components": [{"replicas": n}]}, rank=None) for n in (2, 4, 8)
    ]

    actions = compute_actions(desired, current)

    assert actions.creates == ()
    assert actions.deletes == ()
    assert actions.status_updates == ()
