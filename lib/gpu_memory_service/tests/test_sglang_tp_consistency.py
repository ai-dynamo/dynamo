# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.sglang.tp_consistency import (
    GmsTPConsistencyError,
    TPConsistency,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
]


def test_asymmetric_empty_directory_view_selects_common_miss(monkeypatch):
    cohort = TPConsistency(world_size=2)
    monkeypatch.setattr(
        cohort, "_gather", lambda _: [("lookup", [(4, 7)]), ("lookup", [])]
    )
    assert cohort.common_prefix("lookup", [(4, 7)]) == []


def test_asymmetric_short_directory_view_uses_only_identical_prefix(monkeypatch):
    cohort = TPConsistency(world_size=2)
    monkeypatch.setattr(
        cohort,
        "_gather",
        lambda _: [("lookup", [(4, 7), (5, 8)]), ("lookup", [(4, 7), (6, 8)])],
    )
    assert cohort.common_prefix("lookup", [(4, 7), (5, 8)]) == [(4, 7)]


def test_lookup_and_common_prefix_share_one_collective(monkeypatch):
    cohort = TPConsistency(world_size=2)
    gathers = []

    def gather(value):
        gathers.append(value)
        return [value, ("lookup", True, [4])]

    monkeypatch.setattr(cohort, "_gather", gather)
    value, common = cohort.run_common_prefix("lookup", lambda: ("local-entry", [4, 5]))

    assert value == "local-entry"
    assert common == [4]
    assert len(gathers) == 1


def test_lookup_failure_is_voted_before_becoming_fatal(monkeypatch):
    cohort = TPConsistency(world_size=2)
    votes = []

    def gather(value):
        votes.append(value)
        return [value, ("lookup", True, [])]

    monkeypatch.setattr(cohort, "_gather", gather)
    with pytest.raises(GmsTPConsistencyError, match="lookup"):
        cohort.run_common_prefix(
            "lookup", lambda: (_ for _ in ()).throw(ValueError("directory"))
        )
    assert votes == [("lookup", False, [])]


def test_leader_true_broadcasts_leader_candidate(monkeypatch):
    import torch.distributed as dist

    cohort = TPConsistency(world_size=2)
    calls = []
    monkeypatch.setattr(cohort, "_rank", lambda: 1)

    def broadcast(vote, *, src, group):
        calls.append((src, group, bool(vote.item())))
        vote.fill_(1)

    monkeypatch.setattr(dist, "broadcast", broadcast)

    assert cohort.leader_true("candidate", False) is True
    assert calls == [(0, None, False)]


def test_leader_true_collective_failure_is_fatal(monkeypatch):
    import torch.distributed as dist

    cohort = TPConsistency(world_size=2)
    monkeypatch.setattr(cohort, "_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "broadcast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("peer")),
    )

    with pytest.raises(GmsTPConsistencyError, match="candidate"):
        cohort.leader_true("candidate", True)


def test_digest_transaction_uses_one_compact_tensor_collective(monkeypatch):
    import torch.distributed as dist

    cohort = TPConsistency(world_size=2)
    calls = []

    def all_gather(outputs, vote, *, group):
        calls.append((vote.numel(), group))
        for output in outputs:
            output.copy_(vote)

    monkeypatch.setattr(dist, "all_gather", all_gather)
    assert cohort.transact_digest("publish", b"layout", lambda: 7) == 7
    assert calls == [(33, None)]


def test_digest_transaction_rejects_peer_failure(monkeypatch):
    import torch.distributed as dist

    cohort = TPConsistency(world_size=2)

    def all_gather(outputs, vote, *, group):
        for output in outputs:
            output.copy_(vote)
        outputs[1][0] = 0

    monkeypatch.setattr(dist, "all_gather", all_gather)
    with pytest.raises(GmsTPConsistencyError, match="transaction failed"):
        cohort.transact_digest("publish", b"layout", lambda: None)


def test_digest_transaction_rejects_divergent_layout(monkeypatch):
    import torch.distributed as dist

    cohort = TPConsistency(world_size=2)

    def all_gather(outputs, vote, *, group):
        for output in outputs:
            output.copy_(vote)
        outputs[1][-1] ^= 1

    monkeypatch.setattr(dist, "all_gather", all_gather)
    with pytest.raises(GmsTPConsistencyError, match="disagreement"):
        cohort.transact_digest("publish", b"layout", lambda: None)


def test_one_rank_failure_prevents_successful_rank_from_proceeding(monkeypatch):
    cohort = TPConsistency(world_size=2)
    monkeypatch.setattr(
        cohort, "_gather", lambda _: [("adopt", True), ("adopt", False)]
    )
    with pytest.raises(GmsTPConsistencyError, match="adopt"):
        cohort.run("adopt", lambda: [4])


def test_local_failure_is_voted_before_becoming_fatal(monkeypatch):
    cohort = TPConsistency(world_size=2)
    votes = []

    def gather(value):
        votes.append(value)
        return [value, ("adopt", True)]

    monkeypatch.setattr(cohort, "_gather", gather)
    with pytest.raises(GmsTPConsistencyError):
        cohort.run("adopt", lambda: (_ for _ in ()).throw(ValueError("lease")))
    assert votes == [("adopt", False)]


def test_divergent_pressure_page_selection_fails_closed(monkeypatch):
    cohort = TPConsistency(world_size=2)
    monkeypatch.setattr(
        cohort, "_gather", lambda _: [("reserve", [4]), ("reserve", [5])]
    )
    with pytest.raises(GmsTPConsistencyError, match="reserve"):
        cohort.agree("reserve", [4])


def test_collective_failure_is_fatal_not_a_local_cache_miss(monkeypatch):
    import torch.distributed as dist

    def broken_collective(*_args, **_kwargs):
        raise RuntimeError("peer unavailable")

    monkeypatch.setattr(dist, "all_gather_object", broken_collective)
    with pytest.raises(GmsTPConsistencyError, match="channel failed"):
        TPConsistency(world_size=2).common_prefix("lookup", [(4, 7)])


def test_tp1_keeps_direct_local_operation_semantics():
    cohort = TPConsistency()
    cohort.agree("reserve", [4])
    assert cohort.common_prefix("lookup", [(4, 7)]) == [(4, 7)]
    assert cohort.run("adopt", lambda: 7) == 7
    with pytest.raises(ValueError):
        cohort.run("adopt", lambda: (_ for _ in ()).throw(ValueError("lease")))


def _allocator(monkeypatch, peer_failure=None):
    monkeypatch.setenv("GMS_SGLANG_TP_LEASE_WINDOW_PAGES", "1")
    import torch
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease
    from gpu_memory_service.integrations.sglang import install_kv_leases as hooks

    cohort = TPConsistency(world_size=2)

    def gather(value):
        stage = value[0]
        if stage.endswith(":vote") and peer_failure in ("reserve", "pages"):
            return [value, (stage, False)]
        return [value, value]

    monkeypatch.setattr(cohort, "_gather", gather)
    monkeypatch.setattr(cohort, "_rank", lambda: 0)
    released = []
    client = SimpleNamespace(
        acquire=lambda *_args, **_kwargs: [KVLease(4, 2)],
        release=lambda leases: released.extend(leases),
    )
    allocator = SimpleNamespace(
        free_pages=torch.tensor([4, 5]), _gms_tp_consistency=cohort
    )
    state = {"client": client, "leases_by_page": {}, "retained_pages": set()}
    monkeypatch.setattr(hooks, "torch", torch)
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    return hooks, allocator, state, released


def test_tp_reservation_window_amortizes_successful_collectives(monkeypatch):
    import torch
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease
    from gpu_memory_service.integrations.sglang import install_kv_leases as hooks

    monkeypatch.setenv("GMS_SGLANG_TP_LEASE_WINDOW_PAGES", "3")
    cohort = TPConsistency(world_size=2)
    gathers = []

    def gather(value):
        gathers.append(value)
        return [value, value]

    monkeypatch.setattr(cohort, "_gather", gather)
    monkeypatch.setattr(cohort, "_rank", lambda: 0)
    client = SimpleNamespace(
        acquire=lambda count, *, preferred_blocks, strict_preferred: [
            KVLease(page, 1) for page in preferred_blocks[:count]
        ],
        release=lambda _leases: None,
    )
    allocator = SimpleNamespace(
        free_pages=torch.tensor([3, 1, 2, 4, 5, 6]), _gms_tp_consistency=cohort
    )
    state = {"client": client, "leases_by_page": {}, "retained_pages": set()}
    monkeypatch.setattr(hooks, "torch", torch)
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    first = hooks._reserve_pages(allocator, [3], local_free=6, operation="test")
    first_gathers = len(gathers)
    allocator.free_pages = allocator.free_pages[1:]
    second = hooks._reserve_pages(allocator, [1], local_free=5, operation="test")

    assert [lease.block_id for lease in first] == [3]
    assert [lease.block_id for lease in second] == [1]
    assert first_gathers > 0
    assert len(gathers) == first_gathers
    assert state["tp_reserved_pages"] == [2]


def test_native_capacity_agreement_is_only_on_exhaustion(monkeypatch):
    from gpu_memory_service.integrations.sglang import install_kv_leases as hooks

    cohort = TPConsistency(world_size=2)
    gathers = []
    monkeypatch.setattr(
        cohort, "_gather", lambda value: gathers.append(value) or [value, value]
    )
    allocator = SimpleNamespace(_gms_tp_consistency=cohort)

    hooks._agree_native_capacity(allocator, "alloc", 2, 3)
    assert gathers == []
    hooks._agree_native_capacity(allocator, "alloc", 4, 3)
    assert len(gathers) == 1


@pytest.mark.parametrize("failure", ["reserve", "pages"])
def test_tp_reservation_failure_releases_only_new_leases_before_native_use(
    monkeypatch, failure
):
    hooks, allocator, state, released = _allocator(monkeypatch, failure)
    assert hooks._reserve_pages(allocator, [4], local_free=2, operation="test") is None
    assert allocator.free_pages.tolist() == [4, 5]
    assert state["leases_by_page"] == {}
    assert [lease.block_id for lease in released] == [
        4
    ] * hooks._TP_RESERVATION_ATTEMPTS


def test_tp_reservation_agrees_before_recording_pages(monkeypatch):
    hooks, allocator, state, released = _allocator(monkeypatch)
    leases = hooks._reserve_pages(allocator, [4], local_free=2, operation="test")
    assert [lease.block_id for lease in leases] == [4]
    assert list(state["leases_by_page"]) == [4]
    assert released == []


@pytest.mark.parametrize("failure", ["peer-vote", "local-release"])
def test_tp_failed_rollback_never_releases_the_same_batch_twice(monkeypatch, failure):
    hooks, allocator, state, released = _allocator(monkeypatch, "reserve")
    cohort = allocator._gms_tp_consistency
    original_gather = cohort._gather

    def gather(value):
        stage = value[0]
        if stage.endswith(":rollback") and failure == "peer-vote":
            return [value, (stage, False)]
        return original_gather(value)

    monkeypatch.setattr(cohort, "_gather", gather)
    if failure == "local-release":

        def ambiguous_release(leases):
            released.extend(leases)
            raise RuntimeError("release outcome unknown")

        state["client"].release = ambiguous_release

    with pytest.raises(GmsTPConsistencyError, match="rollback"):
        hooks._reserve_pages(allocator, [4], local_free=2, operation="test")
    assert [lease.block_id for lease in released] == [4]
    assert state["leases_by_page"] == {}
    assert allocator.free_pages.tolist() == [4, 5]


def _pressure_allocator(monkeypatch, *, rank=0, failure=None):
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease

    hooks, allocator, state, released = _allocator(monkeypatch)
    cohort = allocator._gms_tp_consistency
    content_hash = b"h" * 32
    victims = [
        {
            "content_hash": content_hash,
            "engine_id": "engine-0",
            "slot_ids": [4],
            "generations": [2],
        }
    ]
    selected = []
    pinned = []
    directory_entries = {}

    def select(count, *, eligible_slot_ids):
        selected.append((count, eligible_slot_ids))
        if failure == "select":
            raise RuntimeError("directory unavailable")
        return victims

    def adopt(leases):
        pinned.extend(leases)
        if failure == "ring_generation":
            return []
        return [KVLease(lease.block_id, lease.generation + 1) for lease in leases]

    def gather(value):
        stage = value[0]
        if stage == "pressure:victims" and failure == "victim_disagreement":
            divergent = [(b"x" * 32, "engine-0", (4,), (2,))]
            return [value, (stage, divergent)]
        if stage == "pressure:eligible":
            return [value, (stage, [4])]
        if stage == "pressure:validate" and failure == "peer_validation":
            return [value, (stage, False)]
        return [value, value]

    def publish(items):
        for item in items:
            directory_entries[item["content_hash"]] = {
                "state": "ready",
                "tier": "hbm",
                "engine_id": item["engine_id"],
                "slot_ids": item["slot_ids"],
                "generations": item["generations"],
            }
        return len(items)

    monkeypatch.setattr(cohort, "_gather", gather)
    monkeypatch.setattr(cohort, "_rank", lambda: rank)
    state["client"].free_count = lambda: 0
    state["client"].adopt = adopt
    state["leases_by_page"][4] = KVLease(4, 9 if failure == "generation" else 2)
    state["retained_pages"].add(4)
    allocator._gms_kv_directory = SimpleNamespace(
        authoritative=True,
        ensure_hbm_capacity=select,
        publish=publish,
        lookup_authoritative=lambda hashes: [
            directory_entries.get(content_hash) for content_hash in hashes
        ],
    )
    return hooks, allocator, state, released, selected, pinned


@pytest.mark.parametrize("rank", [0, 1])
def test_tp_reclaim_uses_same_leader_victim_and_common_native_free_set(
    monkeypatch, rank
):
    hooks, allocator, state, released, selected, pinned = _pressure_allocator(
        monkeypatch, rank=rank
    )
    assert hooks._ensure_directory_capacity(allocator, 1) == 1
    assert selected == [(1, [4])]
    assert [(lease.block_id, lease.generation) for lease in pinned] == [(4, 2)]
    assert [(lease.block_id, lease.generation) for lease in released] == [(4, 3)]
    assert state["leases_by_page"] == {}
    assert state["retained_pages"] == set()


@pytest.mark.parametrize(
    "failure",
    [
        "peer_validation",
        "generation",
        "select",
        "victim_disagreement",
        "ring_generation",
    ],
)
def test_tp_reclaim_failure_never_releases_any_page(monkeypatch, failure):
    hooks, allocator, state, released, _selected, _pinned = _pressure_allocator(
        monkeypatch, failure=failure
    )
    with pytest.raises(GmsTPConsistencyError):
        hooks._ensure_directory_capacity(allocator, 1)
    assert released == []
    assert 4 in state["leases_by_page"]
    assert 4 in state["retained_pages"]


@pytest.mark.parametrize("world_size", [1, 2])
def test_pressure_never_retires_retained_pages_still_in_native_tree(
    monkeypatch, world_size
):
    import torch

    hooks, allocator, _state, released, selected, _pinned = _pressure_allocator(
        monkeypatch
    )
    allocator.free_pages = torch.tensor([], dtype=torch.int64)
    if world_size == 1:
        allocator._gms_tp_consistency = TPConsistency()
    assert hooks._ensure_directory_capacity(allocator, 1) == 0
    assert selected == []
    assert released == []


def test_tp_reservation_retries_after_group_reclaim(monkeypatch):
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease

    hooks, allocator, state, released, selected, _pinned = _pressure_allocator(
        monkeypatch
    )
    attempts = []

    def acquire(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("capacity")
        return [KVLease(4, 4)]

    state["client"].acquire = acquire
    leases = hooks._reserve_tp_pages(allocator, [4], "test")
    assert len(attempts) == 2
    assert selected == [(1, [4])]
    assert [(lease.block_id, lease.generation) for lease in released] == [(4, 3)]
    assert leases == [KVLease(4, 4)]
    assert state["leases_by_page"] == {4: KVLease(4, 4)}


@pytest.mark.parametrize("reclaim_available", [False, True])
def test_uniform_capacity_exhaustion_returns_native_backpressure(
    monkeypatch, reclaim_available
):
    hooks, allocator, state, released, selected, _pinned = _pressure_allocator(
        monkeypatch
    )
    attempts = []

    def exhausted(*_args, **_kwargs):
        attempts.append(True)
        raise RuntimeError("capacity")

    state["client"].acquire = exhausted
    if not reclaim_available:
        allocator._gms_kv_directory.ensure_hbm_capacity = lambda *_args, **_kwargs: []
    assert hooks._reserve_tp_pages(allocator, [4], "test") is None
    assert len(attempts) == hooks._TP_RESERVATION_ATTEMPTS
    assert allocator.free_pages.tolist() == [4, 5]
    assert [(lease.block_id, lease.generation) for lease in released] == (
        [(4, 3)] if reclaim_available else []
    )
