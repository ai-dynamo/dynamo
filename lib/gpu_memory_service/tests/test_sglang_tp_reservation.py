# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.sglang import install_kv_leases as hooks
from gpu_memory_service.integrations.sglang.tp_consistency import TPConsistency

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
]


class _Votes:
    """Execute the real agreement protocol on two simulated scheduler ranks."""

    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=5)
        self.values = [None, None]

    def gather(self, rank, value):
        self.values[rank] = value
        self.barrier.wait()
        values = list(self.values)
        self.barrier.wait()
        return values


class _Ring:
    def __init__(self, generation):
        self.lock = threading.Lock()
        self.generations = {page: generation for page in range(1, 5)}
        self.held = {}
        self.released = []

    def acquire(self, owner, count, preferred_blocks, strict_preferred):
        assert strict_preferred, "peers must never independently pick fallback IDs"
        with self.lock:
            pages = [page for page in preferred_blocks if page not in self.held][:count]
            if len(pages) != count:
                raise RuntimeError("contention")
            leases = []
            for page in pages:
                self.generations[page] += 1
                lease = KVLease(page, self.generations[page])
                self.held[page] = (owner, lease)
                leases.append(lease)
            return leases

    def release(self, owner, leases):
        with self.lock:
            for lease in leases:
                assert self.held[lease.block_id] == (owner, lease)
                del self.held[lease.block_id]
                self.released.append(lease)


def _cohort_allocators(
    monkeypatch, rings, owner, before_acquire=None, after_acquire=None
):
    monkeypatch.setenv("GMS_SGLANG_TP_LEASE_WINDOW_PAGES", "1")
    votes = _Votes()
    allocators = []
    for rank, ring in enumerate(rings):
        cohort = TPConsistency(world_size=2)
        cohort._rank = lambda rank=rank: rank
        cohort._gather = lambda value, rank=rank: votes.gather(rank, value)

        def acquire(count, *, preferred_blocks, strict_preferred, rank=rank, ring=ring):
            if before_acquire is not None:
                before_acquire(owner, rank, preferred_blocks)
            leases = ring.acquire(owner, count, preferred_blocks, strict_preferred)
            if after_acquire is not None:
                after_acquire(owner, rank, leases)
            return leases

        client = SimpleNamespace(
            acquire=acquire,
            release=lambda leases, ring=ring: ring.release(owner, leases),
        )
        allocator = SimpleNamespace(
            free_pages=torch.tensor([1, 2, 3, 4]), _gms_tp_consistency=cohort
        )
        monkeypatch.setitem(
            hooks._STATE,
            id(allocator),
            {
                "client": client,
                "leases_by_page": {},
                "retained_pages": set(),
            },
        )
        allocators.append(allocator)
    monkeypatch.setattr(hooks, "torch", torch)
    return allocators


def _reserve(allocator):
    return hooks._reserve_pages(allocator, [1], local_free=4, operation="test")


@pytest.mark.parametrize("retained_ranks", [1, 2])
def test_window_refill_skips_retained_native_free_prefix(monkeypatch, retained_ranks):
    rings = [_Ring(0), _Ring(7)]
    allocators = _cohort_allocators(monkeypatch, rings, "test")
    for allocator, ring in zip(allocators[:retained_ranks], rings[:retained_ranks]):
        lease = ring.acquire("test", 1, [1], True)[0]
        state = hooks._STATE[id(allocator)]
        state["leases_by_page"][1] = lease
        state["retained_pages"].add(1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_reserve, allocators))
    assert [[lease.block_id for lease in result] for result in results] == [[2], [2]]
    assert [allocator.free_pages[0].item() for allocator in allocators] == [2, 2]


def test_full_retained_window_reclaims_actual_demand(monkeypatch):
    rings = [_Ring(0), _Ring(7)]
    allocators = _cohort_allocators(monkeypatch, rings, "test")
    monkeypatch.setenv("GMS_SGLANG_TP_LEASE_WINDOW_PAGES", "4096")
    for allocator, ring in zip(allocators, rings):
        state = hooks._STATE[id(allocator)]
        leases = ring.acquire("test", 4, [1, 2, 3, 4], True)
        state["leases_by_page"].update({lease.block_id: lease for lease in leases})
        state["retained_pages"].update([1, 2, 3, 4])

    def reclaim(allocator, count):
        assert count == 1, "speculation must not evict the whole prefix cache"
        state = hooks._STATE[id(allocator)]
        hooks._release_tracked_leases(state, [state["leases_by_page"][1]])
        state["retained_pages"].remove(1)
        return 1

    monkeypatch.setattr(hooks, "_ensure_directory_capacity", reclaim)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_reserve, allocators))
    assert [[lease.block_id for lease in result] for result in results] == [[1], [1]]
    assert all(hooks._STATE[id(a)]["retained_pages"] == {2, 3, 4} for a in allocators)


def test_window_contention_falls_back_to_request_size(monkeypatch):
    rings = [_Ring(0), _Ring(7)]
    rings[1].acquire("competitor", 3, [1, 2, 3], True)
    allocators = _cohort_allocators(monkeypatch, rings, "test")
    monkeypatch.setenv("GMS_SGLANG_TP_LEASE_WINDOW_PAGES", "4")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_reserve, allocators))
    assert all(result is not None for result in results)
    assert [[lease.block_id for lease in result] for result in results] == [[4], [4]]


@pytest.mark.parametrize("rollback", ["allocation", "adoption", "queued-release"])
def test_rollback_realigns_remaining_window(monkeypatch, rollback):
    rings = [_Ring(0), _Ring(7)]
    allocator = _cohort_allocators(monkeypatch, rings, "test")[0]
    state = hooks._STATE[id(allocator)]
    leases = rings[0].acquire("test", 3, [1, 2, 3], True)
    state["leases_by_page"].update({lease.block_id: lease for lease in leases})
    state["tp_reserved_pages"] = [2, 3]
    state["tp_reservation_aligned"] = True
    if rollback == "allocation":
        hooks._rollback_reserved_pages(state, leases[:1])
    elif rollback == "queued-release":
        state["tp_reserved_pages"] = [1, 2, 3]
        hooks._release_tracked_leases(state, leases[:1])
    else:
        allocator.free_pages = torch.tensor([2, 3, 4])
        allocator.need_sort = False
        hooks.rollback_adopted_hbm_pages(allocator, leases[:1])
    next_leases = hooks._consume_tp_reservation(allocator, 1)
    assert [lease.block_id for lease in next_leases] == [2]
    assert allocator.free_pages[0].item() == 2


def test_small_pool_window_preserves_standby_headroom(monkeypatch):
    rings = [_Ring(0), _Ring(7)]
    primary = _cohort_allocators(monkeypatch, rings, "primary")
    shadow = _cohort_allocators(monkeypatch, rings, "shadow")
    monkeypatch.setenv("GMS_SGLANG_TP_LEASE_WINDOW_PAGES", "4096")
    with ThreadPoolExecutor(max_workers=2) as executor:
        primary_leases = list(executor.map(_reserve, primary))
        shadow_leases = list(executor.map(_reserve, shadow))
    assert all(leases is not None for leases in primary_leases + shadow_leases)
    assert primary_leases[0][0].block_id != shadow_leases[0][0].block_id


def test_explicit_demand_can_use_entire_small_pool(monkeypatch):
    rings = [_Ring(0), _Ring(7)]
    allocators = _cohort_allocators(monkeypatch, rings, "primary")
    monkeypatch.setenv("GMS_SGLANG_TP_LEASE_WINDOW_PAGES", "4096")

    def reserve_all(allocator):
        return hooks._reserve_pages(
            allocator, [1, 2, 3, 4], local_free=4, operation="test"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve_all, allocators))
    assert [[lease.block_id for lease in result] for result in results] == [
        [1, 2, 3, 4]
    ] * 2


def test_opposite_primary_shadow_rank_order_keeps_one_layout_per_cohort(monkeypatch):
    # Primary wins rank zero first, shadow wins rank one first. Independent
    # fallback selection would give both cohorts contradictory page layouts.
    rings = [_Ring(0), _Ring(17)]
    primary_chosen = threading.Event()
    shadow_peer_reserved = threading.Event()

    def before(owner, rank, _pages):
        if owner == "shadow" and rank == 0:
            assert primary_chosen.wait(5)
        if owner == "primary" and rank == 1:
            assert shadow_peer_reserved.wait(5)

    def after(owner, rank, _leases):
        if owner == "primary" and rank == 0:
            primary_chosen.set()
        if owner == "shadow" and rank == 1:
            shadow_peer_reserved.set()

    primary = _cohort_allocators(monkeypatch, rings, "primary", before, after)
    shadow = _cohort_allocators(monkeypatch, rings, "shadow", before, after)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_reserve, primary + shadow))
    assert [[lease.block_id for lease in result] for result in results] == [
        [1],
        [1],
        [2],
        [2],
    ]
    assert results[0][0].generation != results[1][0].generation
    assert results[2][0].generation != results[3][0].generation


def test_peer_nack_rolls_back_then_tries_an_alternative_native_free_page(monkeypatch):
    rings = [_Ring(0), _Ring(7)]
    rings[1].acquire("competitor", 1, [1], True)
    allocators = _cohort_allocators(monkeypatch, rings, "test")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_reserve, allocators))
    assert [[lease.block_id for lease in result] for result in results] == [[2], [2]]
    assert [lease.block_id for lease in rings[0].released] == [1]
    assert rings[1].released == []
    assert [allocator.free_pages.tolist() for allocator in allocators] == [
        [2, 1, 3, 4]
    ] * 2


def test_bounded_contention_returns_backpressure_without_native_mutation(monkeypatch):
    rings = [_Ring(0), _Ring(7)]
    rings[1].acquire("competitor", 4, [1, 2, 3, 4], True)
    allocators = _cohort_allocators(monkeypatch, rings, "test")
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(_reserve, allocators)) == [None, None]
    assert rings[0].held == {}
    assert len(rings[0].released) == hooks._TP_RESERVATION_ATTEMPTS
    assert [allocator.free_pages.tolist() for allocator in allocators] == [
        [1, 2, 3, 4]
    ] * 2
    assert all(
        hooks._STATE[id(allocator)]["leases_by_page"] == {} for allocator in allocators
    )


@pytest.mark.parametrize("operation", ["token", "paged", "extend", "decode"])
def test_zero_page_operations_never_call_gms_tp_consistency(monkeypatch, operation):
    allocator = SimpleNamespace(
        page_size=4, need_sort=False, free_pages=torch.tensor([1, 2])
    )
    monkeypatch.setitem(hooks._STATE, id(allocator), {})
    monkeypatch.setattr(
        hooks, "_agree_native_capacity", lambda *_: pytest.fail("zero-page collective")
    )
    monkeypatch.setattr(
        hooks, "_reserve_pages", lambda *_a, **_k: pytest.fail("zero-page reservation")
    )
    monkeypatch.setattr(hooks, "get_num_new_pages", lambda **_: 0)
    sentinel = object()
    if operation == "token":
        monkeypatch.setattr(hooks, "orig_token_alloc", lambda *_: sentinel)
        result = hooks._gms_token_alloc(allocator, 0)
    elif operation == "paged":
        monkeypatch.setattr(hooks, "orig_paged_alloc", lambda *_: sentinel)
        result = hooks._gms_paged_alloc(allocator, 0)
    elif operation == "extend":
        monkeypatch.setattr(
            hooks, "orig_paged_alloc_extend", lambda *_a, **_k: sentinel
        )
        result = hooks._gms_paged_alloc_extend(allocator, [1], [1], [2], [2], [4], 1, 0)
    else:
        monkeypatch.setattr(hooks, "orig_paged_alloc_decode", lambda *_: sentinel)
        result = hooks._gms_paged_alloc_decode(allocator, [2], [2], [4])
    assert result is sentinel
