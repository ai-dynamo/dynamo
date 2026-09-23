# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import deque
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


def test_window_preflights_shared_capacity_before_reservation(monkeypatch):
    allocator = SimpleNamespace(
        free_pages=torch.tensor([1, 2, 3, 4]),
        _gms_tp_consistency=TPConsistency(),
    )
    state = {
        "client": SimpleNamespace(),
        "leases_by_page": {},
        "retained_pages": set(),
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "torch", torch)
    events = []

    def ensure_capacity(_allocator, count):
        events.append(("capacity", count))
        return 0

    def reserve(_allocator, pages, _operation, *, count, reclaim=True):
        events.append(("reserve", count, reclaim))
        leases = [KVLease(page, 1) for page in pages[:count]]
        state["leases_by_page"].update({lease.block_id: lease for lease in leases})
        return leases

    monkeypatch.setattr(hooks, "_ensure_directory_capacity", ensure_capacity)
    monkeypatch.setattr(hooks, "_reserve_tp_pages", reserve)
    leases = hooks._refill_tp_reservation(allocator, 1, "test")
    assert [lease.block_id for lease in leases] == [1]
    assert events[0] == ("capacity", 1)
    assert events[1][0] == "reserve"


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


def test_exclusive_adoption_rollback_restores_cpu_free_page_order(monkeypatch):
    allocator = SimpleNamespace(
        free_pages=torch.tensor([3, 4]),
        need_sort=False,
    )
    lease = KVLease(2, 7)
    state = {
        "leases_by_page": {2: lease},
        "retained_pages": set(),
        "exclusive_steady_state": True,
        "cpu_free_pages": deque([3, 4]),
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "torch", torch)

    hooks.rollback_adopted_hbm_pages(allocator, [lease])

    assert allocator.free_pages.tolist() == [2, 3, 4]
    assert list(state["cpu_free_pages"]) == [2, 3, 4]


def test_steady_state_free_refills_active_window_without_ring_transition(monkeypatch):
    released = []
    parked = []
    allocator = SimpleNamespace()
    lease = KVLease(2, 7)
    state = {
        "client": SimpleNamespace(
            release=lambda leases: released.extend(leases),
            park_idle=lambda leases: parked.extend(leases),
        ),
        "leases_by_page": {2: lease},
        "retained_pages": set(),
        "tp_reserved_pages": [3],
        "tp_reservation_aligned": True,
        "steady_state": True,
        "exclusive_steady_state": True,
        "cpu_free_pages": deque([2]),
        "active_free_pages": set(),
        "active_window_pages": 1,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    hooks._release_pages(allocator, torch.tensor([2]))
    hooks._publish_steady_native_free(allocator, [2], "test", placement="prepend")

    assert released == []
    assert parked == []
    assert state["active_free_pages"] == {2}
    assert state["leases_by_page"] == {2: lease}
    assert state["tp_reserved_pages"] == [3, 2]


def test_steady_state_free_parks_only_active_window_surplus(monkeypatch):
    parked = []
    allocator = SimpleNamespace()
    leases = {2: KVLease(2, 7), 3: KVLease(3, 8)}
    state = {
        "client": SimpleNamespace(
            release=lambda _leases: None,
            park_idle=lambda batch: parked.extend(batch),
        ),
        "leases_by_page": leases,
        "retained_pages": set(),
        "tp_reserved_pages": [],
        "steady_state": True,
        "exclusive_steady_state": True,
        "cpu_free_pages": deque([2, 3]),
        "active_free_pages": {2},
        "active_window_pages": 1,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    hooks._release_pages(allocator, torch.tensor([3]))
    hooks._publish_steady_native_free(allocator, [3], "test", placement="prepend")

    assert parked == [leases[3]]
    assert state["active_free_pages"] == {2}


def test_recovery_mode_free_still_releases_lease(monkeypatch):
    released = []
    allocator = SimpleNamespace()
    lease = KVLease(2, 7)
    state = {
        "client": SimpleNamespace(release=lambda leases: released.extend(leases)),
        "leases_by_page": {2: lease},
        "retained_pages": set(),
        "tp_reserved_pages": [],
        "tp_reservation_aligned": False,
        "steady_state": False,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    hooks._release_pages(allocator, torch.tensor([2]))

    assert released == [lease]
    assert state["leases_by_page"] == {}


def test_exclusive_steady_state_exposes_only_owned_writable_pages(monkeypatch):
    acquired = []

    def acquire(count, *, preferred_blocks, allow_partial, strict_preferred):
        assert count == 2
        assert preferred_blocks == [3, 4]
        assert allow_partial is True
        assert strict_preferred is True
        leases = [KVLease(3, 5)]
        acquired.extend(leases)
        return leases

    allocator = SimpleNamespace(
        free_pages=torch.tensor([1, 2, 3, 4]),
        need_sort=False,
        _gms_tp_consistency=TPConsistency(),
    )
    state = {
        "client": SimpleNamespace(
            acquire=acquire, release=lambda leases: None, park_idle=lambda leases: None
        ),
        "leases_by_page": {1: KVLease(1, 3), 2: KVLease(2, 4)},
        "retained_pages": {2},
        "tp_reserved_pages": [1],
        "tp_reservation_aligned": True,
        "steady_state": False,
        "exclusive_steady_state": False,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "torch", torch)

    assert hooks.enter_exclusive_steady_state(allocator) == 2
    assert allocator.free_pages.tolist() == [1, 3]
    assert state["leases_by_page"][3] == acquired[0]
    assert state["exclusive_hidden_pages"] == {2, 4}
    assert state["tp_reserved_pages"] == []
    assert state["steady_state"] is True
    assert state["exclusive_steady_state"] is True
    assert list(state["cpu_free_pages"]) == [1, 3]


def test_exclusive_steady_state_reconciles_different_standby_deltas(monkeypatch):
    """Promotion agrees final ownership, not rank-local warmup history."""
    votes = _Votes()
    allocators = []
    parked = [[], []]
    for rank, initially_owned in enumerate((1, 2)):
        cohort = TPConsistency(world_size=2)
        cohort._rank = lambda rank=rank: rank
        cohort._gather = lambda value, rank=rank: votes.gather(rank, value)
        owned = {initially_owned: KVLease(initially_owned, 10 + rank)}

        def acquire(
            count,
            *,
            preferred_blocks,
            allow_partial,
            strict_preferred,
            owned=owned,
            rank=rank,
        ):
            assert allow_partial is True
            assert strict_preferred is True
            leases = [KVLease(page, 20 + rank) for page in preferred_blocks[:count]]
            owned.update({lease.block_id: lease for lease in leases})
            return leases

        allocator = SimpleNamespace(
            free_pages=torch.tensor([1, 2, 3, 4]),
            need_sort=False,
            _gms_tp_consistency=cohort,
        )
        state = {
            "client": SimpleNamespace(
                acquire=acquire,
                release=lambda _leases: None,
                park_idle=lambda leases, rank=rank: parked[rank].extend(leases),
            ),
            "leases_by_page": owned,
            "retained_pages": set(),
            "tp_reserved_pages": [],
            "tp_reservation_aligned": True,
            "steady_state": False,
            "exclusive_steady_state": False,
        }
        monkeypatch.setitem(hooks._STATE, id(allocator), state)
        allocators.append(allocator)

    monkeypatch.setattr(hooks, "torch", torch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(hooks.enter_exclusive_steady_state, allocators))

    assert results == [4, 4]
    assert [allocator.free_pages.tolist() for allocator in allocators] == [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    assert [set(hooks._STATE[id(a)]["leases_by_page"]) for a in allocators] == [
        {1, 2, 3, 4},
        {1, 2, 3, 4},
    ]
    assert [[lease.block_id for lease in batch] for batch in parked] == [
        [2, 3, 4],
        [2, 3, 4],
    ]


def test_exclusive_steady_state_withholds_asymmetric_rank_pages(monkeypatch):
    votes = _Votes()
    allocators = []
    released = [[], []]
    for rank in range(2):
        cohort = TPConsistency(world_size=2)
        cohort._rank = lambda rank=rank: rank
        cohort._gather = lambda value, rank=rank: votes.gather(rank, value)
        owned = {}

        def acquire(
            count,
            *,
            preferred_blocks,
            allow_partial,
            strict_preferred,
            owned=owned,
            rank=rank,
        ):
            assert allow_partial is True
            assert strict_preferred is True
            available = preferred_blocks[:-1] if rank == 0 else preferred_blocks
            leases = [KVLease(page, 20 + rank) for page in available[:count]]
            owned.update({lease.block_id: lease for lease in leases})
            return leases

        allocator = SimpleNamespace(
            free_pages=torch.tensor([1, 2, 3, 4]),
            need_sort=False,
            _gms_tp_consistency=cohort,
        )
        state = {
            "client": SimpleNamespace(
                acquire=acquire,
                release=lambda leases, rank=rank: released[rank].extend(leases),
                park_idle=lambda _leases: None,
            ),
            "leases_by_page": owned,
            "retained_pages": set(),
            "tp_reserved_pages": [],
            "tp_reservation_aligned": True,
            "steady_state": False,
            "exclusive_steady_state": False,
        }
        monkeypatch.setitem(hooks._STATE, id(allocator), state)
        allocators.append(allocator)

    monkeypatch.setattr(hooks, "torch", torch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(hooks.enter_exclusive_steady_state, allocators))

    assert results == [3, 3]
    assert [allocator.free_pages.tolist() for allocator in allocators] == [
        [1, 2, 3],
        [1, 2, 3],
    ]
    assert [hooks._STATE[id(a)]["exclusive_hidden_pages"] for a in allocators] == [
        {4},
        {4},
    ]
    assert released[0] == []
    assert [lease.block_id for lease in released[1]] == [4]


def test_exclusive_steady_state_parks_tail_and_refills_bounded_window(monkeypatch):
    monkeypatch.setenv("GMS_SGLANG_ACTIVE_LEASE_WINDOW_PAGES", "1")
    leases = {
        1: KVLease(1, 3),
        2: KVLease(2, 4),
        3: KVLease(3, 5),
    }
    parked = []
    activated = []
    client = SimpleNamespace(
        acquire=lambda *_args, **_kwargs: [],
        release=lambda _leases: None,
        park_idle=lambda batch: parked.extend(batch),
        activate_idle=lambda batch: activated.extend(batch),
    )
    allocator = SimpleNamespace(
        free_pages=torch.tensor([1, 2, 3]),
        need_sort=False,
        _gms_tp_consistency=TPConsistency(),
    )
    state = {
        "client": client,
        "leases_by_page": dict(leases),
        "retained_pages": set(),
        "tp_reserved_pages": [],
        "tp_reservation_aligned": True,
        "steady_state": False,
        "exclusive_steady_state": False,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "torch", torch)

    assert hooks.enter_exclusive_steady_state(allocator) == 3
    assert state["active_free_pages"] == {1}
    assert parked == [leases[2], leases[3]]

    hooks._ensure_steady_active_window(allocator, 2, "test")
    assert activated == [leases[2]]
    assert state["active_free_pages"] == {1, 2}
    assert hooks._consume_steady_pages(state, 2) == [1, 2]
    assert state["active_free_pages"] == set()


def test_active_prefix_rebalances_fragmented_free_pages(monkeypatch):
    leases = {page: KVLease(page, page + 10) for page in (1, 2, 3, 4)}
    parked = []
    activated = []
    allocator = SimpleNamespace(_gms_tp_consistency=TPConsistency())
    state = {
        "client": SimpleNamespace(
            park_idle=lambda batch: parked.extend(batch),
            activate_idle=lambda batch: activated.extend(batch),
        ),
        "leases_by_page": leases,
        "cpu_free_pages": deque([1, 2, 3, 4]),
        "active_free_pages": {2, 3, 4},
        "active_window_pages": 2,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    hooks._set_steady_active_prefix(allocator, 0, "test")

    assert parked == [leases[3], leases[4]]
    assert activated == [leases[1]]
    assert state["active_free_pages"] == {1, 2}


def test_token_free_updates_cpu_mirror_and_parks_tail(monkeypatch):
    leases = {1: KVLease(1, 11), 2: KVLease(2, 12)}
    parked = []
    allocator = SimpleNamespace(
        page_size=1,
        need_sort=False,
        free_group=None,
        _gms_tp_consistency=TPConsistency(),
    )
    state = {
        "client": SimpleNamespace(
            park_idle=lambda batch: parked.extend(batch),
            activate_idle=lambda _batch: None,
        ),
        "leases_by_page": leases,
        "retained_pages": set(),
        "tp_reserved_pages": [],
        "tp_reservation_aligned": True,
        "steady_state": True,
        "exclusive_steady_state": True,
        "cpu_free_pages": deque([1]),
        "cpu_staged_pages": [],
        "active_free_pages": {1},
        "active_window_pages": 1,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "orig_token_free", lambda _self, _pages: "freed")
    monkeypatch.setattr(hooks, "torch", torch)

    assert hooks._gms_token_free(allocator, torch.tensor([2])) == "freed"
    assert list(state["cpu_free_pages"]) == [1, 2]
    assert state["active_free_pages"] == {1}
    assert parked == [leases[2]]


def test_token_sorted_free_mirror_merges_with_native_release_pages(monkeypatch):
    allocator = SimpleNamespace(release_pages=torch.tensor([3]))
    state = {
        "exclusive_steady_state": True,
        "cpu_free_pages": deque([1, 4]),
        "cpu_staged_pages": [3, 2],
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    def merge(_self):
        allocator.release_pages = torch.tensor([])
        return "merged"

    monkeypatch.setattr(hooks, "orig_token_merge_and_sort_free", merge)

    assert hooks._gms_token_merge_and_sort_free(allocator) == "merged"
    assert list(state["cpu_free_pages"]) == [1, 2, 3, 4]
    assert state["cpu_staged_pages"] == []


def test_exclusive_steady_state_preserves_one_page_for_standby(monkeypatch):
    acquired = []

    def acquire(count, *, preferred_blocks, allow_partial, strict_preferred):
        assert count == 1
        assert preferred_blocks == [3]
        assert allow_partial is True
        assert strict_preferred is True
        leases = [KVLease(3, 5)]
        acquired.extend(leases)
        return leases

    allocator = SimpleNamespace(
        free_pages=torch.tensor([1, 2, 3, 4]),
        need_sort=False,
        page_size=64,
        _gms_tp_consistency=TPConsistency(),
        _gms_standby_headroom_pages=1,
    )
    state = {
        "client": SimpleNamespace(
            acquire=acquire, release=lambda leases: None, park_idle=lambda leases: None
        ),
        "leases_by_page": {1: KVLease(1, 3), 2: KVLease(2, 4)},
        "retained_pages": {2},
        "tp_reserved_pages": [],
        "tp_reservation_aligned": True,
        "steady_state": False,
        "exclusive_steady_state": False,
        "exclusive_hidden_pages": set(),
        "standby_headroom_pages": set(),
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "torch", torch)

    assert hooks.enter_exclusive_steady_state(allocator) == 2
    assert allocator.free_pages.tolist() == [1, 3]
    assert state["standby_headroom_pages"] == {4}
    assert state["exclusive_hidden_pages"] == {2}
    assert hooks.hidden_recoverable_tokens(allocator) == 128


def test_exclusive_extend_records_pages_without_device_read(monkeypatch):
    req = SimpleNamespace(_gms_kv_page_ids=[2])
    allocator = SimpleNamespace(
        page_size=64, need_sort=False, free_pages=torch.tensor([3, 4])
    )
    state = {
        "exclusive_steady_state": True,
        "cpu_free_pages": deque([3, 4]),
        "active_free_pages": {3, 4},
        "active_batch": SimpleNamespace(reqs=[req]),
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    expected = object()

    def allocate(*_args, **_kwargs):
        allocator.free_pages = allocator.free_pages[1:]
        return expected

    monkeypatch.setattr(hooks, "orig_paged_alloc_extend", allocate)

    result = hooks._gms_paged_alloc_extend(
        allocator,
        torch.tensor([64]),
        torch.tensor([64]),
        torch.tensor([65]),
        torch.tensor([65]),
        torch.tensor([127]),
        1,
    )

    assert result is expected
    assert req._gms_kv_page_ids == [2, 3]
    assert list(state["cpu_free_pages"]) == [4]


def test_exclusive_decode_records_pages_without_device_read(monkeypatch):
    req = SimpleNamespace(_gms_kv_page_ids=[2])
    allocator = SimpleNamespace(
        page_size=64, need_sort=False, free_pages=torch.tensor([3, 4])
    )
    state = {
        "exclusive_steady_state": True,
        "cpu_free_pages": deque([3, 4]),
        "active_free_pages": {3, 4},
        "active_batch": SimpleNamespace(reqs=[req]),
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    expected = object()

    def allocate(*_args, **_kwargs):
        allocator.free_pages = allocator.free_pages[1:]
        return expected

    monkeypatch.setattr(hooks, "orig_paged_alloc_decode", allocate)

    result = hooks._gms_paged_alloc_decode(
        allocator, torch.tensor([65]), torch.tensor([65]), torch.tensor([127])
    )

    assert result is expected
    assert req._gms_kv_page_ids == [2, 3]
    assert list(state["cpu_free_pages"]) == [4]


def test_startup_extend_records_reserved_pages(monkeypatch):
    req = SimpleNamespace(_gms_kv_page_ids=[2])
    leases = [KVLease(3, 7)]
    allocator = SimpleNamespace(
        page_size=64,
        need_sort=False,
        free_pages=torch.tensor([3, 4]),
        _gms_tp_consistency=TPConsistency(),
    )
    state = {
        "client": SimpleNamespace(acquire=lambda *args, **kwargs: leases),
        "leases_by_page": {},
        "retained_pages": set(),
        "active_batch": SimpleNamespace(reqs=[req]),
        "tp_reserved_pages": [],
        "tp_reservation_aligned": True,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(
        hooks, "orig_paged_alloc_extend", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(hooks, "get_num_new_pages", lambda **kwargs: 1)

    result = hooks._gms_paged_alloc_extend(
        allocator,
        torch.tensor([64]),
        torch.tensor([64]),
        torch.tensor([65]),
        torch.tensor([65]),
        torch.tensor([127]),
        1,
    )

    assert result is not None
    assert req._gms_kv_page_ids == [2, 3]


def test_startup_decode_records_reserved_pages(monkeypatch):
    req = SimpleNamespace(_gms_kv_page_ids=[2])
    leases = [KVLease(3, 7)]
    allocator = SimpleNamespace(
        page_size=64,
        need_sort=False,
        free_pages=torch.tensor([3, 4]),
        _gms_tp_consistency=TPConsistency(),
    )
    state = {
        "client": SimpleNamespace(acquire=lambda *args, **kwargs: leases),
        "leases_by_page": {},
        "retained_pages": set(),
        "active_batch": SimpleNamespace(reqs=[req]),
        "tp_reserved_pages": [],
        "tp_reservation_aligned": True,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "orig_paged_alloc_decode", lambda *args: object())
    monkeypatch.setattr(hooks, "get_num_new_pages", lambda **kwargs: 1)

    result = hooks._gms_paged_alloc_decode(
        allocator, torch.tensor([65]), torch.tensor([65]), torch.tensor([127])
    )

    assert result is not None
    assert req._gms_kv_page_ids == [2, 3]


def test_exclusive_adoption_can_consume_hidden_recoverable_page(monkeypatch):
    allocator = SimpleNamespace(
        page_size=2,
        num_pages=4,
        need_sort=False,
        free_pages=torch.tensor([1, 3]),
    )
    old = KVLease(2, 7)
    successor = KVLease(2, 8)
    state = {
        "client": SimpleNamespace(adopt=lambda leases: [successor]),
        "leases_by_page": {},
        "retained_pages": set(),
        "exclusive_steady_state": True,
        "exclusive_hidden_pages": {2, 4},
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    indices, leases = hooks.adopt_hbm_pages(allocator, [2], [old.generation])

    assert indices.tolist() == [4, 5]
    assert leases == [successor]
    assert allocator.free_pages.tolist() == [1, 3]
    assert state["exclusive_hidden_pages"] == {4}
    assert state["leases_by_page"] == {2: successor}
    assert hooks.hidden_recoverable_tokens(allocator) == 2


def test_pressure_activates_hidden_recovery_capacity_in_bounded_batch(monkeypatch):
    victims = [
        {
            "content_hash": bytes([page]) * 32,
            "engine_id": "engine-0",
            "slot_ids": [page],
            "generations": [page + 10],
        }
        for page in (2, 4)
    ]

    class Directory:
        authoritative = True

        def ensure_hbm_capacity(self, count, *, eligible_slot_ids=None):
            assert count == 2
            assert eligible_slot_ids == [2, 4, 5, 6]
            return victims

    def adopt(leases):
        return [KVLease(lease.block_id, lease.generation + 1) for lease in leases]

    allocator = SimpleNamespace(
        page_size=2,
        size=16,
        free_pages=torch.tensor([1, 3]),
        get_all_free_pages=lambda: torch.tensor([1, 3]),
        _gms_kv_directory=Directory(),
        _gms_tp_consistency=TPConsistency(),
        _gms_recovery_candidates={item["content_hash"] for item in victims},
    )
    state = {
        "client": SimpleNamespace(adopt=adopt),
        "leases_by_page": {},
        "retained_pages": set(),
        "exclusive_steady_state": True,
        "cpu_free_pages": deque([1, 3]),
        "exclusive_hidden_pages": {2, 4, 5, 6},
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "torch", torch)

    activated = hooks.activate_hidden_recovery_capacity(allocator, 2)

    assert activated == 4
    assert allocator.free_pages.tolist() == [1, 3, 2, 4]
    assert list(state["cpu_free_pages"]) == [1, 3, 2, 4]
    assert state["exclusive_hidden_pages"] == {5, 6}
    assert state["leases_by_page"] == {2: KVLease(2, 13), 4: KVLease(4, 15)}
    assert allocator._gms_recovery_candidates == set()


def test_pressure_stops_retrying_single_unretirable_recovery_page(monkeypatch):
    calls = []

    class Directory:
        authoritative = True

        def ensure_hbm_capacity(self, count, *, eligible_slot_ids=None):
            calls.append((count, eligible_slot_ids))
            return []

    allocator = SimpleNamespace(
        page_size=2,
        size=16,
        free_pages=torch.tensor([1, 3]),
        _gms_kv_directory=Directory(),
        _gms_tp_consistency=TPConsistency(),
        _gms_recovery_candidates={b"h" * 32},
    )
    state = {
        "client": SimpleNamespace(),
        "leases_by_page": {},
        "retained_pages": set(),
        "exclusive_steady_state": True,
        "exclusive_hidden_pages": {4},
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    assert hooks.activate_hidden_recovery_capacity(allocator, 2) == 0
    assert hooks.activate_hidden_recovery_capacity(allocator, 2) == 0

    assert calls == [(1, [4])]
    assert state["recovery_capacity_exhausted"] is True
    assert state["exclusive_hidden_pages"] == {4}
    assert allocator._gms_recovery_candidates == set()


def test_exclusive_release_demotes_before_native_free(monkeypatch):
    allocator = SimpleNamespace(need_sort=False)
    state = {"exclusive_steady_state": True}
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    events = []
    monkeypatch.setattr(
        hooks,
        "_demote_exact_retained_pages",
        lambda self, pages: events.append(("demote", pages)),
    )
    monkeypatch.setattr(
        hooks, "_publish_steady_native_free", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        hooks,
        "orig_paged_release_page_ids",
        lambda self, *pages: events.append(("native", pages)) or "released",
    )
    pages = torch.tensor([2, 3])

    assert hooks._gms_paged_release_page_ids(allocator, pages) == "released"
    assert events[0] == ("demote", [2, 3])
    assert events[1][0] == "native"


def test_exclusive_release_uses_matching_cpu_page_hint(monkeypatch):
    class DevicePages:
        def numel(self):
            return 2

        def detach(self):
            raise AssertionError("matching CPU hint must avoid a device read")

    allocator = SimpleNamespace(need_sort=False)
    state = {
        "exclusive_steady_state": True,
        # Native free_group_end may combine several request segments into one
        # _release_page_ids call.
        "cpu_release_pages": [[2], [3]],
        "cpu_free_pages": [],
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    events = []
    monkeypatch.setattr(
        hooks,
        "_demote_exact_retained_pages",
        lambda self, pages: events.append(("demote", pages)),
    )
    monkeypatch.setattr(
        hooks, "_publish_steady_native_free", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        hooks,
        "_forget_local_page_mappings",
        lambda self, pages: events.append(("forget", pages)),
    )
    monkeypatch.setattr(
        hooks,
        "orig_paged_release_page_ids",
        lambda self, *pages: "released",
    )

    assert hooks._gms_paged_release_page_ids(allocator, DevicePages()) == "released"
    assert events == [("demote", [2, 3]), ("forget", [2, 3])]
    assert state["cpu_release_pages"] == []
    assert list(state["cpu_free_pages"]) == [2, 3]


def test_exclusive_release_rejects_wrong_sized_cpu_page_hint(monkeypatch):
    allocator = SimpleNamespace(need_sort=False)
    state = {
        "exclusive_steady_state": True,
        "cpu_release_pages": [[9]],
        "cpu_free_pages": [],
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)
    monkeypatch.setattr(hooks, "_demote_exact_retained_pages", lambda *_args: None)
    monkeypatch.setattr(
        hooks, "_publish_steady_native_free", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(hooks, "_forget_local_page_mappings", lambda *_args: None)
    monkeypatch.setattr(
        hooks,
        "orig_paged_release_page_ids",
        lambda self, *pages: "released",
    )

    pages = torch.tensor([2, 3])
    assert hooks._gms_paged_release_page_ids(allocator, pages) == "released"
    assert state["cpu_release_pages"] == []
    assert list(state["cpu_free_pages"]) == [2, 3]


def test_exact_sealed_demotion_accepts_canonical_hash_order(monkeypatch):
    victims = [
        {
            "content_hash": b"a" * 32,
            "engine_id": "engine-0",
            "slot_ids": [3],
            "generations": [7],
        },
        {
            "content_hash": b"b" * 32,
            "engine_id": "engine-0",
            "slot_ids": [2],
            "generations": [7],
        },
    ]

    class Directory:
        authoritative = True

        def ensure_hbm_capacity(self, count, *, eligible_slot_ids=None):
            assert count == 2
            assert eligible_slot_ids == [2, 3]
            return victims

    def adopt(leases):
        return [KVLease(lease.block_id, lease.generation + 1) for lease in leases]

    allocator = SimpleNamespace(
        _gms_kv_directory=Directory(),
        _gms_tp_consistency=TPConsistency(),
        _gms_recovery_candidates={item["content_hash"] for item in victims},
    )
    state = {
        "client": SimpleNamespace(adopt=adopt),
        "leases_by_page": {2: KVLease(2, 7), 3: KVLease(3, 7)},
        "retained_pages": {2, 3},
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    hooks._demote_exact_retained_pages(allocator, [2, 3])

    assert state["retained_pages"] == set()
    assert state["leases_by_page"] == {2: KVLease(2, 8), 3: KVLease(3, 8)}
    assert allocator._gms_recovery_candidates == set()


def test_steady_state_demotes_cold_recovery_batch_without_releasing_pages(
    monkeypatch,
):
    retained = set(range(1, 14))
    leases = {page: KVLease(page, 7) for page in retained}
    released = []
    adopted = []
    victims = [
        {
            "content_hash": bytes([page]) * 32,
            "engine_id": "engine-0",
            "slot_ids": [page],
            "generations": [7],
        }
        for page in range(1, 6)
    ]

    class Directory:
        authoritative = True

        def ensure_hbm_capacity(self, count, *, eligible_slot_ids=None):
            assert count == 5
            assert eligible_slot_ids == list(range(1, 14))
            return victims

    def adopt(old):
        adopted.extend(old)
        return [KVLease(lease.block_id, lease.generation + 1) for lease in old]

    allocator = SimpleNamespace(
        size=16,
        page_size=1,
        _gms_kv_directory=Directory(),
        _gms_tp_consistency=TPConsistency(),
        _gms_recovery_candidates={victim["content_hash"] for victim in victims},
    )
    state = {
        "client": SimpleNamespace(
            adopt=adopt, release=lambda value: released.extend(value)
        ),
        "leases_by_page": leases,
        "retained_pages": retained,
        "tp_reserved_pages": [],
        "tp_reservation_aligned": True,
        "steady_state": True,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    assert hooks.maintain_steady_state_headroom(allocator) == 5

    assert [lease.block_id for lease in adopted] == [1, 2, 3, 4, 5]
    assert released == []
    assert state["retained_pages"] == set(range(6, 14))
    assert all(state["leases_by_page"][page].generation == 8 for page in range(1, 6))
    assert allocator._gms_recovery_candidates == set()


def test_steady_state_headroom_counts_hidden_predecessor_pages(monkeypatch):
    retained = {9, 10, 11}
    leases = {page: KVLease(page, 7) for page in retained}
    victims = [
        {
            "content_hash": bytes([page]) * 32,
            "engine_id": "engine-0",
            "slot_ids": [page],
            "generations": [7],
        }
        for page in sorted(retained)
    ]

    class Directory:
        authoritative = True

        def ensure_hbm_capacity(self, count, *, eligible_slot_ids=None):
            # 10 hidden + 3 current pages exceeds the 12-page high watermark;
            # demote the current pages down to the 8-page recovery target.
            assert count == 3
            assert eligible_slot_ids == [9, 10, 11]
            return victims

    allocator = SimpleNamespace(
        size=16,
        page_size=1,
        _gms_kv_directory=Directory(),
        _gms_tp_consistency=TPConsistency(),
        _gms_recovery_candidates={victim["content_hash"] for victim in victims},
    )
    state = {
        "client": SimpleNamespace(
            adopt=lambda old: [
                KVLease(lease.block_id, lease.generation + 1) for lease in old
            ]
        ),
        "leases_by_page": leases,
        "retained_pages": retained,
        "exclusive_hidden_pages": set(range(20, 30)),
        "tp_reserved_pages": [],
        "tp_reservation_aligned": True,
        "steady_state": True,
    }
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    assert hooks.maintain_steady_state_headroom(allocator) == 3
    assert state["retained_pages"] == set()
    assert state["exclusive_hidden_pages"] == set(range(20, 30))
    assert all(state["leases_by_page"][page].generation == 8 for page in retained)


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


@pytest.mark.parametrize("fail_preparation", [False, True])
def test_preparation_is_voted_before_native_state_changes(
    monkeypatch, fail_preparation
):
    rings = [_Ring(0), _Ring(7)]
    allocators = _cohort_allocators(monkeypatch, rings, "test")
    # Leader chooses page 1; rank 1 must stage a reordered native free tensor.
    allocators[1].free_pages = torch.tensor([2, 3, 4, 1])
    originals = [allocator.free_pages for allocator in allocators]
    monkeypatch.setattr(hooks, "_TP_RESERVATION_ATTEMPTS", 1)
    events = [[], []]
    for rank, allocator in enumerate(allocators):
        gather = allocator._gms_tp_consistency._gather

        def checked_gather(value, rank=rank, gather=gather, allocator=allocator):
            events[rank].append(value[0])
            # Neither the selection nor prepare vote may observe installed state.
            assert allocator.free_pages is originals[rank]
            return gather(value)

        allocator._gms_tp_consistency._gather = checked_gather

    def prepare(*args, **kwargs):
        if fail_preparation:
            raise RuntimeError("injected tensor preparation failure")
        return torch.tensor(*args, **kwargs)

    monkeypatch.setattr(hooks, "torch", SimpleNamespace(tensor=prepare))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda a: hooks._reserve_tp_pages(a, [1], "test", reclaim=False),
                allocators,
            )
        )
    if fail_preparation:
        assert results == [None, None]
        assert all(not ring.held for ring in rings)
        assert all(
            a.free_pages is original for a, original in zip(allocators, originals)
        )
        assert all(stages[-1].endswith(":rollback") for stages in events)
    else:
        assert [[lease.block_id for lease in leases] for leases in results] == [
            [1],
            [1],
        ]
        assert [a.free_pages[0].item() for a in allocators] == [1, 1]
        assert all(len(stages) == 2 for stages in events)
