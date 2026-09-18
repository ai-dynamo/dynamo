# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.sglang import install_kv_leases as adapter

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
]


class _Client:
    def __init__(self, *, fail_once=False, on_release=None):
        self.fail_once = fail_once
        self.on_release = on_release
        self.calls = []

    def release(self, leases):
        batch = list(leases)
        self.calls.append(batch)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("release failed")
        if self.on_release is not None:
            self.on_release(batch)


def _allocator(client, lease_map, retained=()):
    allocator = SimpleNamespace()
    adapter._STATE[id(allocator)] = {
        "client": client,
        "leases_by_page": lease_map,
        "retained_pages": set(retained),
    }
    return allocator


def _cleanup(allocator):
    adapter._STATE.pop(id(allocator), None)


def test_release_failure_preserves_leases_for_retry():
    leases = {1: KVLease(1, 7), 2: KVLease(2, 9)}
    client = _Client(fail_once=True)
    allocator = _allocator(client, leases)
    pages = torch.tensor([1, 2])
    try:
        with pytest.raises(RuntimeError, match="release failed"):
            adapter._release_pages(allocator, pages)
        assert leases == {1: KVLease(1, 7), 2: KVLease(2, 9)}

        adapter._release_pages(allocator, pages)
        assert client.calls == [
            [KVLease(1, 7), KVLease(2, 9)],
            [KVLease(1, 7), KVLease(2, 9)],
        ]
        assert leases == {}
    finally:
        _cleanup(allocator)


def test_release_removes_pages_from_tp_reservation_window():
    leases = {1: KVLease(1, 7), 2: KVLease(2, 9)}
    client = _Client()
    allocator = _allocator(client, leases)
    adapter._STATE[id(allocator)]["tp_reserved_pages"] = [1, 2]
    try:
        adapter._release_pages(allocator, torch.tensor([1]))
        assert adapter._STATE[id(allocator)]["tp_reserved_pages"] == [2]
        assert leases == {2: KVLease(2, 9)}
    finally:
        _cleanup(allocator)


def test_paged_free_preserves_agreed_reservation_prefix(monkeypatch):
    # install() binds the running SGLang torch module before this wrapper can
    # execute in production; direct unit invocation must establish that seam.
    monkeypatch.setattr(adapter, "torch", torch)
    leases = {4: KVLease(4, 7)}
    allocator = _allocator(_Client(), leases)
    allocator.free_pages = torch.tensor([2, 3, 5])
    allocator.need_sort = False
    state = adapter._STATE[id(allocator)]
    state["tp_reserved_pages"] = [2, 3]
    state["tp_reservation_aligned"] = True

    def native_release(self, *page_ids):
        self.free_pages = torch.cat((*page_ids, self.free_pages))

    monkeypatch.setattr(adapter, "orig_paged_release_page_ids", native_release)
    try:
        adapter._gms_paged_release_page_ids(allocator, torch.tensor([4]))
        assert allocator.free_pages.tolist() == [2, 3, 4, 5]
        assert state["tp_reservation_aligned"] is True
    finally:
        _cleanup(allocator)


def test_release_does_not_remove_a_successor_generation():
    leases = {1: KVLease(1, 7)}

    def install_successor(_batch):
        leases[1] = KVLease(1, 8)

    allocator = _allocator(_Client(on_release=install_successor), leases)
    try:
        adapter._release_pages(allocator, torch.tensor([1]))
        assert leases == {1: KVLease(1, 8)}
    finally:
        _cleanup(allocator)


def test_release_deduplicates_pages_and_skips_retained_pages():
    leases = {1: KVLease(1, 7), 2: KVLease(2, 9)}
    client = _Client()
    allocator = _allocator(client, leases, retained={2})
    try:
        adapter._release_pages(allocator, torch.tensor([1, 1, 2]))
        assert client.calls == [[KVLease(1, 7)]]
        assert leases == {2: KVLease(2, 9)}
    finally:
        _cleanup(allocator)


def test_allocator_arms_parent_death_fence_before_opening_lease_client(monkeypatch):
    events = []
    allocator = SimpleNamespace(size=8, page_size=2)
    client = SimpleNamespace(namespace="test", owner_id=7)

    monkeypatch.setattr(
        adapter,
        "arm_parent_death_signal",
        lambda: events.append("fenced"),
    )
    monkeypatch.setattr(
        adapter,
        "_make_client",
        lambda _allocator, pages: events.append(("client", pages)) or client,
    )
    try:
        adapter._initialize_allocator(allocator)
        assert events == ["fenced", ("client", 4)]
        assert allocator._gms_kv_lease_client is client
        assert allocator._gms_kv_leases_by_page == {}
        assert allocator._gms_retained_pages == set()
    finally:
        _cleanup(allocator)


def test_paged_clear_revokes_all_derived_fast_path_state(monkeypatch):
    monkeypatch.setattr(adapter, "torch", torch)
    leases = {1: KVLease(1, 7)}
    allocator = _allocator(_Client(), leases, retained=(1,))
    allocator.free_pages = torch.tensor([1, 2, 3])
    allocator.num_pages = 4
    state = adapter._STATE[id(allocator)]
    state.update(
        exclusive_steady_state=True,
        steady_state=True,
        exclusive_hidden_pages={4},
        cpu_free_pages=[2, 3],
        cpu_staged_pages=[1],
        cpu_release_pages=[[2]],
        tp_reserved_pages=[2],
        tp_reservation_aligned=True,
        active_batch=object(),
    )
    observed = {}

    def native_clear(self):
        observed.update(state)
        self.free_pages = torch.tensor([1, 2, 3, 4])

    monkeypatch.setattr(adapter, "orig_paged_clear", native_clear)
    try:
        adapter._gms_paged_clear(allocator)
        assert observed["exclusive_steady_state"] is False
        assert observed["cpu_free_pages"] is None
        assert observed["active_batch"] is None
        assert state["steady_state"] is False
        assert state["exclusive_hidden_pages"] == set()
        assert state["cpu_staged_pages"] == []
        assert state["cpu_release_pages"] == []
        assert state["tp_reserved_pages"] == []
        assert leases == {}
    finally:
        _cleanup(allocator)


def test_paged_merge_noop_preserves_native_and_cpu_order(monkeypatch):
    monkeypatch.setattr(adapter, "torch", torch)
    allocator = _allocator(_Client(), {})
    allocator.free_pages = torch.tensor([3, 1, 2])
    allocator.staged_pages = []
    state = adapter._STATE[id(allocator)]
    state.update(
        exclusive_steady_state=True,
        cpu_free_pages=[3, 1, 2],
        cpu_staged_pages=[],
    )

    def native_merge(self):
        if not self.staged_pages:
            return
        self.free_pages, _ = torch.sort(
            torch.cat((self.free_pages, *self.staged_pages))
        )
        self.staged_pages = []

    monkeypatch.setattr(adapter, "orig_paged_merge_and_sort_free", native_merge)
    try:
        adapter._gms_paged_merge_and_sort_free(allocator)
        assert allocator.free_pages.tolist() == [3, 1, 2]
        assert state["cpu_free_pages"] == [3, 1, 2]

        allocator.staged_pages = [torch.tensor([4, 0])]
        state["cpu_staged_pages"] = [4, 0]
        adapter._gms_paged_merge_and_sort_free(allocator)
        assert allocator.free_pages.tolist() == [0, 1, 2, 3, 4]
        assert state["cpu_free_pages"] == [0, 1, 2, 3, 4]
        assert state["cpu_staged_pages"] == []
    finally:
        _cleanup(allocator)
