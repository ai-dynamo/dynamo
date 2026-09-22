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
