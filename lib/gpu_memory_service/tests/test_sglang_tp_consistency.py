# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-rank pressure invariants, extended by the later TP integration."""

from types import SimpleNamespace

import pytest
import torch
from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.sglang import install_kv_leases as hooks

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize("native_free", [False, True])
def test_pressure_never_retires_retained_pages_still_in_native_tree(
    monkeypatch, native_free
):
    selected, released = [], []
    allocator = SimpleNamespace(
        free_pages=torch.tensor([4] if native_free else [], dtype=torch.int64)
    )
    victim = {
        "content_hash": b"h" * 32,
        "engine_id": "engine-0",
        "slot_ids": [4],
        "generations": [2],
    }

    def select(count, *, eligible_slot_ids):
        selected.append((count, eligible_slot_ids))
        return [victim]

    client = SimpleNamespace(
        free_count=lambda: 0,
        adopt=lambda leases: [
            KVLease(lease.block_id, lease.generation + 1) for lease in leases
        ],
        release=lambda leases: released.extend(leases),
    )
    state = {
        "client": client,
        "leases_by_page": {4: KVLease(4, 2)},
        "retained_pages": {4},
    }
    allocator._gms_kv_directory = SimpleNamespace(
        authoritative=True, ensure_hbm_capacity=select
    )
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    assert hooks._ensure_directory_capacity(allocator, 1) == int(native_free)
    assert selected == ([(1, [4])] if native_free else [])
    assert released == ([KVLease(4, 3)] if native_free else [])
    assert state["retained_pages"] == (set() if native_free else {4})


def test_pressure_generation_failure_preserves_lease_and_restores_directory(
    monkeypatch,
):
    allocator = SimpleNamespace(free_pages=torch.tensor([4]))
    released, restored = [], []
    victim = {
        "content_hash": b"h" * 32,
        "engine_id": "engine-0",
        "slot_ids": [4],
        "generations": [2],
    }
    client = SimpleNamespace(
        free_count=lambda: 0, release=lambda leases: released.extend(leases)
    )
    state = {
        "client": client,
        "leases_by_page": {4: KVLease(4, 9)},
        "retained_pages": {4},
    }
    allocator._gms_kv_directory = SimpleNamespace(
        authoritative=True,
        ensure_hbm_capacity=lambda *_args, **_kwargs: [victim],
        publish=lambda items: restored.extend(items) or len(items),
        lookup_authoritative=lambda _hashes: [victim],
    )
    monkeypatch.setitem(hooks._STATE, id(allocator), state)

    with pytest.raises(RuntimeError, match="divergent retained generation"):
        hooks._ensure_directory_capacity(allocator, 1)
    assert released == []
    assert len(restored) == 1
    assert state["leases_by_page"] == {4: KVLease(4, 9)}
    assert state["retained_pages"] == {4}
