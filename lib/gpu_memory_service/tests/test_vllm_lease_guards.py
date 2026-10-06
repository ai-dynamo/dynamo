# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards that keep a block from reaching a second consumer while in use.

Each test fails if its guard is removed: the failure mode is silent KV
corruption (wrong tokens), not an error, so the suite must pin them down.
"""

from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.vllm import install_kv_leases as leases_mod

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


class _Client:
    def __init__(self, free=10):
        self.free = free
        self.released = []
        self.sealed = []
        self.adopt_result = None

    def free_count(self):
        return self.free

    def release(self, leases):
        self.released.extend(leases)

    def seal(self, leases):
        self.sealed.extend(leases)

    def adopt(self, old):
        return self.adopt_result


def _block(block_id, ref_cnt=0, block_hash=None):
    return SimpleNamespace(
        block_id=block_id,
        ref_cnt=ref_cnt,
        block_hash=block_hash,
        is_null=False,
        next_free_block=None,
    )


def test_dormant_eviction_never_releases_a_block_in_use():
    """A victim still read by a running request keeps its lease."""
    busy = _block(1, ref_cnt=1, block_hash=b"h1")
    idle = _block(2, ref_cnt=0, block_hash=b"h2")
    published = []
    directory = SimpleNamespace(
        enabled=True,
        ensure_hbm_capacity=lambda _n, eligible_slot_ids, engine_id=None: [
            {"slot_ids": [1], "generations": [5]},
            {"slot_ids": [2], "generations": [7]},
        ],
        publish=lambda items: published.extend(items) or len(items),
    )
    client = _Client()
    busy_lease, idle_lease = KVLease(1, 5), KVLease(2, 7)
    evicted = []
    pool = SimpleNamespace(
        _gms_kv_directory=directory,
        _gms_kv_lease_client=client,
        num_gpu_blocks=4,
        free_block_queue=SimpleNamespace(get_all_free_blocks=lambda: [idle]),
        blocks=[_block(0), busy, idle],
        _gms_kv_leases_by_block={1: busy_lease, 2: idle_lease},
        _gms_kv_directory_slot_by_hash={},
        enable_caching=True,
        _maybe_evict_cached_block=evicted.append,
    )

    released = leases_mod._evict_dormant_directory_blocks(pool, 2, [busy])

    assert released == 1
    assert client.released == [idle_lease]
    assert pool._gms_kv_leases_by_block == {1: busy_lease}
    assert evicted == [idle]
    # The in-use block is republished as ACTIVE, so no peer can adopt it.
    assert [item["slot_id"] for item in published] == [1]
    assert published[0]["active"] is True


def test_adoption_never_hands_out_a_locally_busy_slot(monkeypatch):
    """A directory slot that is busy locally is not installed for a request."""
    busy = _block(3, ref_cnt=1)
    client = _Client()
    client.adopt_result = [
        KVLease(3, leases_mod._successor_generation(4)),
    ]
    dropped = []
    directory = SimpleNamespace(
        enabled=True,
        authoritative=True,
        read_view_is_current_writer=False,
        lookup_and_claim=lambda keys: (
            [{"tier": "hbm", "slot_ids": [3], "generations": [4]}],
            "token",
        ),
        adopt_claim=lambda _token, items: len(items),
        release_claim=lambda _token: None,
        publish=lambda items: dropped.extend(items) or len(items),
    )
    inserted = []
    pool = SimpleNamespace(
        _gms_kv_directory=directory,
        _gms_kv_lease_client=client,
        _gms_hydrate_hbm=False,
        blocks=[_block(0), _block(1), _block(2), busy],
        hash_block_size=16,
        _gms_kv_leases_by_block={},
        _gms_kv_directory_slot_by_hash={},
        _insert_block_hash=lambda *args: inserted.append(args),
        _maybe_evict_cached_block=lambda _block: None,
    )
    monkeypatch.delenv("DYN_GMS_FAILOVER_FROZEN_PREDECESSOR", raising=False)

    result = leases_mod._get_cached_block(pool, lambda *_args: None, b"\x01" * 32, [0])

    assert result is None
    assert inserted == []
    assert pool._gms_kv_leases_by_block == {}
    assert client.released == client.adopt_result


def test_free_block_count_is_bounded_by_the_lease_ring():
    """Admission control: locally free blocks are not all acquirable."""
    pool = SimpleNamespace(_gms_kv_lease_client=_Client(free=3))
    assert leases_mod._get_num_free_blocks(pool, lambda: 10) == 3
    pool._gms_kv_lease_client.free = 50
    assert leases_mod._get_num_free_blocks(pool, lambda: 10) == 10
