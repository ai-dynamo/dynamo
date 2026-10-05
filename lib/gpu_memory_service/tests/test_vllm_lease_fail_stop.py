# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM free-path lease handling: lost ownership and native free semantics.

A failed lease seal on vLLM's free path is lost ownership, not a soft error.

The lease ring seals all-or-nothing and fails only when a successor has moved
the lease generation. The engine must stop without returning the blocks to its
free queue or releasing leases that may now back the successor's KV.
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


class _FencedClient:
    namespace = "test"
    owner_id = "slow-primary"

    def __init__(self):
        self.released = []

    def seal(self, leases):
        raise RuntimeError(f"GMS KV lease seal committed 0/{len(leases)} blocks")

    def release(self, leases):
        self.released.extend(leases)


class _Queue:
    def __init__(self):
        self.blocks = []

    def prepend_n(self, blocks):
        self.blocks[:0] = list(blocks)

    def append_n(self, blocks):
        self.blocks.extend(blocks)


def _pool(client, blocks, leases):
    return SimpleNamespace(
        _gms_kv_lease_client=client,
        _gms_kv_leases_by_block=dict(leases),
        _gms_kv_directory=SimpleNamespace(enabled=True, authoritative=True),
        enable_caching=True,
        free_block_queue=_Queue(),
        _maybe_evict_cached_block=lambda _block: None,
        blocks=blocks,
    )


def test_seal_failure_on_free_path_stops_engine_without_touching_leases():
    blocks = [
        SimpleNamespace(
            block_id=i, ref_cnt=1, block_hash=bytes([i]) * 32, is_null=False
        )
        for i in (1, 2)
    ]
    leases = {1: KVLease(1, 7), 2: KVLease(2, 9)}
    client = _FencedClient()
    pool = _pool(client, blocks, leases)

    with pytest.raises(leases_mod.GMSKVLeaseFenced) as raised:
        leases_mod._free_blocks(pool, blocks)

    assert "0/2" in str(raised.value.__cause__)
    # Nothing that may belong to a successor was handed back or released.
    assert client.released == []
    assert pool.free_block_queue.blocks == []
    assert pool._gms_kv_leases_by_block == leases


class _SealingClient:
    namespace = "test"
    owner_id = "primary"

    def __init__(self):
        self.sealed = []
        self.released = []

    def seal(self, leases):
        self.sealed.append([lease.block_id for lease in leases])

    def release(self, leases):
        self.released.extend(leases)


def test_block_listed_twice_is_freed_once_like_native_block_pool():
    """Native BlockPool classifies inside the decrement loop; so must GMS."""
    block = SimpleNamespace(
        block_id=1, ref_cnt=2, block_hash=b"\x01" * 32, is_null=False
    )
    lease = KVLease(1, 7)
    client = _SealingClient()
    pool = _pool(client, [block], {1: lease})
    pool._gms_kv_directory = None  # no directory: plain release path

    leases_mod._free_blocks(pool, [block, block])

    assert block.ref_cnt == 0
    assert pool.free_block_queue.blocks == [block]
    assert client.released == [lease]
