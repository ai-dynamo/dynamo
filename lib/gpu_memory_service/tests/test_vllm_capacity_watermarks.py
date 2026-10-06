# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.vllm import install_kv_leases as hooks

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_zero_block_allocation_never_touches_ownership_or_capacity():
    def unexpected():
        raise AssertionError("zero-block allocation consulted capacity")

    assert hooks._get_new_blocks(object(), unexpected, 0) == []


def test_small_pool_keeps_concurrent_headroom_with_hysteresis(monkeypatch):
    monkeypatch.delenv("GMS_VLLM_DORMANT_HEADROOM_BLOCKS", raising=False)
    free = [200]
    evicted = []
    pool = SimpleNamespace(
        num_gpu_blocks=4096,
        _gms_admission_concurrency=8,
        _gms_kv_directory=SimpleNamespace(authoritative=True),
        _gms_kv_lease_client=SimpleNamespace(free_count=lambda: free[0]),
    )
    monkeypatch.setattr(
        hooks, "_evict_dormant_directory_blocks", lambda _, n: evicted.append(n) or n
    )
    assert hooks._reserve_dormant_headroom(pool, 28) == 248
    assert evicted == [248]
    free[0] = 300  # below high watermark, but no synchronous refill needed
    assert hooks._reserve_dormant_headroom(pool, 28) == 0
    assert evicted == [248]


def test_concurrency_reserve_does_not_erase_entire_prefix_cache(monkeypatch):
    monkeypatch.delenv("GMS_VLLM_DORMANT_HEADROOM_BLOCKS", raising=False)
    pool = SimpleNamespace(
        num_gpu_blocks=4096,
        _gms_admission_concurrency=1024,
        _gms_kv_directory=SimpleNamespace(authoritative=True),
        _gms_kv_lease_client=SimpleNamespace(free_count=lambda: 0),
    )
    monkeypatch.setattr(hooks, "_evict_dormant_directory_blocks", lambda _, n: n)
    assert hooks._reserve_dormant_headroom(pool, 28) == 1024


def _linked_queue(block_ids):
    node = SimpleNamespace(next_free_block=None)
    for block_id in reversed(block_ids):
        node = SimpleNamespace(block_id=block_id, is_null=False, next_free_block=node)
    return SimpleNamespace(fake_free_list_head=SimpleNamespace(next_free_block=node))


def test_allocation_prefers_unleased_free_blocks(monkeypatch):
    """A leased head would send the lease ring into a full linear scan."""
    pool = SimpleNamespace(
        free_block_queue=_linked_queue([5, 6, 7, 8, 9]),
        _gms_kv_leases_by_block={5: object(), 6: object(), 8: object()},
    )
    # Not the head prefix, so the caller must not popleft_n these.
    assert hooks._preferred_unleased_block_ids(pool, 2) == ([7, 9], False)

    # Beyond the scan window, keep the plain head prefix.
    monkeypatch.setattr(hooks, "_PREFERRED_UNLEASED_SCAN", 2)
    assert hooks._preferred_unleased_block_ids(pool, 1) == ([5], True)


def test_capacity_retires_only_native_lru_candidates():
    blocks = {
        slot: SimpleNamespace(
            block_id=slot,
            block_hash=None if slot == 1 else bytes([slot]),
            ref_cnt=0,
            is_null=False,
        )
        for slot in range(6)
    }
    # Slots 2 and 3 are the oldest native cached blocks. Slot 4 is a newer hot
    # prefix, while slot 5 is active and therefore absent from the free queue.
    free_queue = SimpleNamespace(
        get_all_free_blocks=lambda: [blocks[i] for i in (1, 2, 3, 4)]
    )
    seen = []

    class Directory:
        enabled = True

        def ensure_hbm_capacity(
            self, required, *, eligible_slot_ids=None, engine_id=None
        ):
            seen.append((required, eligible_slot_ids))
            return [
                {
                    "slot_ids": [2],
                    "generations": [7],
                }
            ]

    released = []
    pool = SimpleNamespace(
        num_gpu_blocks=6,
        blocks=blocks,
        free_block_queue=free_queue,
        _gms_kv_directory=Directory(),
        _gms_kv_lease_client=SimpleNamespace(
            release=lambda leases: released.extend(leases)
        ),
        _gms_kv_leases_by_block={slot: KVLease(slot, 7) for slot in (2, 3, 4, 5)},
        _maybe_evict_cached_block=lambda block: setattr(block, "block_hash", None),
    )

    assert hooks._evict_dormant_directory_blocks(pool, 1) == 1
    assert seen == [(1, [2, 3, 4])]
    assert released == [KVLease(2, 7)]
    assert blocks[2].block_hash is None
    assert blocks[4].block_hash == bytes([4])


def test_duplicate_content_keeps_one_durable_slot_and_releases_the_other():
    content = b"same-native-block-hash"
    first = SimpleNamespace(block_id=2, block_hash=content)
    duplicate = SimpleNamespace(block_id=3, block_hash=content)
    first_lease = KVLease(2, 7)
    duplicate_lease = KVLease(3, 11)
    sealed = []
    released = []
    publications = []

    class Client:
        def seal(self, leases):
            sealed.extend(leases)

        def release(self, leases):
            released.extend(leases)

    class Directory:
        enabled = True

        def publish(self, _items):
            raise AssertionError("synchronous publication used")

        def publish_deferred(self, items):
            publications.append(items)
            return len(items)

    pool = SimpleNamespace(
        enable_caching=True,
        _gms_kv_directory=Directory(),
        _gms_kv_lease_client=Client(),
        _gms_kv_leases_by_block={2: first_lease, 3: duplicate_lease},
        _gms_kv_directory_slot_by_hash={},
        _maybe_evict_cached_block=lambda block: setattr(block, "block_hash", None),
    )

    assert hooks._publish_hbm_blocks(pool, [first], active=False)
    assert hooks._publish_hbm_blocks(pool, [duplicate], active=False)

    assert sealed == [first_lease]
    assert released == [duplicate_lease]
    assert len(publications) == 1
    assert duplicate.block_hash is None
    assert pool._gms_kv_leases_by_block == {2: first_lease}


def test_async_capacity_retirement_runs_off_the_engine_thread(monkeypatch):
    """Opt-in retirement returns at once and applies only exact victims later."""
    import threading

    monkeypatch.setenv("DYN_GMS_ASYNC_DIRECTORY_WORK", "1")
    monkeypatch.setenv("GMS_VLLM_DORMANT_HEADROOM_BLOCKS", "2")
    blocks = {
        slot: SimpleNamespace(
            block_id=slot, block_hash=bytes([slot]), ref_cnt=0, is_null=False
        )
        for slot in range(5)
    }
    gate = threading.Event()
    calls = []

    class Directory:
        enabled = True
        authoritative = True

        def ensure_hbm_capacity(
            self, required, *, eligible_slot_ids=None, engine_id=None
        ):
            calls.append((threading.current_thread().name, required, eligible_slot_ids))
            assert gate.wait(5)
            # Slot 3 was re-leased meanwhile: its retired generation is stale.
            return [
                {"slot_ids": [2], "generations": [7]},
                {"slot_ids": [3], "generations": [7]},
            ]

    released = []
    pool = SimpleNamespace(
        num_gpu_blocks=5,
        blocks=blocks,
        free_block_queue=SimpleNamespace(
            get_all_free_blocks=lambda: [blocks[i] for i in (2, 3, 4)]
        ),
        _gms_kv_directory=Directory(),
        _gms_kv_lease_client=SimpleNamespace(
            free_count=lambda: 0, release=lambda leases: released.extend(leases)
        ),
        _gms_kv_leases_by_block={
            2: KVLease(2, 7),
            3: KVLease(3, 9),
            4: KVLease(4, 7),
        },
        _gms_kv_directory_slot_by_hash={},
        _maybe_evict_cached_block=lambda block: setattr(block, "block_hash", None),
    )

    assert hooks._reserve_dormant_headroom(pool, 1) == 0
    assert hooks._reserve_dormant_headroom(pool, 1) == 0
    assert released == []
    assert hooks._collect_async_capacity(pool) == 0  # still in flight

    gate.set()
    assert hooks._collect_async_capacity(pool, wait=True) == 1
    assert len(calls) == 1
    assert calls[0][0].startswith("gms-capacity") and calls[0][1] == 2
    assert released == [KVLease(2, 7)]
    assert blocks[2].block_hash is None
    assert blocks[3].block_hash == bytes([3])
    assert 3 in pool._gms_kv_leases_by_block
    pool._gms_capacity_executor.shutdown()


def test_capacity_retirement_stays_synchronous_by_default(monkeypatch):
    monkeypatch.delenv("DYN_GMS_ASYNC_DIRECTORY_WORK", raising=False)
    assert hooks.async_capacity_retirement_enabled() is False
