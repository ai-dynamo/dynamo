# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.vllm import install_kv_leases as hooks


def block(slot, key=None, refs=1):
    return SimpleNamespace(block_id=slot, block_hash=key, ref_cnt=refs, is_null=False)


def setup(monkeypatch, update):
    events = []

    class NativePool:
        def take_events(self):
            events.append("events")
            return list(events)

    monkeypatch.setattr(hooks, "_initialize_gms_block_pool", lambda self: None)
    pool = hooks._build_gms_block_pool_class(NativePool)()
    pool._gms_kv_lease_client = SimpleNamespace(
        release=lambda leases: events.append(("release", list(leases)))
    )
    pool._gms_kv_leases_by_block = {i: KVLease(i, 7) for i in (1, 2, 3)}
    pool._gms_kv_read_pins_by_block = {}
    pool._gms_kv_directory = SimpleNamespace(authoritative=True)
    pool.enable_caching = True
    pool.free_block_queue = SimpleNamespace(
        prepend_n=lambda blocks: None,
        append_n=lambda blocks: events.append(("free", [b.block_id for b in blocks])),
    )
    monkeypatch.setattr(
        hooks,
        "_publish_hbm_blocks",
        lambda _, blocks, **kw: (
            events.append(("publish", [b.block_id for b in blocks])) or True
        ),
    )
    monkeypatch.setattr(hooks, "_reserve_dormant_headroom", lambda *_: 0)

    def native_init(scheduler):
        scheduler.kv_cache_manager = SimpleNamespace(block_pool=pool)
        scheduler.max_num_running_reqs = 8
        scheduler.update_from_output = lambda: update(pool, events)

    monkeypatch.setattr(hooks, "_original_scheduler_init", native_init)
    scheduler = SimpleNamespace()
    hooks._scheduler_init_with_gms_completion_fence(scheduler)
    return scheduler, pool, events


def test_completed_inflight_full_blocks_publish_after_gpu_completion(monkeypatch):
    first = SimpleNamespace(
        block_id=1,
        block_hash=b"first",
        block_hash_num_tokens=16,
        ref_cnt=1,
        is_null=False,
    )
    completed = SimpleNamespace(
        block_id=2,
        block_hash=b"completed",
        block_hash_num_tokens=32,
        ref_cnt=1,
        is_null=False,
    )
    optimistic = SimpleNamespace(
        block_id=3,
        block_hash=b"optimistic",
        block_hash_num_tokens=48,
        ref_cnt=1,
        is_null=False,
    )
    partial = SimpleNamespace(
        block_id=4,
        block_hash=b"partial",
        block_hash_num_tokens=24,
        ref_cnt=1,
        is_null=False,
    )
    leases = {slot: KVLease(slot, 7) for slot in range(1, 5)}
    first_key = hooks._directory_key(first.block_hash)
    pool = SimpleNamespace(
        _gms_kv_directory=SimpleNamespace(enabled=True, authoritative=True),
        _gms_kv_leases_by_block=leases,
        _gms_kv_directory_slot_by_hash={first_key: leases[1]},
    )
    cache_manager = SimpleNamespace(
        block_size=16,
        req_to_blocks={"request": [first, completed, optimistic, partial]},
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            block_pool=pool,
            coordinator=SimpleNamespace(single_type_managers=[cache_manager]),
        ),
        requests={
            "request": SimpleNamespace(
                num_computed_tokens=48,
                num_in_flight_tokens=16,
                is_finished=lambda: False,
            )
        },
    )
    published = []

    def publish(_pool, blocks, **kwargs):
        published.append(([value.block_id for value in blocks], kwargs))
        return True

    monkeypatch.setattr(hooks, "_publish_hbm_blocks", publish)
    count = hooks._publish_completed_inflight_blocks(
        scheduler, SimpleNamespace(num_scheduled_tokens={"request": 16})
    )

    assert count == 1
    assert published == [([2], {"active": False, "release_duplicates": False})]


def test_incremental_publication_does_not_release_live_duplicate():
    content = b"same-native-block-hash"
    existing_lease = KVLease(2, 7)
    live_lease = KVLease(3, 11)
    released = []
    evicted = []
    pool = SimpleNamespace(
        enable_caching=True,
        _gms_kv_directory=SimpleNamespace(enabled=True),
        _gms_kv_lease_client=SimpleNamespace(
            seal=lambda _leases: None,
            release=lambda leases: released.extend(leases),
        ),
        _gms_kv_leases_by_block={2: existing_lease, 3: live_lease},
        _gms_kv_directory_slot_by_hash={hooks._directory_key(content): existing_lease},
        _maybe_evict_cached_block=lambda value: evicted.append(value.block_id),
    )
    duplicate = SimpleNamespace(block_id=3, block_hash=content)

    assert hooks._publish_hbm_blocks(
        pool, [duplicate], active=False, release_duplicates=False
    )
    assert released == []
    assert evicted == []
    assert pool._gms_kv_leases_by_block[3] == live_lease


def test_connector_eviction_demotes_sealed_block_before_native_mutation(monkeypatch):
    content = b"published-native-hash"
    old = KVLease(3, 7)
    successor = KVLease(3, 8)
    events = []
    value = SimpleNamespace(block_id=3, block_hash=content)

    class NativePool:
        def evict_blocks(self, block_ids):
            events.append(("native_evict", set(block_ids)))
            value.block_hash = None

    monkeypatch.setattr(hooks, "_initialize_gms_block_pool", lambda self: None)
    pool = hooks._build_gms_block_pool_class(NativePool)()
    pool.blocks = [
        SimpleNamespace(block_id=index, block_hash=None) for index in range(4)
    ]
    pool.blocks[3] = value
    pool._gms_kv_leases_by_block = {3: old}
    key = hooks._directory_key(content)
    pool._gms_kv_directory_slot_by_hash = {key: old}

    class Directory:
        authoritative = True

        def flush_deferred(self):
            events.append("flush")
            return True

        def ensure_hbm_capacity(self, required, *, eligible_slot_ids):
            events.append(("retire", required, eligible_slot_ids))
            return [
                {
                    "content_hash": key,
                    "engine_id": "0",
                    "slot_ids": [3],
                    "generations": [7],
                    "tier": "hbm",
                }
            ]

    class Client:
        def adopt(self, leases):
            events.append(("adopt", list(leases)))
            return [successor]

    pool._gms_kv_directory = Directory()
    pool._gms_kv_lease_client = Client()
    pool.evict_blocks({3})

    assert events == [
        "flush",
        ("retire", 1, [3]),
        ("adopt", [old]),
        ("native_evict", {3}),
    ]
    assert pool._gms_kv_leases_by_block[3] == successor
    assert pool._gms_kv_directory_slot_by_hash == {}
    assert value.block_hash is None


def test_prefix_reset_retires_and_releases_all_sealed_blocks(monkeypatch):
    first = SimpleNamespace(block_id=1, block_hash=b"first")
    second = SimpleNamespace(block_id=2, block_hash=b"second")
    leases = {1: KVLease(1, 3), 2: KVLease(2, 5)}
    events = []

    class NativePool:
        num_gpu_blocks = 3

        def get_num_free_blocks(self):
            return 2

        def reset_prefix_cache(self):
            events.append("native_reset")
            for value in self.blocks:
                value.block_hash = None
            return True

    monkeypatch.setattr(hooks, "_initialize_gms_block_pool", lambda self: None)
    pool = hooks._build_gms_block_pool_class(NativePool)()
    pool.blocks = [SimpleNamespace(block_id=0, block_hash=None), first, second]
    pool._gms_kv_leases_by_block = dict(leases)
    pool._gms_kv_directory_slot_by_hash = {
        hooks._directory_key(first.block_hash): leases[1],
        hooks._directory_key(second.block_hash): leases[2],
    }
    monkeypatch.setattr(
        hooks,
        "_demote_sealed_blocks_for_mutation",
        lambda owner, block_ids: (
            events.append(("demote", set(block_ids))),
            owner._gms_kv_directory_slot_by_hash.clear(),
        ),
    )
    pool._gms_kv_lease_client = SimpleNamespace(
        release=lambda values: events.append(("release", list(values)))
    )

    assert pool.reset_prefix_cache()
    assert events == [
        ("demote", {1, 2}),
        ("release", [leases[1], leases[2]]),
        "native_reset",
    ]
    assert pool._gms_kv_leases_by_block == {}


def test_completed_requests_commit_once_before_native_events_and_output(monkeypatch):
    shared = block(1, b"shared", refs=2)
    other = block(2, b"other")

    def update(pool, events):
        pool.free_blocks([shared])
        pool.free_blocks([shared, other])
        assert events == []
        assert shared.ref_cnt == 2
        pool.take_events()
        events.append("output")
        return "done"

    scheduler, pool, events = setup(monkeypatch, update)
    assert scheduler.update_from_output() == "done"
    assert [
        event for event in events if isinstance(event, tuple) and event[0] == "publish"
    ] == [("publish", [1, 2])]
    assert events.index("events") < events.index("output")
    assert shared.ref_cnt == other.ref_cnt == 0
    assert pool._gms_completed_frees is None


def test_failed_native_completion_does_not_publish_or_free(monkeypatch):
    value = block(1, b"kv")

    def update(pool, events):
        pool.free_blocks([value])
        raise RuntimeError("native completion failed")

    scheduler, pool, events = setup(monkeypatch, update)
    with pytest.raises(RuntimeError, match="native completion failed"):
        scheduler.update_from_output()
    assert events == []
    assert value.ref_cnt == 1
    assert 1 in pool._gms_kv_leases_by_block


def test_unexpected_allocation_during_completion_fails_closed(monkeypatch):
    def update(pool, _events):
        pool.get_new_blocks(1)

    scheduler, _, _ = setup(monkeypatch, update)
    with pytest.raises(RuntimeError, match="cannot allocate"):
        scheduler.update_from_output()


def test_conflicting_hash_copies_keep_separate_publications(monkeypatch):
    groups = [[block(1, b"same")], [block(2, b"same")]]
    pool = SimpleNamespace(_gms_completed_frees=groups)
    committed = []
    monkeypatch.setattr(
        hooks,
        "_free_blocks",
        lambda _, blocks, **kw: committed.append([b.block_id for b in blocks]),
    )
    hooks._flush_completed_frees(pool)
    assert committed == [[1], [2]]


def test_return_without_event_drain_still_commits_before_return(monkeypatch):
    value = block(1, b"kv")

    def update(pool, _events):
        pool.free_blocks([value])
        return "output"

    scheduler, _, events = setup(monkeypatch, update)
    assert scheduler.update_from_output() == "output"
    assert ("publish", [1]) in events
    assert value.ref_cnt == 0


def test_lost_publication_ack_keeps_sealed_ownership_and_withholds_output(monkeypatch):
    actual_publish = hooks._publish_hbm_blocks
    value = block(1, b"k" * 32)
    committed = []

    def update(pool, events):
        pool.free_blocks([value])
        pool.take_events()
        events.append("output")

    scheduler, pool, events = setup(monkeypatch, update)
    monkeypatch.setattr(hooks, "_publish_hbm_blocks", actual_publish)
    pool._gms_kv_lease_client.seal = lambda leases: events.append(
        ("seal", list(leases))
    )

    def commit_then_lose_reply(items):
        committed.extend(items)
        raise TimeoutError("commit succeeded, acknowledgement was lost")

    pool._gms_kv_directory = SimpleNamespace(
        authoritative=True, enabled=True, publish=commit_then_lose_reply
    )
    with pytest.raises(RuntimeError, match="retaining sealed leases"):
        scheduler.update_from_output()
    assert len(committed) == 1
    assert pool._gms_kv_leases_by_block[1] == KVLease(1, 7)
    assert value.block_hash == b"k" * 32
    assert "output" not in events
    assert not any(
        isinstance(event, tuple) and event[0] in ("release", "free") for event in events
    )
