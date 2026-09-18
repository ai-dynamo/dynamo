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
