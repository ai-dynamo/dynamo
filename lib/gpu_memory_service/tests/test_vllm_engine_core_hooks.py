# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from gpu_memory_service.integrations.vllm.install_kv_leases import (
    install_gms_engine_core_sleep,
)


def test_sleep_utility_is_visible_on_spawned_engine_core_proc():
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    original = EngineCore.__dict__.get("gms_sleep_no_clear")
    if original is not None:
        delattr(EngineCore, "gms_sleep_no_clear")
    try:
        assert install_gms_engine_core_sleep()
        assert "gms_sleep_no_clear" in EngineCore.__dict__
        assert hasattr(EngineCoreProc, "gms_sleep_no_clear")
        assert not install_gms_engine_core_sleep()
    finally:
        if hasattr(EngineCore, "gms_sleep_no_clear"):
            delattr(EngineCore, "gms_sleep_no_clear")
        if original is not None:
            EngineCore.gms_sleep_no_clear = original


def test_block_pool_hbm_directory_survives_engine_replacement(monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace

    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease
    from vllm.v1.core import kv_cache_coordinator
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    original_block_pool_binding = kv_cache_coordinator.BlockPool
    original_block_pool_methods = dict(BlockPool.__dict__)
    original_allocate_slots = KVCacheManager.allocate_slots
    original_scheduler_init = Scheduler.__init__
    original_directory = leases_mod.ContentDirectory
    original_factory = leases_mod._factory
    original_patched = leases_mod._patched

    class LeaseState:
        free = set(range(1, 8))
        generations = defaultdict(int)
        held = {}
        seal_batches = []
        readers = defaultdict(int)

    state = LeaseState()
    owners = iter(("primary", "reader", "shadow"))

    class Client:
        namespace = "test"

        def __init__(self):
            self.owner_id = next(owners)

        def free_count(self):
            return len(state.free)

        def acquire(self, count, preferred_blocks=None, **_kwargs):
            available = [
                block
                for block in (preferred_blocks or sorted(state.free))
                if block in state.free
            ]
            if len(available) < count:
                raise RuntimeError("no lease")
            result = []
            for block in available[:count]:
                state.free.remove(block)
                state.generations[block] += 1
                generation = state.generations[block]
                state.held[block] = (generation, self.owner_id)
                result.append(KVLease(block, generation))
            return result

        def seal(self, leases):
            state.seal_batches.append([lease.block_id for lease in leases])

        def pin_read(self, leases):
            if any(
                state.held.get(lease.block_id, (None,))[0] != lease.generation
                for lease in leases
            ):
                return None
            for lease in leases:
                state.readers[lease.block_id] += 1
            return tuple(leases)

        def unpin_read(self, claim):
            for lease in claim:
                state.readers[lease.block_id] -= 1

        def release(self, released):
            for lease in released:
                current = state.held.get(lease.block_id)
                if current is not None and current[0] == lease.generation:
                    state.held.pop(lease.block_id)
                    state.free.add(lease.block_id)

        def adopt(self, old):
            if any(
                state.held.get(lease.block_id, (None,))[0] != lease.generation
                for lease in old
            ):
                return []
            result = []
            for lease in old:
                state.generations[lease.block_id] += 1
                generation = state.generations[lease.block_id]
                state.held[lease.block_id] = (generation, self.owner_id)
                result.append(KVLease(lease.block_id, generation))
            return result

    class Directory:
        enabled = True

        @property
        def authoritative(self):
            return self.mode == "authoritative" or (
                self.mode == "shadow" and self.read_view_is_current_writer
            )

        mode = "authoritative"
        read_view_is_current_writer = True

        def __init__(self):
            self.entries = {}
            self.ensure_calls = []
            self.fail_publish = False

        def publish(self, items):
            if self.fail_publish:
                raise RuntimeError("directory unavailable")
            for item in items:
                content_hash = item["content_hash"]
                if not item.get("sealed", True):
                    self.entries.pop(content_hash, None)
                    continue
                slots = item.get("slot_ids")
                if slots is None:
                    slots = [item["slot_id"]]
                generations = item.get("generations")
                if generations is None:
                    generations = [item.get("generation", 0)]
                self.entries[content_hash] = {
                    **item,
                    "slot_ids": slots,
                    "generations": generations,
                    "state": "active" if item.get("active") else "ready",
                }
            return len(items)

        def read_view_items(self, *, tier=None, state="ready", limit=None):
            items = [
                (key, entry)
                for key, entry in self.entries.items()
                if (not state or entry.get("state") == state)
                and (tier is None or entry.get("tier") == tier)
            ]
            return items if limit is None else items[:limit]

        def lookup_and_claim(self, hashes):
            return [
                (
                    self.entries.get(value)
                    if self.entries.get(value, {}).get("state") in {"ready", "active"}
                    else None
                )
                for value in hashes
            ], "claim"

        def lookup_and_read_claim(self, hashes):
            return self.lookup_and_claim(hashes)

        def release_claim(self, _token):
            return True

        def adopt_claim(self, _token, items):
            for item in items:
                entry = self.entries[item["content_hash"]]
                entry["generations"] = item["generations"]
                entry["state"] = "active"
            return len(items)

        def mark_hbm_dormant(self, hashes):
            for value in hashes:
                self.entries[value]["state"] = "ready"
            return len(hashes)

        def ensure_hbm_capacity(self, required, *, eligible_slot_ids=None):
            self.ensure_calls.append(required)
            eligible = None if eligible_slot_ids is None else set(eligible_slot_ids)
            victims = []
            freed = 0
            for content_hash, entry in list(self.entries.items()):
                if entry.get("tier") != "hbm" or entry.get("state") != "ready":
                    continue
                if eligible is not None and not set(entry["slot_ids"]).issubset(
                    eligible
                ):
                    continue
                victims.append(
                    {
                        "content_hash": content_hash,
                        "slot_ids": list(entry["slot_ids"]),
                        "generations": list(entry["generations"]),
                    }
                )
                freed += len(entry["slot_ids"])
                del self.entries[content_hash]
                if freed >= required:
                    break
            return victims

        def close(self):
            return None

    directory = Directory()
    monkeypatch.setenv("GMS_VLLM_HYDRATE_HBM", "1")
    try:
        leases_mod._patched = False
        leases_mod._factory = None
        leases_mod.ContentDirectory = lambda *_args, **_kwargs: directory
        assert leases_mod.install(factory=lambda _total: Client())
        GMSBlockPool = kv_cache_coordinator.BlockPool
        assert GMSBlockPool is leases_mod._gms_block_pool_class
        assert issubclass(GMSBlockPool, BlockPool)
        assert dict(BlockPool.__dict__) == original_block_pool_methods

        primary = GMSBlockPool(8, True, 4)
        blocks = primary.get_new_blocks(2)
        primary.cache_full_blocks(
            SimpleNamespace(block_hashes=[b"a" * 32, b"b" * 32]),
            blocks,
            0,
            2,
            4,
            0,
        )
        native_keys = [bytes(block.block_hash) for block in blocks]
        assert all(len(key) == 36 for key in native_keys)
        content_hashes = [leases_mod._directory_key(key) for key in native_keys]
        # Full-block hashing alone is not a durability event: incomplete or
        # still-referenced KV must never be advertised to a replacement.
        assert directory.entries == {}
        primary.free_blocks(blocks)
        assert state.seal_batches == [[block.block_id for block in blocks]]
        assert all(directory.entries[key]["state"] == "ready" for key in content_hashes)
        assert all(block.block_id not in state.free for block in blocks)

        # A standby may consume immutable READY blocks before writer promotion.
        # It holds both the directory claim and exact ring generation until
        # vLLM's completion-fenced free callback proves GPU reads are done.
        directory.mode = "shadow"
        directory.read_view_is_current_writer = False
        reader = GMSBlockPool(8, True, 4)
        borrowed = reader.get_cached_block(b"a" * 32, [0])
        assert borrowed is not None
        borrowed_block = borrowed[0]
        assert state.held[borrowed_block.block_id][1] == "primary"
        assert state.readers[borrowed_block.block_id] == 1
        reader.free_block_queue.remove(borrowed_block)
        borrowed_block.ref_cnt += 1
        reader.free_blocks([borrowed_block])
        assert state.readers[borrowed_block.block_id] == 0
        assert borrowed_block.block_hash is None
        assert borrowed_block in reader.free_block_queue.get_all_free_blocks()

        directory.mode = "authoritative"
        directory.read_view_is_current_writer = True

        # A stale snapshot member must not block recovery of the valid sibling.
        stale_native_key = b"stale-directory-entry" + b"\0" * 15
        stale_hash = leases_mod._directory_key(stale_native_key)
        state.free.remove(7)
        state.held[7] = (99, "dead")
        directory.entries[stale_hash] = {
            "tier": "hbm",
            "state": "ready",
            "slot_ids": [7],
            "generations": [1],
            "engine_id": "0",
            "local_key": stale_native_key,
        }

        shadow = GMSBlockPool(8, True, 4)
        recovered = shadow.get_cached_block(b"a" * 32, [0])
        assert recovered is not None
        assert recovered[0].block_id == blocks[0].block_id
        assert directory.entries[content_hashes[0]]["state"] == "active"
        assert state.held[blocks[0].block_id][1] == "shadow"

        # The non-requested sibling was hydrated into vLLM native state, so a
        # subsequent lookup is local; the stale record was invalidated only.
        hydrated = shadow.get_cached_block(b"b" * 32, [0])
        assert hydrated is not None
        assert hydrated[0].block_id == blocks[1].block_id
        assert state.held[blocks[1].block_id][1] == "shadow"
        assert shadow.free_block_queue.get_all_free_blocks()[-1] is hydrated[0]
        assert stale_hash not in directory.entries
        assert shadow._gms_hydrate_hbm is False

        from vllm.v1.core import kv_cache_utils

        monkeypatch.setattr(
            kv_cache_utils,
            "make_block_hash_with_group_id",
            lambda *_args: pytest.fail(
                "current writer constructed directory keys after hydration"
            ),
        )
        lookup = directory.lookup_and_claim
        directory.lookup_and_claim = lambda _keys: pytest.fail(
            "current writer queried the recovery directory after hydration"
        )
        try:
            assert shadow.get_cached_block(b"new-miss" * 4, [0]) is None
        finally:
            directory.lookup_and_claim = lookup

        # Saturating the shared ring retires READY entries at finalization, so
        # the next request can allocate from local shared memory without a
        # synchronous directory-capacity RPC. Directory visibility disappears
        # before native hashes and leases are released.
        pressure_blocks = shadow.get_new_blocks(4)
        shadow.cache_full_blocks(
            SimpleNamespace(
                block_hashes=[bytes([value]) * 32 for value in range(3, 7)]
            ),
            pressure_blocks,
            0,
            4,
            4,
            0,
        )
        pressure_hashes = [
            leases_mod._directory_key(block.block_hash) for block in pressure_blocks
        ]
        shadow.free_blocks(pressure_blocks)
        assert directory.ensure_calls == [4]
        assert len(state.free) == 4
        advertised_slots = {
            slot
            for entry in directory.entries.values()
            for slot in entry.get("slot_ids", [])
        }
        assert advertised_slots.isdisjoint(state.free)
        assert all(
            shadow.blocks[block_id].block_hash is None for block_id in state.free
        )
        assert (
            sum(
                content_hash not in directory.entries
                for content_hash in pressure_hashes
            )
            == 3
        )

        # A lost publication reply may hide a committed, adoptable record.
        # Fail closed with the lease retained, never make that slot reusable.
        fallback = shadow.get_new_blocks(1)
        shadow.cache_full_blocks(
            SimpleNamespace(block_hashes=[b"z" * 32]), fallback, 0, 1, 4, 0
        )
        directory.fail_publish = True
        with pytest.raises(RuntimeError, match="retaining sealed leases"):
            shadow.free_blocks(fallback)
        assert fallback[0].block_id not in state.free
        assert fallback[0].block_id in shadow._gms_kv_leases_by_block
        assert fallback[0] not in shadow.free_block_queue.get_all_free_blocks()
    finally:
        kv_cache_coordinator.BlockPool = original_block_pool_binding
        KVCacheManager.allocate_slots = original_allocate_slots
        Scheduler.__init__ = original_scheduler_init
        leases_mod.ContentDirectory = original_directory
        leases_mod._factory = original_factory
        leases_mod._patched = original_patched
        leases_mod._gms_block_pool_class = None


def test_dormant_headroom_preserves_concurrent_admission(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    evictions = []
    pool = SimpleNamespace(
        num_gpu_blocks=10_000,
        _gms_kv_directory=SimpleNamespace(authoritative=True),
        _gms_kv_lease_client=SimpleNamespace(free_count=lambda: 20),
    )
    monkeypatch.setattr(
        leases_mod,
        "_evict_dormant_directory_blocks",
        lambda _pool, count: evictions.append(count) or count,
    )

    assert leases_mod._reserve_dormant_headroom(pool, 8) == 180
    assert evictions == [180]
    monkeypatch.setenv("GMS_VLLM_DORMANT_HEADROOM_BLOCKS", "256")
    assert leases_mod._reserve_dormant_headroom(pool, 8) == 236
    assert evictions == [180, 236]


def test_completed_hbm_blocks_use_daemon_owned_publication_pipeline():
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease

    lease = KVLease(7, 11)
    client = SimpleNamespace(seal=MagicMock())
    directory = SimpleNamespace(
        enabled=True,
        publish_deferred=MagicMock(return_value=1),
        publish=MagicMock(side_effect=AssertionError("synchronous publish used")),
    )
    pool = SimpleNamespace(
        _gms_kv_directory=directory,
        _gms_kv_lease_client=client,
        _gms_kv_leases_by_block={7: lease},
    )
    block = SimpleNamespace(block_id=7, block_hash=b"native-hash")

    assert leases_mod._publish_hbm_blocks(pool, [block], active=False) is True

    client.seal.assert_called_once_with([lease])
    directory.publish_deferred.assert_called_once()
    item = directory.publish_deferred.call_args.args[0][0]
    assert item["slot_id"] == 7
    assert item["generation"] == 11
    assert item["active"] is False
    directory.publish.assert_not_called()


def test_scheduler_completion_fence_uses_native_deferred_free(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    def native_init(instance, marker):
        instance.native_marker = marker
        instance.defer_block_free = False

    monkeypatch.setattr(leases_mod, "_original_scheduler_init", native_init)
    scheduler = SimpleNamespace()
    leases_mod._scheduler_init_with_gms_completion_fence(scheduler, "initialized")

    assert scheduler.native_marker == "initialized"
    assert scheduler.defer_block_free is True


def test_allocate_slots_translates_atomic_lease_race_to_backpressure(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    def contend(*_args, **_kwargs):
        raise leases_mod.GMSKVLeaseUnavailable("lease claimed after free-count check")

    monkeypatch.setattr(leases_mod, "orig_allocate_slots", contend)
    coordinator = SimpleNamespace(
        single_type_managers=[SimpleNamespace(req_to_blocks={}, num_cached_block={})],
        free=MagicMock(),
    )
    manager = SimpleNamespace(coordinator=coordinator)
    request = SimpleNamespace(request_id="new-request")

    assert leases_mod.patched_allocate_slots(manager, request) is None
    coordinator.free.assert_called_once_with("new-request")


def test_allocate_slots_fails_closed_for_mutated_running_request(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    def contend(*_args, **_kwargs):
        raise leases_mod.GMSKVLeaseUnavailable("lease claimed after free-count check")

    monkeypatch.setattr(leases_mod, "orig_allocate_slots", contend)
    coordinator = SimpleNamespace(
        single_type_managers=[
            SimpleNamespace(req_to_blocks={"running": [object()]}, num_cached_block={})
        ],
        free=MagicMock(),
    )
    manager = SimpleNamespace(coordinator=coordinator)
    request = SimpleNamespace(request_id="running")

    with pytest.raises(RuntimeError, match="existing vLLM request"):
        leases_mod.patched_allocate_slots(manager, request)
    coordinator.free.assert_not_called()


def test_allocate_slots_does_not_hide_unrelated_engine_errors(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    def fail(*_args, **_kwargs):
        raise RuntimeError("native allocator invariant")

    monkeypatch.setattr(leases_mod, "orig_allocate_slots", fail)
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(req_to_blocks={}, num_cached_block={})
            ]
        )
    )
    with pytest.raises(RuntimeError, match="native allocator invariant"):
        leases_mod.patched_allocate_slots(
            manager, SimpleNamespace(request_id="request")
        )


def test_bulk_hydration_invalidates_directory_if_native_install_fails():
    from types import SimpleNamespace

    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease

    content_hash = b"h" * 32
    old_entry = {
        "tier": "hbm",
        "state": "ready",
        "slot_ids": [1],
        "generations": [7],
        "local_key": b"n" * 36,
    }
    adoption_order = []

    class Directory:
        enabled = True
        authoritative = True
        read_view_is_current_writer = True

        def __init__(self):
            self.invalidated = []
            self.released = []

        def read_view_items(self, **_kwargs):
            return [(content_hash, old_entry)]

        def lookup_and_claim(self, _keys):
            return [old_entry], "claim"

        def adopt_claim(self, _token, items):
            adoption_order.append("directory")
            assert items == [{"content_hash": content_hash, "generations": [8]}]
            old_entry["state"] = "active"
            old_entry["generations"] = [8]
            return 1

        def release_claim(self, token):
            self.released.append(token)

        def publish(self, items):
            self.invalidated.extend(items)

    class Client:
        def __init__(self):
            self.released = []

        def adopt(self, leases):
            adoption_order.append("ring")
            assert adoption_order == ["directory", "ring"]
            assert leases == [KVLease(1, 7)]
            return [KVLease(1, 8)]

        def release(self, leases):
            self.released.extend(leases)

    directory = Directory()
    client = Client()
    block = SimpleNamespace(block_id=1, ref_cnt=0, block_hash=None)
    pool = SimpleNamespace(
        _gms_hydrate_hbm=True,
        _gms_kv_directory=directory,
        _gms_kv_lease_client=client,
        _gms_kv_leases_by_block={},
        cached_block_hash_to_block=SimpleNamespace(
            get_one_block=lambda _key: None,
        ),
        blocks=[SimpleNamespace(), block],
        hash_block_size=4,
        _insert_block_hash=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("native install failed")
        ),
        _maybe_evict_cached_block=lambda _block: None,
    )

    assert leases_mod._hydrate_hbm_directory(pool, set()) == 0
    assert adoption_order == ["directory", "ring"]
    assert client.released == [KVLease(1, 8)]
    assert directory.invalidated == [
        {
            "content_hash": content_hash,
            "engine_id": leases_mod._directory_pool_id(),
            "slot_ids": [1],
            "generations": [8],
            "tier": "hbm",
            "sealed": False,
        }
    ]
    assert directory.released == []


def test_read_pinned_hbm_adoption_remains_retryable(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod
    from gpu_memory_service.integrations.common.kv_lease_client import KVLease
    from vllm.v1.core import kv_cache_utils

    native_key = b"n" * 36
    content_hash = leases_mod._directory_key(native_key)
    old_entry = {
        "tier": "hbm",
        "state": "ready",
        "slot_ids": [1],
        "generations": [7],
        "engine_id": "primary",
        "local_key": native_key,
    }

    class Directory:
        enabled = True
        authoritative = True
        read_view_is_current_writer = True

        def __init__(self):
            self.published = []

        def lookup_and_claim(self, keys):
            assert keys == [content_hash]
            return [dict(old_entry)], "claim"

        def adopt_claim(self, token, items):
            assert token == "claim"
            assert items == [{"content_hash": content_hash, "generations": [8]}]
            return 1

        def publish(self, items):
            self.published.extend(items)
            return len(items)

        def release_claim(self, _token):
            raise AssertionError("adopt_claim consumed the directory token")

    class Client:
        def __init__(self):
            self.unpinned = []

        def adopt(self, leases):
            assert leases == [KVLease(1, 7)]
            return []

        def pin_read(self, leases):
            assert leases == [KVLease(1, 7)]
            return tuple(leases)

        def unpin_read(self, claim):
            self.unpinned.append(claim)

    directory = Directory()
    client = Client()
    pool = SimpleNamespace(
        _gms_hydrate_hbm=True,
        _gms_kv_directory=directory,
        _gms_kv_lease_client=client,
        _gms_kv_leases_by_block={},
        _gms_kv_directory_slot_by_hash={},
    )
    monkeypatch.setattr(leases_mod, "_hydrate_hbm_directory", lambda *_args: 0)
    monkeypatch.setattr(
        kv_cache_utils,
        "make_block_hash_with_group_id",
        lambda *_args: native_key,
    )

    assert leases_mod._get_cached_block(pool, lambda *_args: None, b"hash", [0]) is None
    assert client.unpinned == [(KVLease(1, 7),)]
    assert directory.published == [
        {
            "content_hash": content_hash,
            "engine_id": "primary",
            "slot_ids": [1],
            "generations": [7],
            "tier": "hbm",
            "sealed": True,
            "active": False,
            "local_key": native_key,
        }
    ]
    assert pool._gms_hydrate_hbm is True


@pytest.mark.parametrize(
    ("engine_id", "expected"),
    [("0", False), ("1", True), ("shadow-a", True)],
)
def test_failover_directory_role_is_derived_in_engine_core(
    monkeypatch, engine_id, expected
):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", engine_id)
    # A stale deployment knob must not override the authoritative replica ID.
    monkeypatch.setenv("GMS_KV_DIRECTORY_STANDBY", "0" if expected else "1")

    assert leases_mod._failover_directory_standby() is expected


def test_non_failover_directory_role_remains_deployment_configurable(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)
    monkeypatch.delenv("DYN_VLLM_GMS_SHADOW_MODE", raising=False)

    assert leases_mod._failover_directory_standby() is None
