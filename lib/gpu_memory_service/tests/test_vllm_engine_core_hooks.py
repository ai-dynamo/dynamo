# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    original_block_pool_binding = kv_cache_coordinator.BlockPool
    original_block_pool_methods = dict(BlockPool.__dict__)
    original_allocate_slots = KVCacheManager.allocate_slots
    original_directory = leases_mod.ContentDirectory
    original_factory = leases_mod._factory
    original_patched = leases_mod._patched

    class LeaseState:
        free = set(range(1, 8))
        generations = defaultdict(int)
        held = {}
        seal_batches = []

    state = LeaseState()
    owners = iter(("primary", "shadow"))

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
        authoritative = True
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
                self.entries.get(value)
                if self.entries.get(value, {}).get("state") == "ready"
                else None
                for value in hashes
            ], "claim"

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

        def ensure_hbm_capacity(self, required):
            self.ensure_calls.append(required)
            victims = []
            freed = 0
            for content_hash, entry in list(self.entries.items()):
                if entry.get("tier") != "hbm" or entry.get("state") != "ready":
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

        # Directory publication failure is fail-closed for recovery but must
        # not kill vLLM's scheduler free path or leak an invisible seal.
        fallback = shadow.get_new_blocks(1)
        shadow.cache_full_blocks(
            SimpleNamespace(block_hashes=[b"z" * 32]), fallback, 0, 1, 4, 0
        )
        directory.fail_publish = True
        shadow.free_blocks(fallback)
        assert fallback[0].block_id in state.free
        assert fallback[0].block_hash is None
        assert shadow.free_block_queue.get_all_free_blocks()[0] is fallback[0]
    finally:
        kv_cache_coordinator.BlockPool = original_block_pool_binding
        KVCacheManager.allocate_slots = original_allocate_slots
        leases_mod.ContentDirectory = original_directory
        leases_mod._factory = original_factory
        leases_mod._patched = original_patched
        leases_mod._gms_block_pool_class = None


def test_allocate_slots_translates_atomic_lease_race_to_backpressure(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    def contend(*_args, **_kwargs):
        raise leases_mod.GMSKVLeaseUnavailable("lease claimed after free-count check")

    monkeypatch.setattr(leases_mod, "orig_allocate_slots", contend)
    assert leases_mod.patched_allocate_slots(object(), object()) is None


def test_allocate_slots_does_not_hide_unrelated_engine_errors(monkeypatch):
    import gpu_memory_service.integrations.vllm.install_kv_leases as leases_mod

    def fail(*_args, **_kwargs):
        raise RuntimeError("native allocator invariant")

    monkeypatch.setattr(leases_mod, "orig_allocate_slots", fail)
    with pytest.raises(RuntimeError, match="native allocator invariant"):
        leases_mod.patched_allocate_slots(object(), object())


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
