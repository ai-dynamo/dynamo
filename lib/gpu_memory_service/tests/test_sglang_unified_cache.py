# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
from array import array
from hashlib import sha256
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
]

from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.sglang import gms_unified_cache as adapter
from gpu_memory_service.integrations.sglang import (
    install_gms_unified_cache,
    install_kv_leases,
    install_vmm_ipc_kv,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType


class _Allocator:
    device = torch.device("cpu")
    page_size = 2

    def __init__(self):
        self._gms_kv_leases_by_page = {}
        self._gms_retained_pages = set()
        self.freed = []

    def free_segment(self, indices, *, start_pos):
        self.freed.extend(int(value) for value in indices[start_pos:].tolist())

    def free_full_segment(self, indices, *, start_pos):
        self.free_segment(indices, start_pos=start_pos)


class _Directory:
    enabled = True
    authoritative = True
    mode = "authoritative"

    def __init__(self, entries=(), *, publish_result=None, publish_error=None):
        self.entries = list(entries)
        self.live = {}
        self.published = []
        self.publish_result = publish_result
        self.publish_error = publish_error
        self.adopted = []
        self.released = []

    def start_async_read(self):
        return True

    def publish_deferred(self, items):
        self.published.extend(items)
        return len(items)

    def publish(self, items):
        if self.publish_error is not None:
            raise self.publish_error
        self.published.extend(items)
        for item in items:
            content_hash = item["content_hash"]
            if item.get("sealed", True):
                self.live[content_hash] = {
                    "state": "active" if item.get("active", False) else "ready",
                    "tier": item.get("tier"),
                    "engine_id": item["engine_id"],
                    "slot_ids": list(item["slot_ids"]),
                    "generations": list(item["generations"]),
                }
                continue
            current = self.live.get(content_hash)
            if current is not None and current["generations"] == list(
                item["generations"]
            ):
                self.live.pop(content_hash)
        return len(items) if self.publish_result is None else self.publish_result

    def lookup_authoritative(self, hashes):
        return [self.live.get(content_hash) for content_hash in hashes]

    def lookup_and_claim(self, _hashes):
        return list(self.entries), "claim"

    def adopt_claim(self, token, items):
        self.adopted.append((token, items))
        return len(items)

    def release_claim(self, token):
        self.released.append(token)


def _cache(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "off")
    allocator = _Allocator()
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=allocator,
        page_size=allocator.page_size,
        tree_components=(ComponentType.FULL,),
    )
    cache = adapter.make_gms_unified_cache_class()(params)
    return cache, allocator


def _key(*tokens):
    return RadixKey(array("q", tokens), None)


def _test_hashes(key, prior_hash=None, *, page_size):
    prior = bytes.fromhex(prior_hash) if prior_hash else b""
    result = []
    for offset in range(0, len(key), page_size):
        prior = sha256(prior + bytes(key[offset : offset + page_size])).digest()
        result.append(prior.hex())
    return result


def test_directory_role_is_lock_authoritative_not_engine_order(monkeypatch):
    monkeypatch.delenv("GMS_KV_DIRECTORY_STANDBY", raising=False)
    monkeypatch.delenv("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", raising=False)
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("ENGINE_ID", "0")

    assert adapter._standby() is True

    monkeypatch.setenv("ENGINE_ID", "99")
    assert adapter._standby() is True

    monkeypatch.setenv("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", "1")
    assert adapter._standby() is False


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("0", False)])
def test_explicit_directory_role_overrides_lock(monkeypatch, value, expected):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_STANDBY", value)

    assert adapter._standby() is expected


def test_non_failover_directory_uses_client_default_role(monkeypatch):
    monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)

    assert adapter._standby() is None


def test_directory_hashes_preserve_native_identity_and_request_scopes():
    plain = _key(1, 2, 3, 4)
    salted = RadixKey(array("q", plain), None, cache_salt="tenant-a")
    scoped = RadixKey(array("q", plain), "adapter-a")

    plain_hashes = adapter._directory_hashes(plain, 2, _test_hashes)
    native_hashes = _test_hashes(plain, page_size=2)

    assert plain_hashes == [bytes.fromhex(value) for value in native_hashes]
    assert adapter._directory_hashes(salted, 2, _test_hashes) != plain_hashes
    assert adapter._directory_hashes(scoped, 2, _test_hashes) != plain_hashes


def test_directory_hashes_support_legacy_key_without_cache_salt():
    class LegacyRadixKey(list):
        extra_key = None

    key = LegacyRadixKey((1, 2, 3, 4))

    assert adapter._directory_hashes(key, 2, _test_hashes) == [
        bytes.fromhex(value) for value in _test_hashes(key, page_size=2)
    ]


def test_uses_native_unified_tree_and_hashes(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    key = _key(1, 2, 3, 4)
    inserted = cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))

    result = cache.match_prefix(MatchPrefixParams(key=key))

    assert type(cache).__mro__[1].__name__ == "UnifiedRadixCache"
    assert result.device_indices.tolist() == [6, 7, 2, 3]
    assert len(cache.tree_core.get_hash_values(inserted.last_device_node)) == 2


def test_publication_preserves_native_physical_page_order(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    directory = _Directory()
    cache._gms_directory = directory
    leases = {1: KVLease(1, 12), 3: KVLease(3, 34)}
    allocator._gms_kv_leases_by_page = leases
    monkeypatch.setattr(
        adapter, "retain_hbm_indices", lambda *_args: list(leases.values())
    )

    cache._publish_finished_prefix(key)

    assert [item["slot_ids"] for item in directory.published] == [[3], [1]]
    assert [item["generations"] for item in directory.published] == [[34], [12]]
    assert all(item["active"] is False for item in directory.published)


def test_publication_validates_layout_before_retaining_pages(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 3, 2])))
    directory = _Directory()
    cache._gms_directory = directory
    allocator._gms_kv_leases_by_page = {
        1: KVLease(1, 12),
        3: KVLease(3, 34),
    }
    retained = []
    monkeypatch.setattr(
        adapter,
        "retain_hbm_indices",
        lambda *_args: retained.append(True),
    )

    with pytest.raises(RuntimeError, match="contiguous KV pages"):
        cache._publish_finished_prefix(key)

    assert retained == []
    assert directory.published == []
    assert allocator._gms_retained_pages == set()


def test_unverifiable_publication_failure_retains_all_leases(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    cache._gms_directory = _Directory(publish_error=RuntimeError("daemon down"))
    leases = {1: KVLease(1, 12), 3: KVLease(3, 34)}
    allocator._gms_kv_leases_by_page = leases
    allocator._gms_retained_pages.add(3)

    def retain(_allocator, _indices):
        allocator._gms_retained_pages.update(leases)
        return list(leases.values())

    monkeypatch.setattr(adapter, "retain_hbm_indices", retain)

    with pytest.raises(RuntimeError, match="retaining leases and failing closed"):
        cache._publish_finished_prefix(key)

    assert allocator._gms_retained_pages == {1, 3}


def test_retention_failure_rolls_back_partial_flags_before_publication(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    directory = _Directory()
    cache._gms_directory = directory
    leases = {1: KVLease(1, 12), 3: KVLease(3, 34)}
    allocator._gms_kv_leases_by_page = leases

    def fail_retain(_allocator, _indices):
        allocator._gms_retained_pages.update(leases)
        raise RuntimeError("seal failed")

    monkeypatch.setattr(adapter, "retain_hbm_indices", fail_retain)

    with pytest.raises(RuntimeError, match="seal failed"):
        cache._publish_finished_prefix(key)

    assert allocator._gms_retained_pages == set()
    assert directory.published == []


def test_incomplete_publication_rolls_back_retention(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    cache._gms_directory = _Directory(publish_result=1)
    leases = {1: KVLease(1, 12), 3: KVLease(3, 34)}
    allocator._gms_kv_leases_by_page = leases

    def retain(_allocator, _indices):
        allocator._gms_retained_pages.update(leases)
        return list(leases.values())

    monkeypatch.setattr(adapter, "retain_hbm_indices", retain)

    with pytest.raises(RuntimeError, match="incomplete.*publication"):
        cache._publish_finished_prefix(key)

    assert allocator._gms_retained_pages == set()

    assert all(
        item.get("sealed", True) is False
        for item in cache._gms_directory.published[-2:]
    )
    assert cache._gms_directory.live == {}


def test_native_miss_adopts_directory_pages_and_retries_match(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    key = _key(10, 11, 12, 13)

    entries = [
        {"state": "ready", "tier": "hbm", "slot_ids": [4], "generations": [8]},
        {"state": "ready", "tier": "hbm", "slot_ids": [2], "generations": [9]},
    ]
    directory = _Directory(entries)
    cache._gms_directory = directory
    leases = [KVLease(4, 9), KVLease(2, 10)]

    def adopt_after_directory_stage(*_args):
        assert directory.adopted
        return torch.tensor([8, 9, 4, 5]), leases

    monkeypatch.setattr(adapter, "adopt_hbm_pages", adopt_after_directory_stage)

    result = cache.match_prefix(MatchPrefixParams(key=key))

    assert result.device_indices.tolist() == [8, 9, 4, 5]
    assert len(directory.adopted) == 1
    assert [item["generations"] for item in directory.adopted[0][1]] == [[9], [10]]
    assert directory.released == []


def test_directory_stage_failure_does_not_advance_the_lease_ring(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    key = _key(20, 21)
    directory = _Directory(
        [{"state": "ready", "tier": "hbm", "slot_ids": [3], "generations": [7]}]
    )
    cache._gms_directory = directory
    directory.adopt_claim = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("stage failed")
    )
    ring_calls = []
    monkeypatch.setattr(
        adapter,
        "adopt_hbm_pages",
        lambda *_args: ring_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="stage failed"):
        cache.match_prefix(MatchPrefixParams(key=key))

    assert ring_calls == []
    assert directory.released == ["claim"]


def test_preinsert_failure_invalidates_directory_then_rolls_back(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    key = _key(20, 21)
    directory = _Directory(
        [{"state": "ready", "tier": "hbm", "slot_ids": [3], "generations": [7]}]
    )
    cache._gms_directory = directory
    leases = [KVLease(3, 8)]
    monkeypatch.setattr(
        adapter,
        "adopt_hbm_pages",
        lambda *_args: (torch.tensor([6, 7]), leases),
    )
    rolled_back = []
    monkeypatch.setattr(
        adapter,
        "rollback_adopted_hbm_pages",
        lambda _allocator, value: rolled_back.extend(value),
    )
    monkeypatch.setattr(
        torch,
        "cat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("pre-insert failure")
        ),
    )

    result = cache.match_prefix(MatchPrefixParams(key=key))

    assert result.device_indices.numel() == 0
    assert directory.published[-1] == {
        "content_hash": cache._hashes_for_key(key, cache.page_size)[0],
        "engine_id": cache._gms_engine_id,
        "slot_ids": [3],
        "generations": [7],
        "tier": "hbm",
        "sealed": False,
    }
    assert rolled_back == leases


def test_postmutation_insert_failure_propagates_without_releasing_pages(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    key = _key(20, 21)
    directory = _Directory(
        [{"state": "ready", "tier": "hbm", "slot_ids": [3], "generations": [7]}]
    )
    cache._gms_directory = directory
    leases = [KVLease(3, 8)]
    monkeypatch.setattr(
        adapter,
        "adopt_hbm_pages",
        lambda *_args: (torch.tensor([6, 7]), leases),
    )
    rolled_back = []
    monkeypatch.setattr(
        adapter,
        "rollback_adopted_hbm_pages",
        lambda _allocator, value: rolled_back.extend(value),
    )
    mutated = []

    def partially_mutate_then_fail(params):
        mutated.append(params)
        raise RuntimeError("failure after native mutation")

    monkeypatch.setattr(cache, "insert", partially_mutate_then_fail)

    with pytest.raises(RuntimeError, match="failure after native mutation"):
        cache.match_prefix(MatchPrefixParams(key=key))

    assert len(mutated) == 1
    assert rolled_back == []
    assert directory.published == []
    assert directory.released == []


def test_allocator_install_rebinds_constructors_without_mutating_native_classes(
    monkeypatch,
):
    from sglang.srt.mem_cache import allocator, kv_cache_configurator

    native_token = allocator.TokenToKVPoolAllocator
    native_paged = allocator.PagedTokenToKVPoolAllocator
    native_token_alloc = native_token.alloc
    native_paged_alloc = native_paged.alloc
    monkeypatch.setattr(install_kv_leases, "_patched", False)
    monkeypatch.setattr(install_kv_leases, "_factory", None)

    assert install_kv_leases.install(factory=lambda *_args: SimpleNamespace())

    assert install_kv_leases.lease_hooks_installed()
    assert issubclass(kv_cache_configurator.TokenToKVPoolAllocator, native_token)
    assert issubclass(kv_cache_configurator.PagedTokenToKVPoolAllocator, native_paged)
    assert native_token.alloc is native_token_alloc
    assert native_paged.alloc is native_paged_alloc


def _mha_values(**overrides):
    values = {
        "size": 4096,
        "page_size": 64,
        "dtype": torch.float16,
        "head_num": 8,
        "head_dim": 128,
        "v_head_dim": 128,
        "layer_num": 2,
        "start_layer": 4,
        "end_layer": 6,
        "kv_cache_layout": "NHD",
    }
    values.update(overrides)
    return values


def test_semantic_physical_tags_include_manifest_layout_kind_and_layer(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "model-a@immutable-revision")

    tags = install_vmm_ipc_kv._semantic_tag_plan("MHATokenToKVPool", _mha_values())
    same = install_vmm_ipc_kv._semantic_tag_plan("MHATokenToKVPool", _mha_values())
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "model-b@immutable-revision")
    changed_model = install_vmm_ipc_kv._semantic_tag_plan(
        "MHATokenToKVPool", _mha_values()
    )
    changed_shape = install_vmm_ipc_kv._semantic_tag_plan(
        "MHATokenToKVPool", _mha_values(head_dim=64)
    )

    assert tags == same
    assert [tag.rsplit(":", 2)[-2:] for tag in tags] == [
        ["k", "layer4"],
        ["k", "layer5"],
        ["v", "layer4"],
        ["v", "layer5"],
    ]
    assert tags != changed_model
    assert changed_model != changed_shape


def test_semantic_mla_plan_has_one_tensor_per_layer(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "deepseek@immutable-revision")

    tags = install_vmm_ipc_kv._semantic_tag_plan(
        "MLATokenToKVPool",
        _mha_values(layer_num=3, start_layer=1, end_layer=4, kv_lora_rank=512),
    )

    assert [tag.rsplit(":", 2)[-2:] for tag in tags] == [
        ["kv", "layer1"],
        ["kv", "layer2"],
        ["kv", "layer3"],
    ]


def test_shared_semantic_plan_requires_deployment_manifest(monkeypatch):
    from gpu_memory_service.integrations.sglang import kv_identity

    monkeypatch.delenv("GMS_KV_DIRECTORY_MANIFEST", raising=False)
    monkeypatch.setattr(kv_identity, "shared_kv_enabled", lambda: True)

    with pytest.raises(RuntimeError, match="GMS_KV_DIRECTORY_MANIFEST"):
        install_vmm_ipc_kv._semantic_tag_plan("MHATokenToKVPool", _mha_values())


def test_persistent_plan_rejects_partial_reattach():
    manager = SimpleNamespace(
        list_persistent=lambda **_kwargs: [SimpleNamespace(tag="kv:new:a")]
    )

    with pytest.raises(RuntimeError, match="only partially present"):
        install_vmm_ipc_kv._prepare_tag_plan(
            manager, "engine", "kv_pool", ["kv:new:a", "kv:new:b"]
        )


def test_persistent_plan_rejects_same_manifest_with_different_layout(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "model@immutable-revision")
    old_plan = install_vmm_ipc_kv._semantic_tag_plan(
        "MHATokenToKVPool", _mha_values(head_dim=128)
    )
    new_plan = install_vmm_ipc_kv._semantic_tag_plan(
        "MHATokenToKVPool", _mha_values(head_dim=64)
    )

    allocations = [
        SimpleNamespace(tag=old_plan[0], claimed=False),
        SimpleNamespace(tag="weights:v1:model", claimed=False),
    ]
    manager = SimpleNamespace(
        list_persistent=lambda **_kwargs: allocations,
    )

    with pytest.raises(RuntimeError, match="Reset the persistent KV pool"):
        install_vmm_ipc_kv._prepare_tag_plan(
            manager,
            "engine",
            "kv_pool",
            new_plan,
        )


def test_persistent_init_marks_pool_only_after_complete_plan(monkeypatch):
    from gpu_memory_service.client.torch import allocator as torch_allocator

    events = []
    instance = SimpleNamespace()
    manager = object()

    def original_init(target):
        assert not hasattr(target, "_gms_persistent_kv")
        events.append("native-init")
        return "initialized"

    monkeypatch.setattr(
        torch_allocator,
        "get_or_create_persistent_allocator",
        lambda *_args, **_kwargs: manager,
    )
    monkeypatch.setattr(
        torch_allocator,
        "set_persistent_allocator_tag_plan",
        lambda *_args: events.append("plan-set"),
    )
    monkeypatch.setattr(
        torch_allocator,
        "validate_persistent_allocator_tag_plan_consumed",
        lambda *_args: events.append("plan-validated"),
    )
    monkeypatch.setattr(
        torch_allocator,
        "clear_persistent_allocator_tag_plan",
        lambda *_args: events.append("plan-cleared"),
    )
    monkeypatch.setattr(install_vmm_ipc_kv, "_prepare_tag_plan", lambda *_args: False)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "_semantic_tag_plan", lambda *_args: ["semantic-tag"]
    )
    monkeypatch.setattr(install_vmm_ipc_kv, "_resolve_kv_pool_device", lambda *_: 0)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "_resolve_socket", lambda *_: "/unused-fake-gms.sock"
    )

    result = install_vmm_ipc_kv._persistent_init(
        original_init, "MHATokenToKVPool", instance, (), {}
    )

    assert result == "initialized"
    assert instance._gms_persistent_kv is True
    assert events == ["plan-set", "native-init", "plan-validated", "plan-cleared"]


def test_persistent_init_does_not_mark_under_consumed_plan(monkeypatch):
    from gpu_memory_service.client.torch import allocator as torch_allocator

    instance = SimpleNamespace()
    manager = object()
    monkeypatch.setattr(
        torch_allocator,
        "get_or_create_persistent_allocator",
        lambda *_args, **_kwargs: manager,
    )
    monkeypatch.setattr(
        torch_allocator, "set_persistent_allocator_tag_plan", lambda *_: None
    )
    monkeypatch.setattr(
        torch_allocator, "clear_persistent_allocator_tag_plan", lambda *_: None
    )
    monkeypatch.setattr(
        torch_allocator,
        "validate_persistent_allocator_tag_plan_consumed",
        lambda *_: (_ for _ in ()).throw(RuntimeError("under-consumed")),
    )
    monkeypatch.setattr(install_vmm_ipc_kv, "_prepare_tag_plan", lambda *_args: False)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "_semantic_tag_plan", lambda *_args: ["tag"]
    )
    monkeypatch.setattr(install_vmm_ipc_kv, "_resolve_kv_pool_device", lambda *_: 0)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "_resolve_socket", lambda *_: "/unused-fake-gms.sock"
    )
    released = []
    monkeypatch.setattr(
        install_vmm_ipc_kv,
        "_release_new_plan",
        lambda *args: released.append(args),
    )

    with pytest.raises(RuntimeError, match="under-consumed"):
        install_vmm_ipc_kv._persistent_init(
            lambda _target: None, "MHATokenToKVPool", instance, (), {}
        )

    assert not hasattr(instance, "_gms_persistent_kv")
    assert released == [(manager, install_vmm_ipc_kv._engine_id(0), ["tag"])]


def test_unified_backend_rejects_nonpersistent_physical_pool():
    allocator = SimpleNamespace(
        _gms_kv_leases_by_page={},
        get_kvcache=lambda: SimpleNamespace(),
    )
    ctx = SimpleNamespace(
        disable_radix_cache=False,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        is_dsa=False,
        enable_hierarchical_cache=False,
        params=SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            enable_session_radix_cache=False,
            is_eagle=False,
            mtp_draft_device_pools=False,
        ),
    )

    with pytest.raises(ValueError, match="outside GMS persistent memory"):
        install_gms_unified_cache._validate(ctx)


def test_ring_only_mode_leaves_native_radix_cache(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "off")
    monkeypatch.setenv("GMS_SGLANG_ENABLE_KV_RING", "1")

    assert install_gms_unified_cache._enabled() is False
    assert install_gms_unified_cache.configure(SimpleNamespace()) is False


def test_shadow_directory_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "shadow")
    monkeypatch.setenv("GMS_SGLANG_ENABLE_KV_RING", "1")

    with pytest.raises(ValueError, match="requires.*authoritative"):
        install_gms_unified_cache.configure(SimpleNamespace())


def test_configure_rejects_mooncake_custom_pool(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "NVLINK")

    with pytest.raises(ValueError, match="SGLANG_MOONCAKE_CUSTOM_MEM_POOL"):
        install_gms_unified_cache.configure(SimpleNamespace())


def test_configure_supports_legacy_server_args_override(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.delenv("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", raising=False)
    monkeypatch.setattr(install_gms_unified_cache, "install", lambda: True)
    from sglang.srt.arg_groups import overrides

    monkeypatch.delattr(overrides, "declare_late_resolution")
    monkeypatch.delattr(overrides, "resolving_view")
    calls = []
    args = SimpleNamespace(
        radix_cache_backend=None,
        disable_radix_cache=False,
        enable_page_major_kv_layout=False,
        enable_unified_memory=False,
        enable_hierarchical_cache=False,
        enable_session_radix_cache=False,
        enable_streaming_session=False,
        speculative_algorithm=None,
        override=lambda source, **fields: calls.append((source, fields)),
    )

    assert install_gms_unified_cache.configure(args) is True
    assert calls == [("dynamo.gms", {"radix_cache_backend": "gms"})]


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [
        ("disable_radix_cache", True, "disabled radix cache"),
        ("enable_page_major_kv_layout", True, "page-major KV layout"),
        ("enable_unified_memory", True, "unified memory"),
        ("enable_hierarchical_cache", True, "hierarchical cache"),
        ("enable_session_radix_cache", True, "session radix cache"),
        ("enable_streaming_session", True, "streaming sessions"),
        ("speculative_algorithm", "EAGLE", "speculative decoding"),
        ("tp_size", 2, "tensor parallelism"),
        ("dp_size", 2, "data parallelism"),
        ("pp_size", 2, "pipeline parallelism"),
        ("dcp_size", 2, "decode context parallelism"),
        ("attn_cp_size", 2, "attention context parallelism"),
        ("attention_cp_size", 2, "attention context parallelism"),
        ("enable_prefill_cp", True, "prefill context parallelism"),
        (
            "enable_prefill_context_parallel",
            True,
            "prefill context parallelism",
        ),
        ("enable_dp_attention", True, "data-parallel attention"),
        ("disaggregation_mode", "prefill", "prefill/decode disaggregation"),
    ],
)
def test_configure_rejects_unsupported_server_mode(
    monkeypatch, attribute, value, reason
):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.delenv("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", raising=False)
    install_calls = []
    monkeypatch.setattr(
        install_gms_unified_cache, "install", lambda: install_calls.append(True)
    )

    from sglang.srt.arg_groups import overrides

    monkeypatch.setattr(overrides, "resolving_view", lambda value: value)
    args = SimpleNamespace(
        radix_cache_backend=None,
        disable_radix_cache=False,
        enable_page_major_kv_layout=False,
        enable_unified_memory=False,
        enable_hierarchical_cache=False,
        enable_session_radix_cache=False,
        enable_streaming_session=False,
        speculative_algorithm=None,
    )
    setattr(args, attribute, value)

    with pytest.raises(ValueError, match=reason):
        install_gms_unified_cache.configure(args)

    assert install_calls == []
