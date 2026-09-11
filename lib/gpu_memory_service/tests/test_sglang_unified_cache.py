# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
from array import array
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
        self.freed = []

    def free_segment(self, indices, *, start_pos):
        self.freed.extend(int(value) for value in indices[start_pos:].tolist())

    def free_full_segment(self, indices, *, start_pos):
        self.free_segment(indices, start_pos=start_pos)


class _Directory:
    enabled = True
    authoritative = True
    mode = "authoritative"

    def __init__(self, entries=()):
        self.entries = list(entries)
        self.published = []
        self.adopted = []
        self.released = []

    def start_async_read(self):
        return True

    def publish_deferred(self, items):
        self.published.extend(items)
        return len(items)

    def publish(self, items):
        self.published.extend(items)
        return len(items)

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


def test_rejects_streaming_sessions_before_cache_construction():
    allocator = SimpleNamespace(_gms_kv_leases_by_page={})
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
            mtp_draft_device_pools=None,
        ),
        server_args=SimpleNamespace(enable_streaming_session=True),
    )

    with pytest.raises(ValueError, match="streaming sessions"):
        install_gms_unified_cache._validate(ctx)


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


def test_failed_native_install_invalidates_then_rolls_back(monkeypatch):
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
        cache, "insert", lambda _params: SimpleNamespace(last_device_node=None)
    )

    result = cache.match_prefix(MatchPrefixParams(key=key))

    assert result.device_indices.numel() == 0
    assert directory.published[-1]["sealed"] is False
    assert rolled_back == leases


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
