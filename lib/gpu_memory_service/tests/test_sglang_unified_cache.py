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
        self.read_view_is_current_writer = False
        self.read_view = []
        self.snapshot = None

    def start_async_read(self):
        return True

    def freeze_current_writer_view(self):
        return bool(self.read_view_is_current_writer)

    def read_view_items(self, *, tier=None, state="ready", limit=None):
        items = [
            (content_hash, entry)
            for content_hash, entry in self.read_view
            if (tier is None or entry.get("tier") == tier)
            and (not state or entry.get("state") == state)
        ]
        return items if limit is None else items[:limit]

    def snapshot_authoritative(self):
        entries = self.read_view if self.snapshot is None else self.snapshot
        return dict(entries)

    def publish_deferred(self, items):
        # The fake commits immediately so enqueue failures and compensation
        # remain observable without a background thread.
        return self.publish(items)

    def flush_deferred(self, timeout=None):
        return True

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

    def may_have_hbm_candidate(self, _hashes):
        entries = self.entries or self.live.values()
        return any(
            entry is not None
            and entry.get("tier") == "hbm"
            and entry.get("state") in ("ready", "active")
            for entry in entries
        )

    def lookup_and_claim(self, _hashes):
        return list(self.entries), "claim"

    def adopt_claim(self, token, items):
        self.adopted.append((token, items))
        return len(items)

    def release_claim(self, token):
        self.released.append(token)


def _cache(monkeypatch, tree_core_backend=None):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "off")
    allocator = _Allocator()
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=allocator,
        page_size=allocator.page_size,
        tree_components=(ComponentType.FULL,),
        tree_core_backend=tree_core_backend,
    )
    cache = adapter.make_gms_unified_cache_class()(params)
    return cache, allocator


def _matched_indices(cache, result):
    """Read a match's KV indices off its node path, as SGLang does."""
    if result.device_prefix_len == 0:
        return []
    path = cache.path_device_indices(result.last_device_node)
    return path[: result.device_prefix_len].tolist()


def _key(*tokens):
    return RadixKey(array("q", tokens), None)


def _test_hashes(key, prior_hash=None, *, page_size):
    prior = bytes.fromhex(prior_hash) if prior_hash else b""
    result = []
    for offset in range(0, len(key), page_size):
        prior = sha256(prior + bytes(key[offset : offset + page_size])).digest()
        result.append(prior.hex())
    return result


def test_standby_headroom_covers_configured_warmup_concurrency(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("DYN_SGLANG_GMS_FAILOVER_PROMOTION_WARMUP_CONCURRENCY", "4")

    assert adapter._standby_headroom_pages() == 4


def test_standby_headroom_has_no_cost_outside_failover(monkeypatch):
    monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)
    monkeypatch.setenv("DYN_SGLANG_GMS_FAILOVER_PROMOTION_WARMUP_CONCURRENCY", "4")

    assert adapter._standby_headroom_pages() == 0


def test_directory_role_is_lock_authoritative_not_engine_order(monkeypatch):
    monkeypatch.delenv("GMS_KV_DIRECTORY_STANDBY", raising=False)
    monkeypatch.delenv("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", raising=False)
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("ENGINE_ID", "0")

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


@pytest.mark.parametrize("is_bigram", [False, True])
@pytest.mark.parametrize("scope", [None, "tenant-a"])
def test_native_directory_hashes_are_stable_for_page_aligned_prefixes(is_bigram, scope):
    from sglang.srt.mem_cache.utils import get_hash_str

    tokens = array("q", range(1, 10 if is_bigram else 9))
    key = RadixKey(
        tokens,
        scope,
        is_bigram=is_bigram,
        cache_salt="salt-a",
    )
    full = adapter._directory_hashes(key, 2, get_hash_str)

    for prefix_len in (2, 4, 6, 8):
        assert (
            adapter._directory_hashes(key[:prefix_len], 2, get_hash_str)
            == full[: prefix_len // 2]
        )


def test_reset_invalidates_directory_before_discarding_derived_state(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    content_hash = b"r" * 32
    lease = KVLease(3, 9)
    allocator._gms_kv_leases_by_page[3] = lease
    allocator._gms_retained_pages.add(3)
    directory = _Directory()
    directory.live[content_hash] = {
        "state": "ready",
        "tier": "hbm",
        "engine_id": cache._gms_engine_id,
        "slot_ids": [3],
        "generations": [9],
    }
    cache._gms_directory = directory
    cache._gms_steady_state = True
    cache._gms_recovery_candidates.add(content_hash)
    cache._gms_local_pages_by_hash[content_hash] = 3
    cache._gms_local_hashes_by_page[3] = {content_hash}
    cache._gms_retained_order[3] = None

    cache.reset()

    assert content_hash not in directory.live
    assert cache._gms_steady_state is False
    assert cache._gms_recovery_candidates == set()
    assert cache._gms_local_pages_by_hash == {}
    assert cache._gms_local_hashes_by_page == {}
    assert cache._gms_retained_order == {}


def test_uses_native_unified_tree_without_enabling_storage_hashing(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    key = _key(1, 2, 3, 4)
    inserted = cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    cache._gms_tp.agree = lambda *_args: pytest.fail(
        "native prefix hits must not enter TP consensus"
    )

    result = cache.match_prefix(MatchPrefixParams(key=key))

    assert type(cache).__mro__[1].__name__ == "UnifiedRadixCache"
    assert _matched_indices(cache, result) == [6, 7, 2, 3]
    assert cache.tree_core.get_hash_values(inserted.last_device_node) == []


def test_native_hit_seeds_request_pages_when_directory_map_is_incomplete(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    allocator._gms_kv_leases_by_page = {
        3: KVLease(3, 13),
        1: KVLease(1, 11),
    }
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    req = SimpleNamespace(session=None)

    cache.match_prefix(MatchPrefixParams(key=key, req=req))

    assert req._gms_kv_page_ids == [3, 1]


def test_fresh_unfinished_insert_uses_request_cpu_pages_without_gpu_sync(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    allocator._gms_kv_leases_by_page = {
        3: KVLease(3, 13),
        1: KVLease(1, 11),
    }
    key = _key(1, 2, 3, 4)
    req = SimpleNamespace(session=None, _gms_kv_page_ids=[3, 1], finished=lambda: False)

    def native_checkpoint(target, native_req, **_kwargs):
        # SGLang inserts, then re-matches through its own tree walk to repoint
        # the request's row onto the tree's pages.
        target.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
        target._match_tree(MatchPrefixParams(key=key, req=native_req))

    monkeypatch.setattr(
        type(cache).__mro__[1], "checkpoint_into_tree", native_checkpoint
    )
    monkeypatch.setattr(
        torch.Tensor,
        "cpu",
        lambda *_args, **_kwargs: pytest.fail(
            "fresh native insertion must not synchronize the GPU index tensor"
        ),
    )

    cache.checkpoint_into_tree(req, up_to=4)

    assert req._gms_kv_page_ids == [3, 1]
    assert cache._gms_fresh_insert is None


def test_duplicate_unfinished_insert_uses_native_pages_not_request_pages(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    allocator._gms_kv_leases_by_page = {
        page: KVLease(page, 10 + page) for page in (1, 3, 4, 5)
    }
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    req = SimpleNamespace(session=None, _gms_kv_page_ids=[4, 5], finished=lambda: False)

    def native_checkpoint(target, native_req, **_kwargs):
        target.insert(InsertParams(key=key, value=torch.tensor([8, 9, 10, 11])))
        target._match_tree(MatchPrefixParams(key=key, req=native_req))

    monkeypatch.setattr(
        type(cache).__mro__[1], "checkpoint_into_tree", native_checkpoint
    )

    cache.checkpoint_into_tree(req, up_to=4)

    assert req._gms_kv_page_ids == [3, 1]
    assert cache._gms_fresh_insert is None


class _ReqToTokenPool:
    def __init__(self, row):
        self.req_to_token = torch.tensor([row], dtype=torch.int64)

    def write(self, index, values):
        self.req_to_token[index] = values


def _checkpoint_req(tokens, pages, *, finished):
    return SimpleNamespace(
        rid="r",
        session=None,
        finished=lambda: finished,
        finished_reason=None,
        full_untruncated_fill_ids=array("q", tokens),
        origin_input_ids=array("q", tokens),
        extra_key=None,
        cache_salt=None,
        priority=0,
        session_id=None,
        kv_rotation_base=None,
        lock=None,
        last_node=None,
        is_retracted=False,
        kv=SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=0,
            cache_inserted_len=0,
            kv_allocated_len=len(tokens),
        ),
        _gms_kv_page_ids=list(pages),
    )


def test_native_running_checkpoint_attaches_request_pages(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    allocator._gms_kv_leases_by_page = {3: KVLease(3, 13), 1: KVLease(1, 11)}
    cache.req_to_token_pool = _ReqToTokenPool([6, 7, 2, 3])
    req = _checkpoint_req([1, 2, 3, 4], [3, 1], finished=False)
    attached = []
    original_attach = cache._attach_request_pages
    monkeypatch.setattr(
        cache,
        "_attach_request_pages",
        lambda params, result: attached.append(params.req)
        or original_attach(params, result),
    )

    # SGLang's own checkpoint inserts and re-matches through its tree walk.
    cache.checkpoint(req, up_to=4)

    assert attached == [req]
    assert req._gms_kv_page_ids == [3, 1]
    assert req.kv.cache_protected_len == 4
    assert cache._gms_unfinished_req is None
    assert cache._gms_fresh_insert is None
    assert cache._gms_directory.published == []


def test_native_final_checkpoint_publishes_page_aligned_prefix(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    leases = {3: KVLease(3, 13), 1: KVLease(1, 11), 4: KVLease(4, 14)}
    allocator._gms_kv_leases_by_page = leases
    monkeypatch.setattr(
        adapter,
        "retain_hbm_pages",
        lambda _allocator, pages: [leases[page] for page in pages],
    )
    cache.req_to_token_pool = _ReqToTokenPool([6, 7, 2, 3, 8])
    req = _checkpoint_req([1, 2, 3, 4, 5], [3, 1, 4], finished=True)
    attached = []
    monkeypatch.setattr(
        cache, "_attach_request_pages", lambda params, _result: attached.append(1)
    )

    cache.checkpoint(req, up_to=5)

    # Only the sealed, page-aligned prefix is published; the partial tail
    # page stays request-owned for release_kv_cache to free.
    assert [item["slot_ids"] for item in cache._gms_directory.published] == [
        [3],
        [1],
    ]
    assert attached == []
    assert req.kv.cache_protected_len == 4


def test_native_hit_never_records_unleased_page(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    allocator._gms_kv_leases_by_page = {3: KVLease(3, 13)}
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    req = SimpleNamespace(session=None)

    cache.match_prefix(MatchPrefixParams(key=key, req=req))

    assert req._gms_kv_page_ids is None


def test_second_native_match_preserves_still_allocated_request_suffix(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    allocator._gms_kv_leases_by_page = {
        page: KVLease(page, 10 + page) for page in (1, 3, 4)
    }
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    req = SimpleNamespace(
        session=None,
        # The seventh native page may be only partly filled when the second
        # lookup runs; it still belongs to this request.
        kv=SimpleNamespace(kv_allocated_len=5),
        _gms_kv_page_ids=[3, 1, 4],
        is_retracted=False,
    )

    cache.match_prefix(MatchPrefixParams(key=key, req=req))

    assert req._gms_kv_page_ids == [3, 1, 4]

    req.kv.kv_allocated_len = 4
    cache.match_prefix(MatchPrefixParams(key=key, req=req))
    assert req._gms_kv_page_ids == [3, 1]

    req.kv.kv_allocated_len = 5
    req._gms_kv_page_ids = [3, 1, 4]
    allocator._gms_kv_leases_by_page.pop(4)
    cache.match_prefix(MatchPrefixParams(key=key, req=req))
    assert req._gms_kv_page_ids == [3, 1]


def test_finished_publication_reuses_insert_result_without_second_match(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    lease = KVLease(3, 12)
    allocator._gms_kv_leases_by_page = {3: lease}
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    monkeypatch.setattr(adapter, "retain_hbm_pages", lambda *_args: [lease])
    monkeypatch.setattr(
        type(cache).__mro__[1],
        "match_prefix",
        lambda *_args, **_kwargs: pytest.fail(
            "finished publication must reuse the native insert result"
        ),
    )
    monkeypatch.setattr(
        type(cache).__mro__[1],
        "checkpoint_into_tree",
        lambda self, *_args, **_kwargs: self.insert(
            InsertParams(key=_key(1, 2), value=torch.tensor([6, 7]))
        ),
    )

    cache.checkpoint_into_tree(SimpleNamespace(finished=lambda: True), up_to=2)

    assert len(cache._gms_directory.published) == 1


def test_finished_publication_uses_cpu_pages_without_device_collection(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    lease = KVLease(3, 12)
    allocator._gms_kv_leases_by_page = {3: lease}
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    monkeypatch.setattr(adapter, "retain_hbm_pages", lambda *_args: [lease])
    monkeypatch.setattr(
        cache.tree_core,
        "collect_full_device_indices",
        lambda *_args, **_kwargs: pytest.fail(
            "CPU page metadata must avoid device index collection"
        ),
    )
    monkeypatch.setattr(
        type(cache).__mro__[1],
        "checkpoint_into_tree",
        lambda self, *_args, **_kwargs: self.insert(
            InsertParams(key=_key(1, 2), value=torch.tensor([6, 7]))
        ),
    )
    req = SimpleNamespace(_gms_kv_page_ids=[3], finished=lambda: True)

    cache.checkpoint_into_tree(req, up_to=2)

    assert len(cache._gms_directory.published) == 1
    assert cache._gms_directory.published[0]["slot_ids"] == [3]


def test_live_prefix_publication_seals_only_newly_committed_full_pages(monkeypatch):
    monkeypatch.setattr(adapter, "_LIVE_CONFIRM_CALLS", 1)
    monkeypatch.setattr(adapter, "_LIVE_PUBLISH_DELAY_CALLS", 0)
    cache, allocator = _cache(monkeypatch)
    leases = {page: KVLease(page, 10 + page) for page in (3, 4, 5)}
    allocator._gms_kv_leases_by_page = leases
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    acknowledgements = []
    votes = []

    def vote(stage, digest, operation):
        votes.append((stage, digest))
        return operation()

    cache._gms_tp = SimpleNamespace(transact_digest=vote)
    monkeypatch.setattr(
        cache._gms_directory,
        "flush_deferred",
        lambda timeout: acknowledgements.append(timeout) or True,
    )
    monkeypatch.setattr(
        cache,
        "_hashes_for_key",
        lambda key, page_size: [
            index.to_bytes(32, "little")
            for index in range(1, len(key) // page_size + 1)
        ],
    )

    def retain(_allocator, pages):
        allocator._gms_retained_pages.update(pages)
        return [leases[page] for page in pages]

    monkeypatch.setattr(adapter, "retain_hbm_pages", retain)
    req = SimpleNamespace(
        kv=SimpleNamespace(kv_committed_len=4),
        origin_input_ids=[1, 2, 3, 4, 5, 6],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        _gms_kv_page_ids=[3, 4, 5],
        finished=lambda: False,
    )

    cache._gms_publish_live_prefixes([req])
    cache._gms_publish_live_prefixes([req])

    assert len(cache._gms_directory.published) == 2
    assert req._gms_published_kv_len == 4
    assert acknowledgements == [2.0]
    assert len(votes) == 1 and votes[0][0] == "live:ack"

    req.kv.kv_committed_len = 6
    cache._gms_publish_live_prefixes([req])

    # Published now, confirmed by the next output: no daemon wait or TP
    # barrier on the output path that publishes.
    assert len(cache._gms_directory.published) == 3
    assert cache._gms_directory.published[-1]["slot_ids"] == [5]
    assert acknowledgements == [2.0]
    assert req._gms_published_kv_len == 4

    cache._gms_publish_live_prefixes([req])
    assert acknowledgements == [2.0, 2.0]
    assert len(votes) == 2
    assert req._gms_published_kv_len == 6


def test_live_prefix_is_not_marked_published_without_daemon_ack(monkeypatch):
    monkeypatch.setattr(adapter, "_LIVE_CONFIRM_CALLS", 1)
    monkeypatch.setattr(adapter, "_LIVE_PUBLISH_DELAY_CALLS", 0)
    cache, _allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    monkeypatch.setattr(cache, "_publish_finished_prefix", lambda *_a, **_kw: None)
    monkeypatch.setattr(cache._gms_directory, "flush_deferred", lambda timeout: False)
    req = SimpleNamespace(
        kv=SimpleNamespace(kv_committed_len=2),
        origin_input_ids=[1, 2],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        _gms_kv_page_ids=[3],
        finished=lambda: False,
    )

    cache._gms_tp = SimpleNamespace(
        transact_digest=lambda _stage, _digest, operation: operation()
    )
    cache._gms_publish_live_prefixes([req])
    with pytest.raises(RuntimeError, match="acknowledging live SGLang"):
        cache._gms_publish_live_prefixes([req])

    assert not hasattr(req, "_gms_published_kv_len")


def test_live_prefix_is_not_marked_published_without_tp_vote(monkeypatch):
    monkeypatch.setattr(adapter, "_LIVE_CONFIRM_CALLS", 1)
    monkeypatch.setattr(adapter, "_LIVE_PUBLISH_DELAY_CALLS", 0)
    cache, _allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    monkeypatch.setattr(cache, "_publish_finished_prefix", lambda *_a, **_kw: None)

    def fail_peer_vote(_stage, _digest, operation):
        operation()
        raise RuntimeError("peer rank did not commit")

    cache._gms_tp = SimpleNamespace(transact_digest=fail_peer_vote)
    req = SimpleNamespace(
        kv=SimpleNamespace(kv_committed_len=2),
        origin_input_ids=[1, 2],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        _gms_kv_page_ids=[3],
        finished=lambda: False,
    )

    cache._gms_publish_live_prefixes([req])
    with pytest.raises(RuntimeError, match="peer rank"):
        cache._gms_publish_live_prefixes([req])

    assert not hasattr(req, "_gms_published_kv_len")


def test_live_publish_failure_is_reported_through_the_vote(monkeypatch):
    monkeypatch.setattr(adapter, "_LIVE_CONFIRM_CALLS", 1)
    monkeypatch.setattr(adapter, "_LIVE_PUBLISH_DELAY_CALLS", 0)
    """A rank that fails to publish fails the next vote, so peers stop too."""
    cache, _allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()

    def broken_publish(*_args, **_kwargs):
        raise RuntimeError("local publish failed")

    monkeypatch.setattr(cache, "_publish_finished_prefix", broken_publish)
    votes = []

    def vote(stage, _digest, operation):
        try:
            operation()
        except Exception:
            votes.append((stage, False))
            raise RuntimeError("SGLang GMS TP transaction failed") from None
        votes.append((stage, True))

    cache._gms_tp = SimpleNamespace(transact_digest=vote)
    req = SimpleNamespace(
        kv=SimpleNamespace(kv_committed_len=2),
        origin_input_ids=[1, 2],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        _gms_kv_page_ids=[3],
        finished=lambda: False,
    )

    cache._gms_publish_live_prefixes([req])
    with pytest.raises(RuntimeError, match="TP transaction failed"):
        cache._gms_publish_live_prefixes([req])

    assert votes == [("live:ack", False)]
    assert not hasattr(req, "_gms_published_kv_len")


def test_live_publications_are_confirmed_together(monkeypatch):
    """Several publications share one daemon wait and one TP vote."""
    monkeypatch.setattr(adapter, "_LIVE_CONFIRM_CALLS", 3)
    monkeypatch.setattr(adapter, "_LIVE_PUBLISH_DELAY_CALLS", 0)
    cache, _allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    published = []
    monkeypatch.setattr(
        cache,
        "_publish_finished_prefix",
        lambda key, **_kw: published.append(len(key)),
    )
    acknowledgements = []
    monkeypatch.setattr(
        cache._gms_directory,
        "flush_deferred",
        lambda timeout: acknowledgements.append(timeout) or True,
    )
    votes = []

    def vote(stage, digest, operation):
        votes.append(stage)
        return operation()

    cache._gms_tp = SimpleNamespace(transact_digest=vote)

    def request(rid):
        return SimpleNamespace(
            rid=rid,
            kv=SimpleNamespace(kv_committed_len=2),
            origin_input_ids=[1, 2],
            output_ids=[],
            extra_key=None,
            cache_salt=None,
            _gms_kv_page_ids=[3],
            finished=lambda: False,
        )

    first, second = request("a"), request("b")
    cache._gms_publish_live_prefixes([first])
    cache._gms_publish_live_prefixes([first, second])
    # Pending publications are not republished while awaiting confirmation.
    assert len(published) == 2
    assert votes == [] and acknowledgements == []
    assert not hasattr(first, "_gms_published_kv_len")

    cache._gms_publish_live_prefixes([first, second])
    cache._gms_publish_live_prefixes([first, second])
    assert votes == ["live:ack"] and acknowledgements == [2.0]
    assert first._gms_published_kv_len == second._gms_published_kv_len == 2
    assert len(published) == 2


def test_live_prefix_waits_and_finished_requests_skip_it(monkeypatch):
    """New pages publish after a delay; a request that finishes first never does."""
    monkeypatch.setattr(adapter, "_LIVE_PUBLISH_DELAY_CALLS", 2)
    cache, _allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    published = []
    monkeypatch.setattr(
        cache, "_publish_finished_prefix", lambda key, **_kw: published.append(key)
    )
    cache._gms_tp = SimpleNamespace(
        transact_digest=lambda _stage, _digest, operation: operation()
    )

    def request(rid):
        done = {"value": False}
        return done, SimpleNamespace(
            rid=rid,
            kv=SimpleNamespace(kv_committed_len=2),
            origin_input_ids=[1, 2],
            output_ids=[],
            extra_key=None,
            cache_salt=None,
            _gms_kv_page_ids=[3],
            finished=lambda: done["value"],
        )

    _live_done, live = request("live")
    short_done, short = request("short")
    cache._gms_publish_live_prefixes([live, short])
    cache._gms_publish_live_prefixes([live, short])
    assert published == []
    short_done["value"] = True
    cache._gms_publish_live_prefixes([live, short])
    assert len(published) == 1
    assert live._gms_live_pending_len == 2
    assert not hasattr(short, "_gms_live_pending_len")


def test_finished_publication_falls_back_when_captured_node_has_no_indices(
    monkeypatch,
):
    cache, allocator = _cache(monkeypatch)
    lease = KVLease(3, 12)
    allocator._gms_kv_leases_by_page = {3: lease}
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    monkeypatch.setattr(adapter, "retain_hbm_pages", lambda *_args: [lease])
    monkeypatch.setattr(cache, "_hashes_for_key", lambda *_args: [b"x" * 32])
    inserted = cache.insert(InsertParams(key=_key(1, 2), value=torch.tensor([6, 7])))
    collect = cache.tree_core.collect_full_device_indices
    calls = []

    def collect_once_missing(*args, **kwargs):
        # The captured node yields nothing; SGLang's authoritative lookup
        # then reads the same prefix off a freshly matched node path.
        calls.append(args)
        return None if len(calls) == 1 else collect(*args, **kwargs)

    monkeypatch.setattr(
        cache.tree_core, "collect_full_device_indices", collect_once_missing
    )

    cache._publish_finished_prefix(_key(1, 2), inserted.last_device_node)

    assert len(calls) == 2
    assert len(cache._gms_directory.published) == 1
    assert cache._gms_directory.published[0]["slot_ids"] == [3]


def test_transitional_publication_validates_cpu_pages_against_native_layout(
    monkeypatch,
):
    cache, allocator = _cache(monkeypatch)
    allocator._gms_kv_leases_by_page = {3: KVLease(3, 12)}
    monkeypatch.setattr(cache, "_hashes_for_key", lambda *_args: [b"x" * 32])
    inserted = cache.insert(InsertParams(key=_key(1, 2), value=torch.tensor([6, 7])))

    _hashes, pages, _items = cache._prepare_finished_prefix(
        _key(1, 2), inserted.last_device_node, request_pages=[3]
    )

    assert pages == [3]


def test_empty_second_match_preserves_allocator_page_record(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    req = SimpleNamespace(_gms_kv_page_ids=[3, 4])
    params = SimpleNamespace(req=req, key=_key())
    result = SimpleNamespace(device_prefix_len=0, last_device_node=None)

    cache._attach_request_pages(params, result)

    assert req._gms_kv_page_ids == [3, 4]


def test_finished_free_hints_cpu_page_ids(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    allocator_calls = []
    native_calls = []
    monkeypatch.setattr(
        adapter,
        "hint_hbm_page_release",
        lambda target, pages: allocator_calls.append((target, pages)) or True,
    )
    monkeypatch.setattr(
        type(cache).__mro__[1],
        "free_kv_row",
        lambda self, kv, ranges: native_calls.append((kv, ranges)),
    )
    kv = SimpleNamespace(get_evicted_seqlen=lambda _component: 0)
    req = SimpleNamespace(
        session=None, kv=kv, _gms_kv_page_ids=[3, 4, 7], finished=lambda: True
    )

    # release_kv_cache claims the row first, then frees the request's own
    # pages; only that release free carries the request's CPU page record.
    assert cache.claim_kv_row(req) is False
    cache.free_kv_row(kv, [(2, 5)])
    cache.free_kv_row(kv, [(6, 8)])

    assert allocator_calls == [(allocator, [4, 7])]
    assert native_calls == [(kv, [(2, 5)]), (kv, [(6, 8)])]


def test_release_hint_never_applies_to_another_kv_record(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    hints = []
    monkeypatch.setattr(
        adapter,
        "hint_hbm_page_release",
        lambda _target, pages: hints.append(pages) or True,
    )
    monkeypatch.setattr(
        type(cache).__mro__[1], "free_kv_row", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(type(cache).__mro__[1], "on_release", lambda *_a, **_k: None)
    kv = SimpleNamespace(get_evicted_seqlen=lambda _component: 0)
    other = SimpleNamespace(get_evicted_seqlen=lambda _component: 0)
    req = SimpleNamespace(session=None, kv=kv, _gms_kv_page_ids=[3, 4])

    assert cache.claim_kv_row(req) is False
    cache.free_kv_row(other, [(0, 4)])
    cache.free_kv_row(kv, [(0, 4)])
    assert cache.claim_kv_row(req) is False
    cache.on_release(req, checkpointed=True)
    cache.free_kv_row(kv, [(0, 4)])

    assert hints == []


def test_finished_prefix_ignores_request_partial_tail_page(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    allocator._gms_kv_leases_by_page = {
        6: KVLease(6, 16),
        7: KVLease(7, 17),
        8: KVLease(8, 18),
    }

    _hashes, pages, items = cache._prepare_finished_prefix(
        _key(1, 2, 3, 4), request_pages=[6, 7, 8]
    )

    assert pages == [6, 7]
    assert [item["slot_ids"] for item in items] == [[6], [7]]


def test_native_eviction_hints_pages_from_cpu_content_map(monkeypatch):
    # Rebuilding a node's content hashes needs Python tree nodes.
    cache, allocator = _cache(monkeypatch, tree_core_backend="python")
    key = _key(1, 2, 3, 4)
    inserted = cache.insert(InsertParams(key=key, value=torch.tensor([12, 13, 14, 15])))
    hashes = cache._hashes_for_key(key, cache.page_size)
    cache._remember_local_pages(hashes, [6, 7])
    hints = []
    native = []
    monkeypatch.setattr(
        adapter,
        "hint_hbm_page_release",
        lambda target, pages: hints.append((target, pages)) or True,
    )
    monkeypatch.setattr(
        type(cache).__mro__[1],
        "_evict_device_leaf",
        lambda self, node_id, tracker: native.append((node_id, tracker)) or "result",
    )

    tracker = {}
    result = cache._evict_device_leaf(inserted.last_device_node, tracker)

    assert result == "result"
    assert hints == [(allocator, [6, 7])]
    assert native == [(inserted.last_device_node, tracker)]


def test_native_eviction_without_python_nodes_releases_unhinted(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    monkeypatch.setattr(
        cache.tree_core,
        "node_by_id",
        lambda _node_id: (_ for _ in ()).throw(NotImplementedError("rust")),
    )
    hints = []
    native = []
    monkeypatch.setattr(
        adapter,
        "hint_hbm_page_release",
        lambda target, pages: hints.append(pages) or True,
    )
    monkeypatch.setattr(
        type(cache).__mro__[1],
        "_evict_device_leaf",
        lambda self, node_id, tracker: native.append(node_id) or "result",
    )

    assert cache._evict_device_leaf(7, {}) == "result"
    assert hints == []
    assert native == [7]


def test_promoted_standby_waits_for_all_rank_gms_recovery(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    directory = _Directory()
    directory._standby = True
    directory.read_view_is_current_writer = True
    cache._gms_directory = directory
    ready = [False]
    calls = []

    class Cohort:
        def all_true(self, stage, value):
            calls.append((stage, value))
            return bool(value)

        def run_intersection(self, stage, operation):
            local = operation()
            calls.append((stage, local))
            return local, []

    from gpu_memory_service.integrations.sglang import writer_lifecycle

    monkeypatch.setattr(writer_lifecycle, "gms_recovery_ready", lambda: ready[0])
    cache._gms_tp = Cohort()

    assert cache._maybe_enter_steady_state() is False
    assert calls == [("steady:gms-recovery-ready", False)]
    ready[0] = True
    calls.clear()
    assert cache._maybe_enter_steady_state() is True
    assert calls[0] == ("steady:gms-recovery-ready", True)
    assert cache._gms_steady_state is True


def test_steady_state_requires_common_current_writer_inventory(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    common_hash = b"c" * 32
    local_only_hash = b"l" * 32
    directory = _Directory()
    directory.read_view_is_current_writer = True
    directory.read_view = [
        (common_hash, {"tier": "hbm", "state": "ready"}),
        (local_only_hash, {"tier": "hbm", "state": "ready"}),
    ]
    cache._gms_directory = directory
    calls = []

    class Cohort:
        def all_true(self, stage, value):
            calls.append((stage, value))
            return True

        def run_intersection(self, stage, operation):
            local = operation()
            calls.append((stage, local))
            return local, [common_hash]

    cache._gms_tp = Cohort()

    assert cache._maybe_enter_steady_state() is True
    assert cache._gms_steady_state is True
    assert cache._gms_recovery_candidates == {common_hash}
    assert calls[0] == ("steady:writer-ready", True)
    assert calls[-1] == ("steady:writer-recheck", True)

    # The transition is one-shot for the writer epoch. Ordinary lookups do not
    # re-enter a TP collective after it has completed.
    calls.clear()
    assert cache._maybe_enter_steady_state() is True
    assert calls == []
    state = install_kv_leases._STATE.get(id(allocator))
    if state is not None:
        assert state["steady_state"] is True


def test_steady_state_uses_committed_snapshot_when_async_view_is_stale(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    content_hash = b"s" * 32
    directory = _Directory()
    directory.read_view_is_current_writer = True
    directory.read_view = []
    directory.snapshot = [(content_hash, {"tier": "hbm", "state": "ready"})]
    cache._gms_directory = directory
    cache._gms_tp = SimpleNamespace(
        all_true=lambda _stage, value: bool(value),
        run_intersection=lambda _stage, operation: (operation(), [content_hash]),
    )

    assert cache._maybe_enter_steady_state() is True
    assert cache._gms_recovery_candidates == {content_hash}


@pytest.mark.parametrize("activated_tokens", [0, 8])
def test_steady_state_bootstraps_zero_writable_pages(monkeypatch, activated_tokens):
    cache, allocator = _cache(monkeypatch)
    directory = _Directory()
    directory.read_view_is_current_writer = True
    cache._gms_directory = directory
    cache._gms_tp = SimpleNamespace(
        all_true=lambda _stage, value: bool(value),
        run_intersection=lambda _stage, operation: (operation(), []),
    )
    state = {"exclusive_hidden_pages": {1, 2}, "steady_state": False}
    monkeypatch.setitem(install_kv_leases._STATE, id(allocator), state)
    monkeypatch.setattr(
        install_kv_leases,
        "enter_exclusive_steady_state",
        lambda _allocator: 0,
    )
    calls = []

    def activate(target, required_tokens, *, max_batch_pages):
        calls.append((target, required_tokens, max_batch_pages))
        return activated_tokens

    monkeypatch.setattr(
        install_kv_leases, "activate_hidden_recovery_capacity", activate
    )
    from gpu_memory_service.integrations.sglang import writer_lifecycle

    # A settled (here: refused) phase two ends the wait after one retry.
    monkeypatch.setattr(writer_lifecycle, "gms_reclaim_refused", lambda: True)
    if activated_tokens:
        assert cache._maybe_enter_steady_state() is True
    else:
        with pytest.raises(RuntimeError, match="no writable KV pages"):
            cache._maybe_enter_steady_state()
    expected = (allocator, allocator.page_size, 256)
    assert calls == [expected] * (1 if activated_tokens else 2)


def test_steady_state_definite_miss_avoids_directory_and_tp_vote(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    cache._gms_recovery_candidates = set()
    key = _key(10, 11)
    cache._gms_directory = _Directory(
        [{"state": "ready", "tier": "hbm", "slot_ids": [4], "generations": [8]}]
    )
    cache._gms_tp = SimpleNamespace(
        leader_true=lambda *_args: pytest.fail(
            "steady native misses must not enter TP consensus"
        )
    )
    cache._gms_directory.lookup_and_claim = lambda *_args: pytest.fail(
        "a definite steady-state miss must not call the directory"
    )
    cache._adopt_directory_suffix = lambda *_args: pytest.fail(
        "steady state without recovery candidates must skip adoption checks"
    )

    result = cache.match_prefix(MatchPrefixParams(key=key))

    assert result.device_prefix_len == 0


def test_steady_state_candidate_miss_hashes_only_first_missing_page(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    key = _key(1, 2, 3, 4, 5, 6)
    hashes = cache._hashes_for_key(key, cache.page_size)
    # A later candidate cannot be adopted without the first missing page.
    cache._gms_recovery_candidates = {hashes[-1]}
    cache._gms_directory.lookup_and_claim = lambda *_args: pytest.fail(
        "a noncontiguous candidate must not reach the directory"
    )
    original_hashes = cache._hashes_for_key
    hashed_lengths = []

    def record_hashes(prefix, page_size):
        hashed_lengths.append(len(prefix))
        return original_hashes(prefix, page_size)

    cache._hashes_for_key = record_hashes
    result = cache.match_prefix(MatchPrefixParams(key=key))

    assert result.device_prefix_len == 0
    assert hashed_lengths == [cache.page_size]


def test_steady_state_publication_skips_digest_vote(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    key = _key(1, 2)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7])))
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    content_hash = cache._hashes_for_key(key, cache.page_size)[0]
    lease = KVLease(3, 12)
    allocator._gms_kv_leases_by_page = {3: lease}
    cache._gms_tp = SimpleNamespace(
        transact_digest=lambda *_args: pytest.fail(
            "steady publication must not enter TP consensus"
        )
    )
    monkeypatch.setattr(
        adapter,
        "_logical_layout_digest",
        lambda *_args: pytest.fail("steady publication must not compute a TP digest"),
    )
    monkeypatch.setattr(adapter, "retain_hbm_pages", lambda *_args: [lease])

    cache._publish_finished_prefix(key)

    assert cache._gms_directory.published[0]["content_hash"] == content_hash
    # The recovery-candidate set is a one-time predecessor snapshot. Current
    # writer publications already live in the native tree and must not make
    # ordinary native misses consult the directory.
    assert content_hash not in cache._gms_recovery_candidates


def test_publication_stays_native_until_writer_visible(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    key = _key(1, 2)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7])))
    directory = _Directory()
    directory.authoritative = False
    cache._gms_directory = directory
    lease = KVLease(3, 12)
    allocator._gms_kv_leases_by_page = {3: lease}
    retained = []

    def retain(_allocator, pages):
        retained.extend(int(page) for page in pages)
        return [lease]

    monkeypatch.setattr(adapter, "retain_hbm_pages", retain)

    cache._publish_finished_prefix(key)

    assert directory.published == []
    assert retained == []
    assert allocator._gms_kv_leases_by_page == {3: lease}

    directory.authoritative = True
    cache._publish_finished_prefix(key)

    assert retained == [3]
    assert len(directory.published) == 1


@pytest.mark.parametrize("operation", ["publish", "adopt"])
def test_tp_logical_layout_agrees_with_rank_local_generations(monkeypatch, operation):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    barrier = threading.Barrier(2, timeout=5)
    votes = [None, None]
    caches = []
    key = _key(1, 2)

    def gather(rank, value):
        votes[rank] = value
        barrier.wait()
        result = list(votes)
        barrier.wait()
        return result

    for rank, generation in enumerate([5, 19]):
        cache, allocator = _cache(monkeypatch)
        cohort = adapter.TPConsistency(world_size=2)
        monkeypatch.setattr(cohort, "leader_true", lambda *_args: True)
        cohort._gather = lambda value, rank=rank: gather(rank, value)
        cohort._gather_digest = lambda stage, ok, agreement, rank=rank: gather(
            rank, (ok, sha256(stage.encode("utf-8") + b"\0" + agreement).digest())
        )
        cache._gms_tp = cohort
        cache._gms_directory = _Directory(
            [
                {
                    "state": "ready",
                    "tier": "hbm",
                    "slot_ids": [3],
                    "generations": [generation],
                }
            ]
        )
        allocator._gms_kv_leases_by_page = {3: KVLease(3, generation)}
        if operation == "publish":
            cache.insert(InsertParams(key=key, value=torch.tensor([6, 7])))
        caches.append(cache)

    monkeypatch.setattr(
        adapter,
        "retain_hbm_pages",
        lambda allocator, _: list(allocator._gms_kv_leases_by_page.values()),
    )
    monkeypatch.setattr(
        adapter,
        "adopt_hbm_pages",
        lambda _allocator, pages, generations: (
            torch.tensor([6, 7]),
            [
                KVLease(page, generation + 1)
                for page, generation in zip(pages, generations)
            ],
        ),
    )

    def execute(cache):
        if operation == "publish":
            cache._publish_finished_prefix(key)
            return cache._gms_directory.published[0]["generations"]
        result = cache.match_prefix(MatchPrefixParams(key=key))
        assert _matched_indices(cache, result) == [6, 7]
        return cache._gms_directory.adopted[0][1][0]["generations"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        generations = list(executor.map(execute, caches))
    assert generations == ([[5], [19]] if operation == "publish" else [[6], [20]])


def test_layout_digest_ignores_local_generation_but_fences_page_identity():
    base = {
        "content_hash": b"h" * 32,
        "engine_id": "engine",
        "slot_ids": [4],
        "generations": [1],
        "tier": "hbm",
        "active": False,
    }
    peer = dict(base, generations=[99])
    divergent = dict(base, slot_ids=[5])

    assert adapter._logical_layout_digest([base]) == adapter._logical_layout_digest(
        [peer]
    )
    assert adapter._logical_layout_digest([base]) != adapter._logical_layout_digest(
        [divergent]
    )


def test_publication_preserves_native_physical_page_order(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    key = _key(1, 2, 3, 4)
    cache.insert(InsertParams(key=key, value=torch.tensor([6, 7, 2, 3])))
    directory = _Directory()
    cache._gms_directory = directory
    leases = {1: KVLease(1, 12), 3: KVLease(3, 34)}
    allocator._gms_kv_leases_by_page = leases
    monkeypatch.setattr(
        adapter, "retain_hbm_pages", lambda *_args: list(leases.values())
    )

    cache._publish_finished_prefix(key)

    assert [item["slot_ids"] for item in directory.published] == [[3], [1]]
    assert [item["generations"] for item in directory.published] == [[34], [12]]
    assert all(item["active"] is False for item in directory.published)


def test_publication_retires_oldest_page_with_batched_generation_tombstone(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    allocator.size = 8
    cache._gms_steady_state = True
    directory = _Directory()
    cache._gms_directory = directory
    old_hashes = [bytes([value]) * 32 for value in (1, 2, 3)]
    new_hash = b"n" * 32
    allocator._gms_kv_leases_by_page = {
        page: KVLease(page, 10 + page) for page in (1, 2, 3, 4)
    }
    allocator._gms_retained_pages.update((1, 2, 3))
    for page, content_hash in zip((1, 2, 3), old_hashes):
        cache._gms_local_pages_by_hash[content_hash] = page
        cache._gms_local_hashes_by_page[page] = {content_hash}
        cache._gms_retained_order[page] = None
        cache._gms_recovery_candidates.add(content_hash)

    def retain(_allocator, pages):
        allocator._gms_retained_pages.update(pages)
        return [allocator._gms_kv_leases_by_page[page] for page in pages]

    def demote(_allocator, pages):
        old = [allocator._gms_kv_leases_by_page[page] for page in pages]
        allocator._gms_retained_pages.difference_update(pages)
        return old

    monkeypatch.setattr(adapter, "retain_hbm_pages", retain)
    monkeypatch.setattr(adapter, "demote_hbm_pages_local", demote)
    item = {
        "content_hash": new_hash,
        "engine_id": "0",
        "slot_ids": [4],
        "generations": [14],
        "tier": "hbm",
        "active": False,
    }

    cache._commit_finished_prefixes([([new_hash], [4], [item])])

    assert directory.published == [
        {
            "content_hash": old_hashes[0],
            "engine_id": cache._gms_engine_id,
            "slot_ids": [1],
            "generations": [11],
            "tier": "hbm",
            "sealed": False,
        },
        item,
    ]
    assert allocator._gms_retained_pages == {2, 3, 4}
    assert list(cache._gms_retained_order) == [2, 3, 4]
    # Demotion ends crash-recovery retention, not SGLang's native residency.
    # Keep content identity until the allocator actually releases the page so
    # native LRU eviction does not need a GPU -> CPU synchronization.
    assert cache._gms_local_pages_by_hash[old_hashes[0]] == 1
    assert cache._gms_local_hashes_by_page[1] == {old_hashes[0]}
    assert old_hashes[0] not in cache._gms_recovery_candidates


def test_publication_does_not_scan_retained_order_below_cap(monkeypatch):
    class NoScanOrder(dict):
        def __iter__(self):
            raise AssertionError("retained order scanned without eviction pressure")

    cache, allocator = _cache(monkeypatch)
    allocator.size = 8
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()
    cache._gms_retained_order = NoScanOrder({1: None})
    allocator._gms_retained_pages.add(1)
    allocator._gms_kv_leases_by_page = {2: KVLease(2, 12)}

    def retain(_allocator, pages):
        allocator._gms_retained_pages.update(pages)
        return [allocator._gms_kv_leases_by_page[page] for page in pages]

    monkeypatch.setattr(adapter, "retain_hbm_pages", retain)
    item = {
        "content_hash": b"n" * 32,
        "engine_id": "0",
        "slot_ids": [2],
        "generations": [12],
        "tier": "hbm",
        "active": False,
    }

    cache._commit_finished_prefixes([([item["content_hash"]], [2], [item])])

    assert cache._gms_directory.published == [item]
    assert allocator._gms_retained_pages == {1, 2}
    assert dict(cache._gms_retained_order) == {1: None, 2: None}


def test_concurrent_identical_prefixes_publish_one_canonical_page(monkeypatch):
    cache, allocator = _cache(monkeypatch)
    content_hash = b"h" * 32
    allocator._gms_kv_leases_by_page = {
        3: KVLease(3, 13),
        4: KVLease(4, 14),
    }
    cache._gms_steady_state = True
    cache._gms_directory = _Directory()

    def retain(_allocator, pages):
        allocator._gms_retained_pages.update(pages)
        return [allocator._gms_kv_leases_by_page[page] for page in pages]

    monkeypatch.setattr(adapter, "retain_hbm_pages", retain)
    first = {
        "content_hash": content_hash,
        "engine_id": "0",
        "slot_ids": [3],
        "generations": [13],
        "tier": "hbm",
        "active": False,
    }
    duplicate = {**first, "slot_ids": [4], "generations": [14]}

    cache._commit_finished_prefixes(
        [([content_hash], [3], [first]), ([content_hash], [4], [duplicate])]
    )

    assert cache._gms_directory.published == [first]
    assert allocator._gms_retained_pages == {3}
    assert cache._gms_local_pages_by_hash == {content_hash: 3}
    assert list(cache._gms_retained_order) == [3]


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
        "retain_hbm_pages",
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

    monkeypatch.setattr(adapter, "retain_hbm_pages", retain)

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

    monkeypatch.setattr(adapter, "retain_hbm_pages", fail_retain)

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

    monkeypatch.setattr(adapter, "retain_hbm_pages", retain)

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

    assert _matched_indices(cache, result) == [8, 9, 4, 5]
    assert len(directory.adopted) == 1
    assert [item["generations"] for item in directory.adopted[0][1]] == [[9], [10]]
    assert directory.released == []


def test_tp_stale_empty_peer_turns_directory_hit_into_common_miss(monkeypatch):
    cache, _allocator = _cache(monkeypatch)
    key = _key(10, 11)
    directory = _Directory(
        [{"state": "ready", "tier": "hbm", "slot_ids": [4], "generations": [8]}]
    )
    cache._gms_directory = directory
    cohort = adapter.TPConsistency(world_size=2)
    monkeypatch.setattr(cohort, "leader_true", lambda *_args: True)

    def gather(value):
        stage = value[0]
        if stage == "adopt:lookup":
            return [value, (stage, True, [])]
        return [value, value]

    monkeypatch.setattr(cohort, "_gather", gather)
    cache._gms_tp = cohort
    monkeypatch.setattr(
        adapter,
        "adopt_hbm_pages",
        lambda *_args: pytest.fail("a non-common TP hit must not be adopted"),
    )

    result = cache.match_prefix(MatchPrefixParams(key=key))

    assert result.device_prefix_len == 0
    assert directory.released == ["claim"]
    assert directory.adopted == []


def test_tp_peer_adoption_failure_is_fatal_before_native_insert(monkeypatch):
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
    cohort = adapter.TPConsistency(world_size=2)
    monkeypatch.setattr(cohort, "leader_true", lambda *_args: True)

    def gather(value):
        stage = value[0]
        if stage == "adopt:leases":
            return [value, (stage, False)]
        return [value, value]

    monkeypatch.setattr(cohort, "_gather", gather)
    cache._gms_tp = cohort
    monkeypatch.setattr(
        cache,
        "insert",
        lambda *_args: pytest.fail("TP adoption failure must precede native insertion"),
    )

    with pytest.raises(adapter.GmsTPConsistencyError, match="adopt:leases"):
        cache.match_prefix(MatchPrefixParams(key=key))

    assert rolled_back == []
    assert len(directory.adopted) == 1
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

    assert result.device_prefix_len == 0
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


def _standby_env(monkeypatch, wait_secs="5"):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("GMS_SGLANG_SHADOW_KV_PLAN_WAIT_SECS", wait_secs)


def test_standby_waits_for_primary_plan_to_complete(monkeypatch):
    _standby_env(monkeypatch)
    calls = []

    def listed(**_kwargs):
        calls.append(1)
        # The primary is still allocating: none, then partial, then complete.
        tags = [[], ["kv:new:a"], ["kv:new:a", "kv:new:b"]][min(len(calls) - 1, 2)]
        return [SimpleNamespace(tag=tag) for tag in tags]

    manager = SimpleNamespace(list_persistent=listed)
    assert install_vmm_ipc_kv._prepare_standby_tag_plan(
        manager, "engine", "kv_pool", ["kv:new:a", "kv:new:b"]
    )
    assert len(calls) == 3


def test_standby_never_allocates_a_missing_plan(monkeypatch):
    _standby_env(monkeypatch, wait_secs="0.6")
    manager = SimpleNamespace(list_persistent=lambda **_kwargs: [])
    with pytest.raises(RuntimeError, match="Timed out waiting for the primary"):
        install_vmm_ipc_kv._prepare_standby_tag_plan(
            manager, "engine", "kv_pool", ["kv:new:a"]
        )


def test_primary_plan_preparation_does_not_wait(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("ENGINE_ID", "0")
    manager = SimpleNamespace(list_persistent=lambda **_kwargs: [])
    assert (
        install_vmm_ipc_kv._prepare_standby_tag_plan(
            manager, "engine", "kv_pool", ["kv:new:a"]
        )
        is False
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
    monkeypatch.setenv("GMS_SGLANG_KV_RECOVERY_MODE", "granular")
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

    monkeypatch.delattr(overrides, "declare_late_resolution", raising=False)
    monkeypatch.delattr(overrides, "resolving_view", raising=False)
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


@pytest.mark.parametrize("requested,expected", [(0, 0), (2, 64), (100, 100)])
def test_pressure_headroom_delegates_to_native_eviction(
    monkeypatch, requested, expected
):
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache, allocator = _cache(monkeypatch)
    allocator.size = 1024
    cache._gms_directory = _Directory()
    observed = []
    monkeypatch.setattr(
        UnifiedRadixCache,
        "evict_for_alloc",
        lambda self, params: observed.append(params) or "native-result",
    )
    params = EvictParams(num_tokens=requested)
    assert cache.evict_for_alloc(params) == "native-result"
    assert observed[0].num_tokens == expected
    assert params.num_tokens == requested


def test_steady_state_eviction_keeps_native_quota(monkeypatch):
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache, allocator = _cache(monkeypatch)
    allocator.size = 1024
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    monkeypatch.setattr(
        UnifiedRadixCache, "evict_for_alloc", lambda self, params: params
    )
    params = EvictParams(num_tokens=2)
    assert cache.evict_for_alloc(params) is params


def test_steady_state_eviction_uses_activated_recovery_capacity(monkeypatch):
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache, allocator = _cache(monkeypatch)
    allocator.size = 1024
    cache._gms_directory = _Directory()
    cache._gms_steady_state = True
    monkeypatch.setattr(
        install_kv_leases,
        "activate_hidden_recovery_capacity",
        lambda observed, required: (
            64
            if observed is allocator and required == 64
            else pytest.fail("unexpected recovery activation")
        ),
    )
    observed = []
    monkeypatch.setattr(
        UnifiedRadixCache,
        "evict_for_alloc",
        lambda self, params: observed.append(params) or EvictResult(),
    )

    result = cache.evict_for_alloc(EvictParams(num_tokens=64))

    assert observed[0].num_tokens == 64
    assert result.num_tokens_evicted == 64


def test_steady_state_eviction_does_not_probe_hidden_pages_when_native_suffices(
    monkeypatch,
):
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache, _allocator = _cache(monkeypatch)
    cache._gms_steady_state = True
    monkeypatch.setattr(
        install_kv_leases,
        "activate_hidden_recovery_capacity",
        lambda *_args: pytest.fail("native eviction met the full shortfall"),
    )
    monkeypatch.setattr(
        UnifiedRadixCache,
        "evict_for_alloc",
        lambda self, params: EvictResult(num_tokens_evicted=params.num_tokens),
    )

    result = cache.evict_for_alloc(EvictParams(num_tokens=64))

    assert result.num_tokens_evicted == 64


def test_non_authoritative_eviction_keeps_native_quota(monkeypatch):
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache, allocator = _cache(monkeypatch)
    allocator.size = 1024
    cache._gms_directory = _Directory()
    cache._gms_directory.authoritative = False
    monkeypatch.setattr(
        UnifiedRadixCache, "evict_for_alloc", lambda self, params: params
    )
    params = EvictParams(num_tokens=2)
    assert cache.evict_for_alloc(params) is params


@pytest.mark.parametrize(("settle_after", "expected"), [(3, 4), (None, 0)])
def test_handoff_waits_for_async_reclaim_before_failing(
    monkeypatch, settle_after, expected
):
    from gpu_memory_service.integrations.sglang import writer_lifecycle

    cache, _allocator = _cache(monkeypatch)
    attempts = []

    def activate(_allocator, _page_size, *, max_batch_pages):
        attempts.append(max_batch_pages)
        return 4 if settle_after is not None and len(attempts) > settle_after else 0

    monkeypatch.setattr(
        writer_lifecycle,
        "gms_reclaim_ready",
        lambda: settle_after is not None and len(attempts) >= settle_after,
    )
    monkeypatch.setattr(writer_lifecycle, "gms_reclaim_refused", lambda: False)
    monkeypatch.setenv("GMS_SGLANG_HANDOFF_CAPACITY_WAIT_SECS", "0.3")

    assert cache._activate_handoff_capacity(activate) == expected
    assert len(attempts) >= 2
    assert set(attempts) == {256}


def test_handoff_stops_waiting_after_terminal_refusal(monkeypatch):
    from gpu_memory_service.integrations.sglang import writer_lifecycle

    cache, _allocator = _cache(monkeypatch)
    attempts = []
    monkeypatch.setattr(writer_lifecycle, "gms_reclaim_ready", lambda: False)
    monkeypatch.setattr(writer_lifecycle, "gms_reclaim_refused", lambda: True)
    monkeypatch.setenv("GMS_SGLANG_HANDOFF_CAPACITY_WAIT_SECS", "3600")

    def activate(*_args, **_kwargs):
        attempts.append(1)
        return 0

    assert cache._activate_handoff_capacity(activate) == 0
    assert len(attempts) == 2


@pytest.mark.parametrize(
    ("shadow", "engine_id", "standby"),
    [("1", "1", True), ("1", "0", False), ("0", "1", False), ("1", "2", True)],
)
def test_sglang_failover_standby_detection(monkeypatch, shadow, engine_id, standby):
    import gpu_memory_service.integrations.sglang as gms_sglang

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", shadow)
    monkeypatch.setenv("ENGINE_ID", engine_id)
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    assert gms_sglang._is_failover_standby() is standby
    assert install_vmm_ipc_kv._is_failover_standby() is standby
