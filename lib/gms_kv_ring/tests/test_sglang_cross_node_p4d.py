# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGL-P4d: cross-node hash registration in SGLang's GMSRadixCache.

Mirrors the vLLM P4d test (`test_register_content_addresses_batch.py`)
but at the SGLang radix-cache layer. Validates that:

  - With GMS_KVR_CROSS_NODE unset, `_spill_node` makes NO cross-node
    registration call.
  - With GMS_KVR_CROSS_NODE=1, `_spill_node` walks the node→root,
    computes per-page hashes via the SHARED `prefix_block_hashes`
    function, and batch-registers them with the daemon client.
  - The hash values match what vLLM would compute for the same
    (tokens, page_size, salt) — cross-engine compatibility.
  - When the daemon reports `skipped=True` (no transport), the
    cache self-disables to avoid per-spill RPC cost.
  - RPC failure self-disables too — best-effort, never blocks the
    spill path.

Runs against REAL SGLang (must be importable) with a mock connector
and mock daemon client. No GPU required."""

from __future__ import annotations

import pytest

# Same warning-suppression preamble as test_sglang_gms_radix_cache.py.
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

sglang = pytest.importorskip(
    "sglang",
    reason="SGLang must be importable in the active Python environment",
)
torch = pytest.importorskip("torch")


from gms_kv_ring.common.prefix_hashes import prefix_block_hashes  # noqa: E402
from gpu_memory_service.integrations.sglang.gms_radix_cache import (  # noqa: E402
    make_gms_radix_cache_class,
)

# ---------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------


class _FakeDaemonClient:
    def __init__(self, skipped: bool = False, raise_on_call: bool = False):
        self._skipped = skipped
        self._raise = raise_on_call
        self.calls: list[list[dict]] = []
        self.staging_hits: dict[bytes, dict] = {}
        self.scan_calls: list[list[bytes]] = []

    def register_content_addresses_batch(self, items):
        if self._raise:
            raise RuntimeError("simulated daemon error")
        self.calls.append([dict(it) for it in items])
        total = sum(sum(r[2] for r in it["ranges"]) for it in items)
        return (total, self._skipped)

    def staging_scan(self, hashes):
        self.scan_calls.append(list(hashes))
        return {h: self.staging_hits[h] for h in hashes if h in self.staging_hits}


class _FakeHandle:
    def __init__(self, client):
        self._client = client
        self.restore_staging_ranges_calls: list[list[tuple]] = []
        self.restore_staging_ranges_ok = True

    def restore_staging_ranges_sync(self, range_hits):
        self.restore_staging_ranges_calls.append(
            [(h, gen, list(ranges)) for h, gen, ranges in range_hits]
        )
        return self.restore_staging_ranges_ok


class _FakeBlockConnector:
    """Mock BlockGdsConnector that succeeds on every spill and
    exposes a `.handle._client` that the GMSRadixCache uses for
    the cross-node RPC."""

    def __init__(self, daemon_client: _FakeDaemonClient):
        self.handle = _FakeHandle(daemon_client)
        self.storage: dict[int, int] = {}

    def is_available(self) -> bool:
        return True

    def evict_blocks_to_storage(self, block_ids, generations=None):
        from gms_kv_ring.engines.gds_block_connector import EvictResult

        bids = list(block_ids)
        for bid in bids:
            self.storage[bid] = bid
        return EvictResult(succeeded=len(bids), failed=0, failed_ids=[])

    def restore_blocks_remap(self, triples):
        return {dst: True for entry in triples for dst, *_ in [entry[1:]]}


class _FakeAllocator:
    def __init__(self):
        self.device = torch.device("cpu")
        self._next = 1_000_000

    def alloc(self, n: int):
        if n <= 0:
            return torch.empty((0,), dtype=torch.int64)
        out = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return out

    def free(self, indices):
        if not hasattr(self, "freed"):
            self.freed = []
        self.freed.append([int(v) for v in indices.tolist()])


# ---------------------------------------------------------------------
# Cache construction helper
# ---------------------------------------------------------------------


def _build_cache(daemon_client, page_size: int = 4):
    """Build a GMSRadixCache plumbed with the fake daemon client."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    alloc = _FakeAllocator()
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=alloc,
        page_size=page_size,
        enable_kv_cache_events=False,
    )

    def layout(slot_idx):
        # Single layer; one byte range per slot of 64B.
        return [(0, int(slot_idx) * 64, 64)]

    conn = _FakeBlockConnector(daemon_client)
    Cls = make_gms_radix_cache_class()
    return (
        Cls(
            params,
            gds=conn,
            block_layout_fn=layout,
            engine_id="sgl-test",
        ),
        conn,
        alloc,
    )


# Insert / evict / match shim — picks the right SGLang API.
_NEW_API = False
EvictParams = InsertParams = MatchPrefixParams = None
for _mod in (
    "sglang.srt.mem_cache.base_prefix_cache",
    "sglang.srt.mem_cache.cache_init_params",
):
    try:
        _m = __import__(
            _mod,
            fromlist=["EvictParams", "InsertParams", "MatchPrefixParams"],
        )
        EvictParams = _m.EvictParams
        InsertParams = _m.InsertParams
        MatchPrefixParams = _m.MatchPrefixParams
        _NEW_API = True
        break
    except (ImportError, AttributeError):
        continue

from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402


def _insert(cache, token_ids):
    n = len(token_ids)
    value = cache.token_to_kv_pool_allocator.alloc(n)
    key = RadixKey(token_ids=token_ids, extra_key=None)
    if _NEW_API:
        cache.insert(InsertParams(key=key, value=value, priority=0))
    else:
        cache.insert(key, value=value, priority=0)
    return value


def _evict(cache, num_tokens):
    if _NEW_API:
        return cache.evict(EvictParams(num_tokens=num_tokens))
    cache.evict(num_tokens)
    return None


def _match(cache, token_ids):
    key = RadixKey(token_ids=token_ids, extra_key=None)
    if _NEW_API:
        return cache.match_prefix(MatchPrefixParams(key=key))
    return cache.match_prefix(key)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_no_registration_when_env_unset(monkeypatch):
    """Default (no env): _spill_node must NOT call
    register_content_addresses_batch."""
    monkeypatch.delenv("GMS_KVR_CROSS_NODE", raising=False)
    daemon = _FakeDaemonClient()
    cache, conn, alloc = _build_cache(daemon, page_size=4)
    assert cache._cross_node_register is False

    _insert(cache, [10, 20, 30, 40, 50, 60, 70, 80])
    _evict(cache, num_tokens=8)
    assert daemon.calls == []
    # And the spill itself still landed (other paths unaffected).
    assert len(cache._spilled) >= 1


def test_registers_per_page_hashes_when_enabled(monkeypatch):
    """With GMS_KVR_CROSS_NODE=1, _spill_node must batch-register
    one item PER PAGE_SIZE chunk of slot_indices, with hashes
    matching the shared prefix_block_hashes algorithm."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    monkeypatch.delenv("GMS_KVR_CROSS_NODE_SALT", raising=False)
    daemon = _FakeDaemonClient()
    PAGE = 4
    cache, conn, alloc = _build_cache(daemon, page_size=PAGE)
    assert cache._cross_node_register is True

    tokens = [10, 20, 30, 40, 50, 60, 70, 80]  # 2 full pages
    val = _insert(cache, tokens)
    orig_slots = [int(x) for x in val.tolist()]
    _evict(cache, num_tokens=8)

    # Exactly one batched call.
    assert len(daemon.calls) == 1
    items = daemon.calls[0]
    # 8 slots / page_size=4 = 2 items (one per page).
    assert len(items) == 2

    # Hash values must match what `prefix_block_hashes` computes
    # for the (full prefix, page_size, "" salt) — i.e., the same
    # algorithm vLLM uses, so cross-engine matching works.
    expected = prefix_block_hashes(tokens, PAGE, "")
    assert len(expected) == 2
    assert items[0]["content_hash"] == expected[0]
    assert items[1]["content_hash"] == expected[1]

    # Engine_id and ranges shape are right.
    assert items[0]["engine_id"] == "sgl-test"
    assert items[1]["engine_id"] == "sgl-test"
    # Each item's ranges cover PAGE slots × 1 layer = PAGE entries.
    assert len(items[0]["ranges"]) == PAGE
    assert len(items[1]["ranges"]) == PAGE
    # Layout fn was layer=0, offset=slot*64, size=64.
    first_slot = orig_slots[0]
    assert items[0]["ranges"][0] == (0, first_slot * 64, 64)


def test_skipped_self_disables(monkeypatch):
    """When the daemon returns skipped=True (no transport), the
    cache must self-disable to avoid the per-spill RPC cost on
    subsequent evictions."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    daemon = _FakeDaemonClient(skipped=True)
    cache, conn, alloc = _build_cache(daemon, page_size=4)

    _insert(cache, [1, 2, 3, 4])
    _evict(cache, num_tokens=4)
    # First call made (and returned skipped).
    assert len(daemon.calls) == 1
    # Self-disabled after seeing skipped=True.
    assert cache._cross_node_register is False

    # Second eviction must NOT hit the daemon.
    _insert(cache, [5, 6, 7, 8])
    _evict(cache, num_tokens=4)
    assert len(daemon.calls) == 1  # unchanged


def test_rpc_failure_self_disables(monkeypatch):
    """Best-effort: an RPC exception must not block the spill;
    the cache self-disables for the remainder of its lifetime."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    daemon = _FakeDaemonClient(raise_on_call=True)
    cache, conn, alloc = _build_cache(daemon, page_size=4)

    _insert(cache, [1, 2, 3, 4])
    _evict(cache, num_tokens=4)
    # Spill still happened (cache._spilled populated).
    assert len(cache._spilled) >= 1
    # And cross-node was disabled.
    assert cache._cross_node_register is False


def test_misaligned_page_size_skips_registration(monkeypatch):
    """When a node's slot count isn't a multiple of page_size, the
    hook skips registration (the hashes wouldn't match what a peer
    would compute on lookup)."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    daemon = _FakeDaemonClient()
    cache, conn, alloc = _build_cache(daemon, page_size=4)

    # 5 tokens — not a multiple of page_size=4. SGLang may store
    # 5 slots in a single node depending on insertion behavior;
    # the hook should NOT register a hash for a partial page.
    _insert(cache, [1, 2, 3, 4, 5])
    _evict(cache, num_tokens=5)
    # Either no spilled nodes (because the cache decided to split),
    # OR a spilled node was misaligned and skipped. Either way:
    # no daemon calls should mention partial pages.
    for call in daemon.calls:
        for item in call:
            # Each item's range count must be a multiple of page_size.
            assert len(item["ranges"]) % 4 == 0


def test_salt_changes_hash(monkeypatch):
    """Different GMS_KVR_CROSS_NODE_SALT must yield different
    hashes for the same tokens — cache-isolation property."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    monkeypatch.setenv("GMS_KVR_CROSS_NODE_SALT", "tenant-A")
    daemon_a = _FakeDaemonClient()
    cache_a, _, _ = _build_cache(daemon_a, page_size=4)
    _insert(cache_a, [1, 2, 3, 4])
    _evict(cache_a, num_tokens=4)
    assert len(daemon_a.calls) == 1
    hash_a = daemon_a.calls[0][0]["content_hash"]

    monkeypatch.setenv("GMS_KVR_CROSS_NODE_SALT", "tenant-B")
    daemon_b = _FakeDaemonClient()
    cache_b, _, _ = _build_cache(daemon_b, page_size=4)
    _insert(cache_b, [1, 2, 3, 4])
    _evict(cache_b, num_tokens=4)
    assert len(daemon_b.calls) == 1
    hash_b = daemon_b.calls[0][0]["content_hash"]

    assert hash_a != hash_b


def test_match_prefix_restores_cold_staged_pages(monkeypatch):
    """A cold SGLang cache can consume staged cross-node pages by
    restoring them into fresh slots and inserting the prefix."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    daemon = _FakeDaemonClient()
    cache, conn, alloc = _build_cache(daemon, page_size=4)
    cache._staging_scan = daemon.staging_scan

    tokens = [101, 102, 103, 104, 105, 106, 107, 108]
    hashes = prefix_block_hashes(tokens, 4, "")
    daemon.staging_hits = {
        hashes[0]: {"generation": 7, "bytes_size": 256, "crc32": 1},
        hashes[1]: {"generation": 8, "bytes_size": 256, "crc32": 2},
    }

    result = _match(cache, tokens)

    assert len(result.device_indices) == len(tokens)
    assert len(daemon.scan_calls) == 1
    assert daemon.scan_calls[0] == hashes
    assert len(conn.handle.restore_staging_ranges_calls) == 1
    call = conn.handle.restore_staging_ranges_calls[0]
    assert [entry[0] for entry in call] == hashes
    assert [entry[1] for entry in call] == [7, 8]
    # One page hash maps to page_size explicit per-slot ranges.
    assert len(call[0][2]) == 4
    first_slot = int(result.device_indices[0])
    assert call[0][2][0] == (0, first_slot * 64, 64)

    # The prefix is now resident locally; the next match does not
    # need another staged restore.
    result2 = _match(cache, tokens)
    assert len(result2.device_indices) == len(tokens)
    assert len(conn.handle.restore_staging_ranges_calls) == 1


def test_match_prefix_restores_only_missing_staged_suffix(monkeypatch):
    """If the first page is already local, SGLang restores only the
    staged suffix and inserts it after the existing local value."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    daemon = _FakeDaemonClient()
    cache, conn, alloc = _build_cache(daemon, page_size=4)
    cache._staging_scan = daemon.staging_scan

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    local = _insert(cache, tokens[:4])
    hashes = prefix_block_hashes(tokens, 4, "")
    daemon.staging_hits = {
        hashes[1]: {"generation": 12, "bytes_size": 256, "crc32": 9},
    }

    result = _match(cache, tokens)

    assert len(result.device_indices) == len(tokens)
    assert result.device_indices[:4].tolist() == local.tolist()
    assert len(conn.handle.restore_staging_ranges_calls) == 1
    call = conn.handle.restore_staging_ranges_calls[0]
    assert [entry[0] for entry in call] == [hashes[1]]
    assert [entry[1] for entry in call] == [12]
    assert daemon.scan_calls[0] == [hashes[1]]


def test_match_prefix_frees_slots_when_staged_restore_fails(monkeypatch):
    """A failed staged copy must not leak freshly allocated SGLang
    slots or expose a partial prefix hit."""
    monkeypatch.setenv("GMS_KVR_CROSS_NODE", "1")
    daemon = _FakeDaemonClient()
    cache, conn, alloc = _build_cache(daemon, page_size=4)
    cache._staging_scan = daemon.staging_scan
    conn.handle.restore_staging_ranges_ok = False

    tokens = [10, 20, 30, 40]
    h = prefix_block_hashes(tokens, 4, "")[0]
    daemon.staging_hits = {
        h: {"generation": 3, "bytes_size": 128, "crc32": 4},
    }

    result = _match(cache, tokens)

    assert len(result.device_indices) == 0
    assert len(conn.handle.restore_staging_ranges_calls) == 1
    assert getattr(alloc, "freed", []) == [
        [1_000_000, 1_000_001, 1_000_002, 1_000_003],
    ]
