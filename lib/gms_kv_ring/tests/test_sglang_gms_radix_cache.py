# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GMSRadixCache — SGLang's RadixCache subclass
that routes eviction through the gms_kv_ring daemon.

These tests run against REAL SGLang (must be importable in the
test env) but use:
  - a mock BlockGdsConnector that stores bytes in a dict
  - a mock allocator that hands out monotonic slot indices

Together they let us validate the spill / restore round-trip
WITHOUT a GPU. The byte-correctness invariant is "what we
stored at spill is what we get back on restore."
"""

from __future__ import annotations

import os

import pytest

# SGLang upstream main imports cutlass DSL which raises a
# SMEM_CAPACIT-related DeprecationWarning. The repo's
# filterwarnings=error escalates that to a fatal collection
# error. Scope-ignore for this real-SGLang test only.
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

sglang = pytest.importorskip(
    "sglang",
    reason="SGLang must be importable in the active Python environment",
)
torch = pytest.importorskip("torch")


from gpu_memory_service.integrations.sglang.gms_radix_cache import (  # noqa: E402
    make_gms_radix_cache_class,
)

# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------


class _FakeBlockConnector:
    """Mock BlockGdsConnector: stores per-slot byte payloads
    in a dict so we can verify byte-correctness round-trips."""

    def __init__(self, available: bool = True):
        self._available = available
        # block_id → "bytes" (we just store the slot's integer
        # value here as a stand-in for actual KV bytes; the
        # restore path's job is to recover EXACTLY that value
        # at the destination slot).
        self.storage: dict[int, int] = {}
        self.evict_calls: list[list[int]] = []
        self.host_evict_calls: list[list[int]] = []
        self.restore_calls: list[list[tuple]] = []
        self.host_restore_calls: list[tuple[str, list[tuple]]] = []

    def is_available(self) -> bool:
        return self._available

    def evict_blocks_to_storage(self, block_ids, generations=None):
        from gms_kv_ring.engines.gds_block_connector import EvictResult

        bids = list(block_ids)
        self.evict_calls.append(bids)
        for bid in bids:
            # Stand-in for "the bytes at this slot." The slot's
            # original index doubles as its payload.
            self.storage[bid] = bid
        return EvictResult(succeeded=len(bids), failed=0, failed_ids=[])

    def evict_blocks_to_host(self, block_ids, generations=None):
        from gms_kv_ring.engines.gds_block_connector import EvictResult

        bids = list(block_ids)
        self.host_evict_calls.append(bids)
        for bid in bids:
            self.storage[bid] = bid
        return EvictResult(succeeded=len(bids), failed=0, failed_ids=[])

    def restore_blocks_remap(self, triples):
        triples = list(triples)
        self.restore_calls.append(triples)
        out = {}
        for entry in triples:
            if len(entry) == 3:
                src, dst, _gen = entry
            else:
                src, dst = entry
            out[dst] = src in self.storage
        return out

    def restore_blocks_remap_from_host(self, src_engine_id, triples):
        triples = list(triples)
        self.host_restore_calls.append((str(src_engine_id), triples))
        out = {}
        for entry in triples:
            if len(entry) == 3:
                src, dst, _gen = entry
            else:
                src, dst = entry
            out[dst] = src in self.storage
        return out


class _FakeAllocator:
    """Mock SGLang allocator. alloc() returns int64 tensors of
    monotonically-increasing slot indices; free() just tracks
    what was freed. Slots are NEVER reused so we can detect
    accidental aliasing in tests."""

    def __init__(self):
        self.device = torch.device("cpu")
        self._next = 1_000_000
        self.freed: list[int] = []

    def alloc(self, n: int):
        if n <= 0:
            return torch.empty((0,), dtype=torch.int64)
        out = torch.arange(
            self._next,
            self._next + n,
            dtype=torch.int64,
        )
        self._next += n
        return out

    def free(self, indices):
        if hasattr(indices, "tolist"):
            self.freed.extend(int(x) for x in indices.tolist())
        else:
            self.freed.extend(int(x) for x in indices)


def _build_cache(connector, allocator, page_size=1):
    """Construct a GMSRadixCache configured for unit testing.
    Bypasses the SGLang Scheduler entirely."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        enable_kv_cache_events=False,
    )

    def layout(slot_idx):
        # One layer, single byte range per slot. Sufficient for
        # the mock connector — real backends use this fn to find
        # the per-layer HBM byte ranges to read/write.
        return [(0, int(slot_idx) * 64, 64)]

    Cls = make_gms_radix_cache_class()
    return Cls(
        params,
        gds=connector,
        block_layout_fn=layout,
        engine_id="test-eng",
    )


# SGLang API version detection — the test helpers below dispatch
# to whichever signature shape is present.
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


def _insert(cache, token_ids: list[int]):
    """Insert a sequence into the radix tree at face-value
    (slot indices = result of allocator.alloc())."""
    n = len(token_ids)
    value = cache.token_to_kv_pool_allocator.alloc(n)
    key = RadixKey(token_ids=token_ids, extra_key=None)
    if _NEW_API:
        cache.insert(InsertParams(key=key, value=value, priority=0))
    else:
        cache.insert(key, value=value, priority=0)
    return value


def _match(cache, token_ids: list[int]):
    key = RadixKey(token_ids=token_ids, extra_key=None)
    if _NEW_API:
        return cache.match_prefix(MatchPrefixParams(key=key))
    return cache.match_prefix(key)


def _evict(cache, num_tokens: int):
    if _NEW_API:
        return cache.evict(EvictParams(num_tokens=num_tokens))
    # Old API returns None; tests must not rely on the return.
    cache.evict(num_tokens)
    return None


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_falls_back_to_vanilla_when_gds_unavailable():
    """When connector reports `is_available()=False`, our
    overrides must collapse to vanilla RadixCache behavior:
    eviction frees + deletes the leaf, no spill recorded."""
    conn = _FakeBlockConnector(available=False)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    _insert(cache, [1, 2, 3, 4])
    res = _evict(cache, num_tokens=1)
    assert res.num_tokens_evicted >= 1
    # No spill should have happened.
    assert conn.evict_calls == []
    # Leaf was DELETED (vanilla path): match returns empty.
    m = _match(cache, [1, 2, 3, 4])
    assert m.device_indices.numel() == 0


def test_evict_spills_to_daemon_and_retains_leaf():
    """With an active connector: eviction spills bytes to the
    daemon AND keeps the tree node so future match_prefix can
    restore."""
    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    val = _insert(cache, [10, 20, 30, 40])
    orig_slots = [int(x) for x in val.tolist()]

    res = _evict(cache, num_tokens=4)
    assert res.num_tokens_evicted == 4

    # Connector saw the spill with the original slot indices.
    assert len(conn.evict_calls) == 1
    assert sorted(conn.evict_calls[0]) == sorted(orig_slots)

    # The HBM slots were returned to the allocator.
    assert sorted(alloc.freed) == sorted(orig_slots)

    # The spill table records this node.
    assert len(cache._spilled) == 1


def test_match_prefix_after_evict_restores_via_daemon():
    """Insert → evict (spills) → match_prefix should restore
    bytes by allocating fresh HBM slots and asking the daemon
    to refill them. We verify byte-correctness: every fresh
    slot's mapped src is the SAME slot that was originally
    spilled (the daemon's `storage[src]` matched dst's restore
    payload)."""
    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    val = _insert(cache, [10, 20, 30, 40])
    orig_slots = [int(x) for x in val.tolist()]

    _evict(cache, num_tokens=4)
    assert len(cache._spilled) == 1

    m = _match(cache, [10, 20, 30, 40])
    # All 4 tokens matched.
    assert m.device_indices.numel() == 4

    # A restore call fired with src=orig_slots, dst=fresh slots.
    assert len(conn.restore_calls) == 1
    triples = conn.restore_calls[0]
    src_ids = [t[0] for t in triples]
    dst_ids = [t[1] for t in triples]
    assert sorted(src_ids) == sorted(orig_slots), (
        "restore src must come from the spill record's "
        "block_ids (original slot indices)"
    )
    # Dst must be FRESH (not the same as original — allocator
    # never reuses).
    assert not (set(dst_ids) & set(orig_slots)), (
        "restore dst should be fresh slot indices, never the "
        "original (allocator gives monotonic indices)"
    )

    # Spill record is dropped after successful restore.
    assert len(cache._spilled) == 0


def test_restore_handles_partial_failure_by_truncation():
    """If the daemon fails to restore some blocks, the match
    must truncate at that point. The caller sees a SHORTER
    (still correct) cache hit instead of garbage."""

    class _PartialFailConnector(_FakeBlockConnector):
        def restore_blocks_remap(self, triples):
            # Fail every restore by returning False for all dst.
            triples = list(triples)
            self.restore_calls.append(triples)
            out = {}
            for entry in triples:
                if len(entry) == 3:
                    _src, dst, _g = entry
                else:
                    _src, dst = entry
                out[dst] = False
            return out

    conn = _PartialFailConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    _insert(cache, [1, 2, 3, 4])
    _evict(cache, num_tokens=4)

    m = _match(cache, [1, 2, 3, 4])
    # Restore failed → match must truncate. With a single
    # spilled node containing all 4 tokens, the truncation
    # falls back to the root (zero match).
    assert m.device_indices.numel() == 0


def test_spill_table_respects_max_cap(monkeypatch):
    """The in-memory spill table must not grow without bound.
    `_max_spilled` evicts oldest entries to make room. (The
    daemon side has its own retention via sweeper.)"""
    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()

    # Tight cap so we hit it quickly.
    monkeypatch.setenv("GMS_SGLANG_MAX_SPILLED_NODES", "2")
    cache = _build_cache(conn, alloc)

    # Three distinct prefixes → three spilled nodes; the cap
    # should drop the oldest entry leaving 2.
    _insert(cache, [1, 2])
    _evict(cache, num_tokens=2)
    _insert(cache, [3, 4])
    _evict(cache, num_tokens=2)
    _insert(cache, [5, 6])
    _evict(cache, num_tokens=2)

    assert len(cache._spilled) == 2, f"spill table grew past cap: {len(cache._spilled)}"


def test_two_disjoint_prefixes_round_trip():
    """Two unrelated prefixes spill + restore independently."""
    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    val_a = _insert(cache, [1, 2, 3])
    val_b = _insert(cache, [9, 8, 7])
    a_slots = [int(x) for x in val_a.tolist()]
    b_slots = [int(x) for x in val_b.tolist()]

    _evict(cache, num_tokens=6)
    assert len(cache._spilled) == 2

    # Restore A.
    m_a = _match(cache, [1, 2, 3])
    assert m_a.device_indices.numel() == 3
    triples_a = conn.restore_calls[-1]
    assert sorted(t[0] for t in triples_a) == sorted(a_slots)

    # Restore B.
    m_b = _match(cache, [9, 8, 7])
    assert m_b.device_indices.numel() == 3
    triples_b = conn.restore_calls[-1]
    assert sorted(t[0] for t in triples_b) == sorted(b_slots)


def test_phase_b_drop_on_restore_failure(monkeypatch):
    """Phase B: when a restore fails partway, drop the spill
    record so subsequent match_prefix attempts on the same
    prefix DON'T retry the broken slots. The first request
    sees the failure → spill record cleared → second request
    treats the prefix as cache-miss."""
    fail_seen = {"n": 0}

    class _FailingConnector(_FakeBlockConnector):
        def restore_blocks_remap(self, triples):
            triples = list(triples)
            self.restore_calls.append(triples)
            fail_seen["n"] += 1
            return {(t[1] if len(t) >= 2 else t[1]): False for t in triples}

    conn = _FailingConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    _insert(cache, [1, 2, 3, 4])
    _evict(cache, num_tokens=4)
    assert len(cache._spilled) == 1, "spill should be recorded"

    # First match: restore is attempted and fails.
    _match(cache, [1, 2, 3, 4])
    assert (
        fail_seen["n"] == 1
    ), f"expected exactly one restore attempt; got {fail_seen['n']}"
    assert len(cache._spilled) == 0, "Phase B: spill record must be dropped on failure"

    # Second match: must NOT retry the daemon. The tree node
    # has value=None and is no longer in _spilled → walk
    # breaks at the first child, returning a cache miss without
    # invoking the connector.
    _match(cache, [1, 2, 3, 4])
    assert fail_seen["n"] == 1, (
        "Phase B failed: second match retried the broken slots "
        f"(restore_calls total = {fail_seen['n']})"
    )


def test_reclaim_cached_kv_spills_evictable_leaf_before_free():
    conn = _FakeBlockConnector()
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    value = _insert(cache, [1, 2, 3, 4])
    reclaimed = cache.reclaim_cached_kv(1)

    assert reclaimed == len(value)
    assert conn.evict_calls == [value.tolist()]
    assert alloc.freed == value.tolist()
    assert len(cache._spilled) == 1


def test_reclaim_cached_kv_can_spill_to_host_tier_without_gds(monkeypatch):
    monkeypatch.setenv("GMS_KVR_HOST_TIER_FALLBACK", "1")
    conn = _FakeBlockConnector(available=False)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    value = _insert(cache, [1, 2, 3, 4])
    reclaimed = cache.reclaim_cached_kv(1)

    assert reclaimed == len(value)
    assert conn.evict_calls == []
    assert conn.host_evict_calls == [value.tolist()]
    assert alloc.freed == value.tolist()


def test_host_tier_mirror_spills_to_host_even_when_gds_available(monkeypatch):
    monkeypatch.setenv("GMS_KVR_HOST_TIER_FALLBACK", "1")
    monkeypatch.setenv("GMS_KVR_HOST_TIER_MIRROR", "1")
    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    value = _insert(cache, [1, 2, 3, 4])
    _evict(cache, num_tokens=4)

    assert conn.evict_calls == []
    assert conn.host_evict_calls == [value.tolist()]
    assert alloc.freed == value.tolist()


def test_shadow_restore_uses_source_engine_host_tier(monkeypatch):
    monkeypatch.setenv("GMS_KVR_HOST_TIER_FALLBACK", "1")
    monkeypatch.setenv("GMS_SGLANG_SOURCE_ENGINE_ID", "primary")
    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    value = _insert(cache, [10, 20, 30, 40])
    _evict(cache, num_tokens=4)

    m = _match(cache, [10, 20, 30, 40])

    assert m.device_indices.numel() == 4
    assert conn.host_restore_calls
    src_engine, triples = conn.host_restore_calls[0]
    assert src_engine == "primary"
    assert sorted(t[0] for t in triples) == sorted(value.tolist())


def test_phase_a_invalidate_all_spilled_drops_state():
    """Phase A: invalidate_all_spilled() clears every spill
    record (the daemon-restart escape hatch)."""
    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    _insert(cache, [1, 2])
    _insert(cache, [3, 4])
    _evict(cache, num_tokens=4)
    assert len(cache._spilled) == 2

    cache.invalidate_all_spilled()
    assert len(cache._spilled) == 0

    # After invalidate, a match must NOT trigger any restore
    # call — the daemon (in a real failover scenario) has lost
    # the bytes and we don't want to ask for them.
    before = len(conn.restore_calls)
    _match(cache, [1, 2])
    assert len(conn.restore_calls) == before


def test_phase_a_epoch_poll_thread_disabled_when_no_socket():
    """Without a daemon_socket, the epoch thread MUST NOT
    start — old SGLang tests and any deployment without
    daemon-restart concerns shouldn't pay for a thread."""
    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)
    assert cache._epoch_thread is None
    assert cache._epoch_stop is not None  # event always exists
    cache.shutdown()  # no-op


def test_phase_a_epoch_change_triggers_invalidation(tmp_path):
    """End-to-end Phase A: a real daemon restart (different
    daemon process on the same socket) bumps the epoch; the
    poll thread sees the change and drops every spill
    record."""
    import asyncio
    import threading as _t
    import time as _time

    from gms_kv_ring.daemon.server import Daemon

    sock = str(tmp_path / "d.sock")
    holders: list = []

    def _spin(d, holder):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        holder["daemon"] = d
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()

    # Start daemon 1.
    d1 = Daemon(sock, supervise_backend=False)
    h1: dict = {}
    holders.append(h1)
    t1 = _t.Thread(target=_spin, args=(d1, h1), daemon=True)
    t1.start()
    deadline = _time.monotonic() + 3.0
    while not os.path.exists(sock) and _time.monotonic() < deadline:
        _time.sleep(0.02)

    conn = _FakeBlockConnector(available=True)
    alloc = _FakeAllocator()
    os.environ["GMS_SGLANG_EPOCH_POLL_S"] = "0.3"
    try:
        cache = _build_cache_with_socket(conn, alloc, sock)
    finally:
        del os.environ["GMS_SGLANG_EPOCH_POLL_S"]

    _insert(cache, [1, 2, 3, 4])
    _evict(cache, num_tokens=4)
    assert len(cache._spilled) == 1

    # Let the poll thread settle on d1's epoch.
    deadline = _time.monotonic() + 3.0
    while cache._daemon_epoch_seen is None and _time.monotonic() < deadline:
        _time.sleep(0.1)
    assert cache._daemon_epoch_seen is not None, "poll thread never observed d1's epoch"

    # Stop d1's serve loop cleanly — the daemon's __init__
    # bound `_stop_event` (asyncio.Event) at construct time;
    # signal it via call_soon_threadsafe.
    if h1.get("loop") is not None and d1._stop_event is not None:
        h1["loop"].call_soon_threadsafe(d1._stop_event.set)
    t1.join(timeout=5)

    # Force the client to disconnect (the socket fd is now
    # broken). Without this, the poll thread's persistent
    # connection might still buffer some response and not
    # notice d1 died until its next call fails. Sleeping for
    # one poll interval is enough for the poll to attempt
    # another call and notice the broken pipe.
    _time.sleep(0.5)

    # Sanity: socket file is gone (d1 cleaned up).
    if os.path.exists(sock):
        os.unlink(sock)

    # Start d2 on the same path — fresh epoch.
    _time.sleep(0.01)
    d2 = Daemon(sock, supervise_backend=False)
    h2: dict = {}
    holders.append(h2)
    t2 = _t.Thread(target=_spin, args=(d2, h2), daemon=True)
    t2.start()
    deadline = _time.monotonic() + 3.0
    while not os.path.exists(sock) and _time.monotonic() < deadline:
        _time.sleep(0.02)

    # Pause the background thread by setting stop while we
    # drive checks directly (avoids interleaving).
    cache._epoch_stop.set()
    if cache._epoch_thread:
        cache._epoch_thread.join(timeout=2.0)
    cache._epoch_stop.clear()

    # Reset client state to force fresh reconnect.
    if cache._epoch_client is not None:
        try:
            cache._epoch_client.close()
        except Exception:
            pass
        cache._epoch_client = None

    # Direct call: reconnects to d2 (since client is None) and
    # observes d2's epoch → triggers invalidation.
    cache._check_daemon_epoch_once()

    try:
        assert len(cache._spilled) == 0, (
            "Phase A failed: spill records not dropped after "
            "daemon restart; cache._spilled still has "
            f"{len(cache._spilled)} entries; "
            f"d1.epoch={d1.epoch}, d2.epoch={d2.epoch}, "
            f"seen={cache._daemon_epoch_seen}"
        )
        assert cache._daemon_epoch_seen == d2.epoch, (
            f"poll didn't update epoch to d2's: "
            f"seen={cache._daemon_epoch_seen} d2={d2.epoch}"
        )
    finally:
        cache.shutdown()
        if h2.get("loop") is not None and d2._stop_event is not None:
            h2["loop"].call_soon_threadsafe(d2._stop_event.set)
        t2.join(timeout=5)


# ---------------------------------------------------------------------
# install_gms_radix_cache.py tests — Phase 3
# ---------------------------------------------------------------------


def test_install_noop_when_env_unset(monkeypatch):
    """Without GMS_SGLANG_ENABLE_KV_RING=1, install() must
    return False and NOT patch SGLang's Scheduler. Default
    behavior is "no monkey-patch surprises" — opt-in only."""
    monkeypatch.delenv("GMS_SGLANG_ENABLE_KV_RING", raising=False)
    # Reset module state.
    import gpu_memory_service.integrations.sglang.install_gms_radix_cache as inst

    inst._INSTALLED = False
    assert inst.install() is False
    assert inst._INSTALLED is False


def test_install_idempotent_when_env_set(monkeypatch):
    """Second install() with the env still set returns False
    (already installed). The patch must not stack."""
    monkeypatch.setenv("GMS_SGLANG_ENABLE_KV_RING", "1")
    import gpu_memory_service.integrations.sglang.install_gms_radix_cache as inst

    # Restore module-level guard for a clean test.
    inst._INSTALLED = False
    from sglang.srt.managers.scheduler import Scheduler

    original_init = Scheduler.__init__
    try:
        first = inst.install()
        # first MAY be False if a previous test already
        # patched (Scheduler.__init__._gms_patched is set
        # process-wide). Either way the second call must NOT
        # re-patch.
        if first:
            assert getattr(
                Scheduler.__init__,
                "_gms_patched",
                False,
            ), "expected patched init after first install()"
        second = inst.install()
        assert second is False, (
            "install() must be idempotent — second call " "should be a no-op"
        )
    finally:
        # Best-effort restore; subsequent tests reset
        # `inst._INSTALLED` themselves.
        if first and Scheduler.__init__ is not original_init:
            Scheduler.__init__ = original_init
        inst._INSTALLED = False


def test_install_skips_when_sglang_unimportable(monkeypatch):
    """If `sglang.srt.managers.scheduler` can't be imported,
    install() must log + return False without raising. Used
    as a defense in mixed-engine deployments where SGLang may
    not be present."""
    monkeypatch.setenv("GMS_SGLANG_ENABLE_KV_RING", "1")
    import sys

    import gpu_memory_service.integrations.sglang.install_gms_radix_cache as inst

    inst._INSTALLED = False
    # Hide the scheduler module to simulate it being absent.
    saved = sys.modules.pop(
        "sglang.srt.managers.scheduler",
        None,
    )
    sys.modules["sglang.srt.managers.scheduler"] = None  # type: ignore
    try:
        assert inst.install() is False
    finally:
        if saved is not None:
            sys.modules["sglang.srt.managers.scheduler"] = saved
        else:
            sys.modules.pop(
                "sglang.srt.managers.scheduler",
                None,
            )
        inst._INSTALLED = False


def _build_cache_with_socket(connector, allocator, socket_path):
    """Variant of _build_cache that enables epoch polling."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        enable_kv_cache_events=False,
    )

    def layout(slot_idx):
        return [(0, int(slot_idx) * 64, 64)]

    Cls = make_gms_radix_cache_class()
    return Cls(
        params,
        gds=connector,
        block_layout_fn=layout,
        engine_id="test-eng",
        daemon_socket=socket_path,
    )


def test_evict_falls_back_to_vanilla_on_spill_failure():
    """If the connector reports failures from evict_blocks_to_
    storage, the leaf should follow the vanilla delete path —
    the tree must NOT retain a half-spilled node that can't be
    restored."""

    class _BrokenConnector(_FakeBlockConnector):
        def evict_blocks_to_storage(self, block_ids, generations=None):
            from gms_kv_ring.engines.gds_block_connector import EvictResult

            bids = list(block_ids)
            return EvictResult(
                succeeded=0,
                failed=len(bids),
                failed_ids=bids,
            )

    conn = _BrokenConnector(available=True)
    alloc = _FakeAllocator()
    cache = _build_cache(conn, alloc)

    _insert(cache, [1, 2, 3, 4])
    _evict(cache, num_tokens=4)

    # No spill recorded (the spill failed).
    assert len(cache._spilled) == 0
    # Match returns empty — node was vanilla-deleted.
    m = _match(cache, [1, 2, 3, 4])
    assert m.device_indices.numel() == 0
