# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
"""SGL-RACE3: per-slot generation tagging on spill (parity with vLLM).

Race #3 closure: a cross-step async restore in flight must NOT
read bytes that have been overwritten by a same-slot re-evict in a
later step. vLLM threads a per-slot generation counter through
`demote_hbm_to_storage` and refuses restores whose expected
generation doesn't match. SGLang now does the same.

What we validate:
  - On first spill of a slot, generation tag is 1 (monotonic from 0).
  - On second spill of the SAME slot, generation tag is 2 (bumped).
  - The connector's `evict_blocks_to_storage` receives the
    per-block generation map matching the slot indices.
  - `_SpillInfo.generations` records the bumped values (used by
    restore for daemon-side enforcement).
  - Warm-restart loads carry generations from the snapshot and
    seed `_slot_generations` so post-restart spills bump from the
    right base (not from 0)."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

sglang = pytest.importorskip(
    "sglang",
    reason="SGLang must be importable in the active Python environment",
)
torch = pytest.importorskip("torch")


from gpu_memory_service.integrations.sglang.gms_radix_cache import (
    make_gms_radix_cache_class,
)


class _TrackingBlockConnector:
    """Captures the `generations` map passed to evict so tests can
    assert per-slot bumping."""

    def __init__(self):
        self.evict_calls: list[tuple[list[int], dict[int, int]]] = []

    def is_available(self) -> bool:
        return True

    def evict_blocks_to_storage(self, block_ids, generations=None):
        from gms_kv_ring.engines.gds_block_connector import EvictResult

        bids = list(block_ids)
        gens = dict(generations) if generations is not None else None
        self.evict_calls.append((bids, gens))
        return EvictResult(succeeded=len(bids), failed=0, failed_ids=[])

    def restore_blocks_remap(self, triples):
        return {entry[1]: True for entry in triples}


class _FakeAllocator:
    def __init__(self):
        self.device = torch.device("cpu")
        self._next = 1_000_000

    def alloc(self, n):
        if n <= 0:
            return torch.empty((0,), dtype=torch.int64)
        out = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return out

    def free(self, indices):
        pass


def _build_cache(connector, page_size=4):
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=_FakeAllocator(),
        page_size=page_size,
        enable_kv_cache_events=False,
    )

    def layout(slot_idx):
        return [(0, int(slot_idx) * 64, 64)]

    Cls = make_gms_radix_cache_class()
    return Cls(
        params,
        gds=connector,
        block_layout_fn=layout,
        engine_id="sgl-race3-test",
    )


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

from sglang.srt.mem_cache.radix_cache import RadixKey


def _insert(cache, token_ids):
    value = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    key = RadixKey(token_ids=token_ids, extra_key=None)
    if _NEW_API:
        cache.insert(InsertParams(key=key, value=value, priority=0))
    else:
        cache.insert(key, value=value, priority=0)
    return value


def _evict(cache, n):
    if _NEW_API:
        return cache.evict(EvictParams(num_tokens=n))
    cache.evict(n)
    return None


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_first_spill_assigns_gen_1():
    """The very first spill of a slot tags it with generation 1
    (monotonic counter starts at 0; bumped once on first use)."""
    conn = _TrackingBlockConnector()
    cache = _build_cache(conn)
    val = _insert(cache, [10, 20, 30, 40])
    slots = [int(x) for x in val.tolist()]

    _evict(cache, 4)

    assert len(conn.evict_calls) == 1
    bids, gens = conn.evict_calls[0]
    assert sorted(bids) == sorted(slots)
    assert gens is not None
    for s in slots:
        assert gens[s] == 1, f"slot {s}: expected gen=1, got {gens[s]}"

    # _SpillInfo records the same generations.
    for info in cache._spilled.values():
        assert info.generations == [1] * info.value_len


def test_re_spill_same_slot_bumps_gen():
    """If a slot is spilled, then later spilled AGAIN (e.g., the
    engine reused it after restoring + re-evicting), the SECOND
    spill must carry generation 2 — daemon then refuses any in-flight
    restore from the first spill carrying expected_gen=1."""
    conn = _TrackingBlockConnector()
    cache = _build_cache(conn)

    # First spill: gens=1.
    val1 = _insert(cache, [10, 20, 30, 40])
    slots = [int(x) for x in val1.tolist()]
    _evict(cache, 4)
    assert conn.evict_calls[-1][1] == {s: 1 for s in slots}

    # Manually inject a second spill for the SAME slots. (In real
    # SGLang use, restore + re-eviction would naturally drive the
    # same indices through `_spill_node` again; we simulate by
    # directly invoking with a freshly-constructed node holding the
    # same slot indices.)
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

    node2 = TreeNode(priority=0)
    node2.key = RadixKey(token_ids=[10, 20, 30, 40], extra_key=None)
    node2.value = torch.tensor(slots, dtype=torch.int64)
    node2.parent = cache.root_node

    ok = cache._spill_node(node2)
    assert ok is True

    # Second evict call to the connector — generations now bumped.
    assert len(conn.evict_calls) == 2
    bids2, gens2 = conn.evict_calls[1]
    assert sorted(bids2) == sorted(slots)
    for s in slots:
        assert gens2[s] == 2, f"slot {s}: expected gen=2 on re-spill, got {gens2[s]}"


def test_disjoint_slots_have_independent_counters():
    """Each slot tracks its own generation. Spilling slot A then
    slot B then slot A again should produce gens 1, 1, 2 — NOT a
    shared monotonic counter."""
    conn = _TrackingBlockConnector()
    cache = _build_cache(conn)

    # Spill slots 1_000_000..1_000_003 (first insert).
    val1 = _insert(cache, [10, 20, 30, 40])
    slots_a = [int(x) for x in val1.tolist()]
    _evict(cache, 4)
    assert conn.evict_calls[-1][1] == {s: 1 for s in slots_a}

    # Spill slots 1_000_004..1_000_007 (second insert).
    val2 = _insert(cache, [50, 60, 70, 80])
    slots_b = [int(x) for x in val2.tolist()]
    _evict(cache, 4)
    assert conn.evict_calls[-1][1] == {s: 1 for s in slots_b}

    # Re-spill slot A.
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

    node_a2 = TreeNode(priority=0)
    node_a2.key = RadixKey(token_ids=[10, 20, 30, 40], extra_key=None)
    node_a2.value = torch.tensor(slots_a, dtype=torch.int64)
    node_a2.parent = cache.root_node
    cache._spill_node(node_a2)
    assert conn.evict_calls[-1][1] == {s: 2 for s in slots_a}


def test_warm_restart_seeds_generation_counter(monkeypatch, tmp_path):
    """After a warm restart, a fresh spill of a slot that was
    previously spilled at gen=K must produce gen=K+1, NOT gen=1.
    Otherwise stale daemon-side state at gen=K would be served on
    a future restore."""
    snap = str(tmp_path / "snapshot.bin")
    monkeypatch.setenv("GMS_SGLANG_PERSIST_PREFIX", "1")
    monkeypatch.setenv("GMS_SGLANG_PERSIST_PATH", snap)
    monkeypatch.setenv("GMS_SGLANG_SNAPSHOT_INTERVAL_S", "3600")

    # First run: spill twice to reach gen=2 on the slots.
    conn1 = _TrackingBlockConnector()
    cache1 = _build_cache(conn1)
    val = _insert(cache1, [10, 20, 30, 40])
    slots = [int(x) for x in val.tolist()]
    _evict(cache1, 4)
    # Re-spill same slots to bump.
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

    node2 = TreeNode(priority=0)
    node2.key = RadixKey(token_ids=[10, 20, 30, 40], extra_key=None)
    node2.value = torch.tensor(slots, dtype=torch.int64)
    node2.parent = cache1.root_node
    cache1._spill_node(node2)
    # _SpillInfo for the second spill should record gen=2.
    last_info = None
    for info in cache1._spilled.values():
        last_info = info
    assert last_info is not None
    assert last_info.generations == [2] * last_info.value_len
    cache1.shutdown()

    # Second run: warm-restart, then spill the same slots again.
    conn2 = _TrackingBlockConnector()
    cache2 = _build_cache(conn2)
    # _slot_generations must have been seeded from the snapshot.
    for s in slots:
        assert cache2._slot_generations.get(s, 0) >= 2, (
            f"slot {s}: post-restart counter not seeded; got "
            f"{cache2._slot_generations.get(s, 0)}"
        )

    # Now spill these slots again — gens must be 3 (bump from 2).
    node3 = TreeNode(priority=0)
    node3.key = RadixKey(token_ids=[10, 20, 30, 40], extra_key=None)
    node3.value = torch.tensor(slots, dtype=torch.int64)
    node3.parent = cache2.root_node
    cache2._spill_node(node3)
    bids, gens = conn2.evict_calls[-1]
    assert sorted(bids) == sorted(slots)
    for s in slots:
        assert gens[s] == 3, (
            f"post-restart re-spill of slot {s}: expected gen=3 "
            f"(snapshot had 2 + 1 bump), got {gens[s]}"
        )
    cache2.shutdown()


def test_spillinfo_generations_match_evict_gens():
    """Invariant: the generations recorded in `_SpillInfo` MUST be
    the same generations passed to `evict_blocks_to_storage`. Used
    by the restore path to send `expected_generation` to the
    daemon — a mismatch would silently break Race #3."""
    conn = _TrackingBlockConnector()
    cache = _build_cache(conn)
    _insert(cache, [10, 20, 30, 40])
    _evict(cache, 4)

    _, gens_map = conn.evict_calls[0]
    for info in cache._spilled.values():
        spilled_gens = {
            int(b): int(g) for b, g in zip(info.block_ids, info.generations)
        }
        assert (
            spilled_gens == gens_map
        ), "evicted generations diverged from spilled record"
