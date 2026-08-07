# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefix-index snapshot/replay across a failover.

A snapshot is a picture of the index at a moment, and between that moment and the crash
the world moves. Blocks *cached* since then are merely absent, which costs a MISS. Blocks
*reused* since then are still named by the snapshot, and replaying one is a HIT on memory
that now belongs to someone else. A per-block generation counter, bumped whenever a block
leaves the free queue, is what separates the two: the snapshot proposes a binding and the
counter vetoes it. Most of what follows checks that the veto is in the right places, and
that losing it installs nothing rather than everything.
"""

from __future__ import annotations

import pytest
from _deps import HAS_GMS

if not HAS_GMS:  # pragma: no cover
    pytest.skip("gpu_memory_service not importable", allow_module_level=True)

vllm = pytest.importorskip("vllm")
import gpu_memory_service.integrations.vllm.kv_index_persist as kvi  # noqa: E402
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId  # noqa: E402

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

LAYOUT = "layout-aaaa|layout-bbbb"
OTHER_LAYOUT = "layout-cccc|layout-dddd"
N_BLOCKS = 64


# --------------------------------------------------------------------------- fakes


class FakeBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.block_hash = None
        self.block_hash_num_tokens = 0
        self.ref_cnt = 0


class FakeFreeQueue:
    """Only the operations replay uses, plus visible order for assertions."""

    def __init__(self, blocks):
        self.order = list(blocks)

    def remove(self, block):
        self.order.remove(block)

    def append(self, block):
        self.order.append(block)


class FakeBlockPool:
    def __init__(self, n_blocks: int = N_BLOCKS):
        self.blocks = [FakeBlock(i) for i in range(n_blocks)]
        self.null_block = self.blocks[0]
        self.free_block_queue = FakeFreeQueue(self.blocks[1:])
        self.cached_block_hashes_by_block: dict[int, set] = {}

    def _insert_block_hash(self, key, block, num_tokens):
        block.block_hash = key
        block.block_hash_num_tokens = num_tokens

    # -- helpers the tests use to drive the pool ---------------------------------
    def cache(self, block_id: int, h: bytes, ntok: int = 16):
        """Hash a block that is already held, as cache_full_blocks does."""
        blk = self.blocks[block_id]
        blk.block_hash = BlockHashWithGroupId(h)
        blk.block_hash_num_tokens = ntok
        return blk

    def evict(self, block_id: int):
        """Drop a block's hash. Removes a name; touches no bytes."""
        blk = self.blocks[block_id]
        blk.block_hash = None
        blk.block_hash_num_tokens = 0

    def hand_out(self, block_id: int):
        """Pop a block off the free queue, which is what makes it writable again."""
        kvi.bump([block_id])


@pytest.fixture
def idx(tmp_path, monkeypatch):
    """Point the module at a scratch path and reset its module state."""
    path = str(tmp_path / "kv_index")
    monkeypatch.setenv("GMS_KV_INDEX_PATH", path)
    monkeypatch.setattr(kvi, "_log_path", None)
    monkeypatch.setattr(kvi, "_last_snapshot", 0.0)
    monkeypatch.setattr(kvi, "_gen", None)
    monkeypatch.setattr(kvi, "_gen_layout", None)
    monkeypatch.setattr(kvi, "_gen_blind", False)
    kvi.open_generations(N_BLOCKS, LAYOUT)
    return path


def h(n: int) -> bytes:
    return bytes([n]) * 32


# --------------------------------------------------------------------------- tests


def test_snapshot_round_trips(idx):
    pool = FakeBlockPool()
    pool.hand_out(5)
    pool.cache(5, h(1), ntok=16)
    pool.cache(9, h(2), ntok=8)

    kvi.write_snapshot(pool, LAYOUT, idx)
    layout_id, records = kvi.read_snapshot(idx)

    assert layout_id == LAYOUT
    # Block 5 was handed out once, block 9 never; the generation rides along.
    assert sorted(records) == sorted([(h(1), 5, 16, 1), (h(2), 9, 8, 0)])


def test_replay_installs_the_prior_index(idx):
    src = FakeBlockPool()
    src.cache(5, h(1))
    src.cache(9, h(2))
    kvi.write_snapshot(src, LAYOUT, idx)

    dst = FakeBlockPool()
    assert kvi.replay_index(dst, LAYOUT) == 2
    assert bytes(dst.blocks[5].block_hash) == h(1)
    assert bytes(dst.blocks[9].block_hash) == h(2)


def test_replay_refuses_a_snapshot_from_a_different_layout(idx):
    src = FakeBlockPool()
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    dst = FakeBlockPool()
    # The snapshot describes pages this engine did not adopt. block_id indexes tensors of
    # a particular shape, so the same number would name different bytes here.
    assert kvi.replay_index(dst, OTHER_LAYOUT) == 0
    assert dst.blocks[5].block_hash is None


def test_block_reused_after_the_snapshot_is_not_replayed(idx):
    """The dangerous direction of staleness: the snapshot still names a reused block."""
    src = FakeBlockPool()
    src.cache(5, h(1))
    src.cache(9, h(2))
    kvi.write_snapshot(src, LAYOUT, idx)

    src.evict(5)
    src.hand_out(5)  # reclaimed and refilled with something else, then we crash

    dst = FakeBlockPool()
    installed = kvi.replay_index(dst, LAYOUT)

    assert installed == 1
    assert dst.blocks[5].block_hash is None, "reused block must not be resurrected"
    assert bytes(dst.blocks[9].block_hash) == h(2)


def test_block_evicted_but_not_reused_is_still_replayed(idx):
    """Eviction removes a name and touches no bytes, so the binding still holds.

    This is what the counters buy over recording evictions: the block still contains the
    prefix, so replaying it is correct and free.
    """
    src = FakeBlockPool()
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    src.evict(5)  # dropped from the index, never handed out again

    dst = FakeBlockPool()
    assert kvi.replay_index(dst, LAYOUT) == 1
    assert bytes(dst.blocks[5].block_hash) == h(1)


def test_block_cached_after_the_snapshot_is_simply_missing(idx):
    """The harmless direction: a late addition costs a MISS, never a wrong hit."""
    src = FakeBlockPool()
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    src.cache(9, h(2))  # cached after the snapshot, then we crash

    dst = FakeBlockPool()
    assert kvi.replay_index(dst, LAYOUT) == 1
    assert bytes(dst.blocks[5].block_hash) == h(1)
    assert dst.blocks[9].block_hash is None


def test_a_later_snapshot_records_the_current_generation(idx):
    """Reuse before a snapshot is absorbed by it, not carried forever."""
    src = FakeBlockPool()
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    src.evict(5)
    src.hand_out(5)
    src.cache(5, h(3))  # same block, different prefix
    kvi.write_snapshot(src, LAYOUT, idx)

    dst = FakeBlockPool()
    assert kvi.replay_index(dst, LAYOUT) == 1
    assert bytes(dst.blocks[5].block_hash) == h(3)


def test_resetting_the_prefix_cache_invalidates_every_record(idx):
    """reset_prefix_cache drops the index without moving bytes, so counters must move.

    Its purpose is to discard a cache computed under stale weights. Every counter would
    otherwise still match and the snapshot would reinstate exactly what was discarded.
    """
    src = FakeBlockPool()
    src.cache(5, h(1))
    src.cache(9, h(2))
    kvi.write_snapshot(src, LAYOUT, idx)

    kvi.bump_all()

    dst = FakeBlockPool()
    assert kvi.replay_index(dst, LAYOUT) == 0


def test_replay_never_disturbs_a_block_this_engine_is_using(idx):
    src = FakeBlockPool()
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    dst = FakeBlockPool()
    dst.blocks[5].ref_cnt = 1  # in flight on the standby already
    assert kvi.replay_index(dst, LAYOUT) == 0
    assert dst.blocks[5].block_hash is None


def test_replayed_blocks_move_to_the_free_queue_tail(idx):
    """vLLM hands out the head of the free queue, so reused blocks must not sit there."""
    src = FakeBlockPool()
    for i in (5, 6, 7):
        src.cache(i, h(i))
    kvi.write_snapshot(src, LAYOUT, idx)

    dst = FakeBlockPool()
    kvi.replay_index(dst, LAYOUT)

    order = [b.block_id for b in dst.free_block_queue.order]
    assert order[0] not in (5, 6, 7), "reused blocks must not be handed out first"
    # Descending id, so the prefix's last blocks are evicted first and a usable
    # leading prefix survives.
    assert order[-3:] == [7, 6, 5]


def test_losing_the_counters_installs_nothing(idx, monkeypatch):
    """Fail closed. Without the veto there is no evidence any record still holds."""
    src = FakeBlockPool()
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    monkeypatch.setattr(kvi, "_gen", None)
    assert kvi.replay_index(FakeBlockPool(), LAYOUT) == 0


def test_a_handout_the_counters_missed_installs_nothing(idx, monkeypatch):
    """A block handed out while the array was unavailable makes every record suspect."""
    src = FakeBlockPool()
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    monkeypatch.setattr(kvi, "_gen", None)
    kvi.bump([5])  # missed
    kvi.open_generations(N_BLOCKS, LAYOUT)
    assert kvi._gen_blind is False, "reopening clears the flag"

    monkeypatch.setattr(kvi, "_gen_blind", True)
    assert kvi.replay_index(FakeBlockPool(), LAYOUT) == 0


def test_counters_from_another_layout_are_reset_not_reinterpreted(idx):
    """Values only mean something within one lineage of the same pages."""
    src = FakeBlockPool()
    src.hand_out(5)
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    kvi.open_generations(N_BLOCKS, OTHER_LAYOUT)
    assert int(kvi._gen[5]) == 0, "counters must not carry across layouts"
    assert kvi.replay_index(FakeBlockPool(), OTHER_LAYOUT) == 0


def test_counters_survive_a_takeover(idx):
    """The array is adopted with the pages, not reset with the process."""
    src = FakeBlockPool()
    src.hand_out(5)
    src.cache(5, h(1))
    kvi.write_snapshot(src, LAYOUT, idx)

    # Standby: fresh module state, same files.
    kvi._gen = None
    kvi._gen_layout = None
    kvi.open_generations(N_BLOCKS)
    kvi.bind_generations(LAYOUT)

    assert int(kvi._gen[5]) == 1
    assert kvi.replay_index(FakeBlockPool(), LAYOUT) == 1


def test_disabled_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_KV_INDEX_PATH", raising=False)
    monkeypatch.setattr(kvi, "_log_path", None)
    assert not kvi.is_enabled()
    kvi.enable_kv_index_persistence()  # must be a no-op, not an error


def test_missing_snapshot_is_not_an_error(idx):
    assert kvi.replay_index(FakeBlockPool(), LAYOUT) == 0
