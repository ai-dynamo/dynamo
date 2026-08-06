# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefix-index snapshot/replay across a failover.

The interesting cases are the two directions of staleness. A snapshot is a picture of
the index at a moment; between that moment and the crash, blocks can be *added* to the
cache (harmless -- they are simply missing, so those prefixes MISS and recompute) or
*evicted and reused* (dangerous -- the snapshot still names them, and replaying that
entry is a HIT on memory that has since been overwritten). Deletions are therefore
streamed as they happen rather than waiting for the next snapshot, and most of what
follows checks that asymmetry holds.
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


# --------------------------------------------------------------------------- fakes


class FakeBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.block_hash = None
        self.block_hash_num_tokens = 0
        self.ref_cnt = 0


class FakeFreeQueue:
    """Only the two operations the requeue uses, plus visible order for assertions."""

    def __init__(self, blocks):
        self.order = list(blocks)

    def remove(self, block):
        self.order.remove(block)

    def append(self, block):
        self.order.append(block)


class FakeBlockPool:
    def __init__(self, n_blocks: int = 64):
        self.blocks = [FakeBlock(i) for i in range(n_blocks)]
        self.null_block = self.blocks[0]
        self.free_block_queue = FakeFreeQueue(self.blocks[1:])
        self.cached_block_hashes_by_block: dict[int, set] = {}

    def _insert_block_hash(self, key, block, num_tokens):
        block.block_hash = key
        block.block_hash_num_tokens = num_tokens

    # -- helpers the tests use to drive the pool ---------------------------------
    def cache(self, block_id: int, h: bytes, ntok: int = 16):
        blk = self.blocks[block_id]
        blk.block_hash = BlockHashWithGroupId(h)
        blk.block_hash_num_tokens = ntok
        return blk

    def evict(self, block_id: int):
        """Reclaim a cached block, as get_new_blocks does when it pops a cached one."""
        blk = self.blocks[block_id]
        hashes = [bytes(blk.block_hash)] if blk.block_hash is not None else []
        blk.block_hash = None
        blk.block_hash_num_tokens = 0
        kvi.append_deletions(hashes)


@pytest.fixture
def idx(tmp_path, monkeypatch):
    """Point the module at a scratch snapshot path and reset its module state."""
    path = str(tmp_path / "kv_index")
    monkeypatch.setenv("GMS_KV_INDEX_PATH", path)
    monkeypatch.setattr(kvi, "_log_path", None)
    monkeypatch.setattr(kvi, "_last_snapshot", 0.0)
    return path


def h(n: int) -> bytes:
    return bytes([n]) * 32


# --------------------------------------------------------------------------- tests


def test_snapshot_round_trips(idx):
    pool = FakeBlockPool()
    pool.cache(5, h(1), ntok=16)
    pool.cache(9, h(2), ntok=8)

    kvi.write_snapshot(pool, LAYOUT, idx)
    layout_id, records = kvi.read_snapshot(idx)

    assert layout_id == LAYOUT
    assert sorted(records) == sorted([(h(1), 5, 16), (h(2), 9, 8)])


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
    # The snapshot describes pages this engine did not adopt; replaying it would point
    # hashes at memory that never held those prefixes.
    assert kvi.replay_index(dst, OTHER_LAYOUT) == 0
    assert dst.blocks[5].block_hash is None


def test_block_evicted_after_the_snapshot_is_not_replayed(idx):
    """The dangerous direction of staleness: the snapshot still names a reused block."""
    src = FakeBlockPool()
    src.cache(5, h(1))
    src.cache(9, h(2))
    kvi.write_snapshot(src, LAYOUT, idx)

    src.evict(5)  # reclaimed and overwritten with something else, then we crash

    dst = FakeBlockPool()
    installed = kvi.replay_index(dst, LAYOUT)

    assert installed == 1
    assert dst.blocks[5].block_hash is None, "evicted block must not be resurrected"
    assert bytes(dst.blocks[9].block_hash) == h(2)


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


def test_a_later_snapshot_supersedes_earlier_deletions(idx):
    """Deletions already reflected in a snapshot are dropped, not applied forever."""
    src = FakeBlockPool()
    src.cache(5, h(1))
    src.cache(9, h(2))
    kvi.write_snapshot(src, LAYOUT, idx)
    src.evict(5)

    # Second snapshot: block 5 has no hash, so its absence is already recorded. The
    # deletion list is cleared, and block 5 being cached again afterwards must survive.
    kvi.maybe_snapshot(_FakeScheduler(src, LAYOUT))
    src.cache(5, h(3))
    kvi.maybe_snapshot(_FakeScheduler(src, LAYOUT, force=True))

    dst = FakeBlockPool()
    kvi.replay_index(dst, LAYOUT)
    assert bytes(dst.blocks[5].block_hash) == h(3)


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


def test_a_torn_deletion_tail_does_not_lose_the_rest(idx):
    src = FakeBlockPool()
    src.cache(5, h(1))
    src.cache(9, h(2))
    kvi.write_snapshot(src, LAYOUT, idx)
    src.evict(5)

    with open(kvi._del_path(), "ab") as f:  # crash mid-append
        f.write(b"\x20\x00partial")

    dst = FakeBlockPool()
    kvi.replay_index(dst, LAYOUT)
    assert dst.blocks[5].block_hash is None, "the complete deletion still applies"
    assert bytes(dst.blocks[9].block_hash) == h(2)


def test_disabled_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_KV_INDEX_PATH", raising=False)
    monkeypatch.setattr(kvi, "_log_path", None)
    assert not kvi.is_enabled()
    kvi.enable_kv_index_persistence()  # must be a no-op, not an error


def test_missing_snapshot_is_not_an_error(idx):
    assert kvi.replay_index(FakeBlockPool(), LAYOUT) == 0


class _FakeScheduler:
    def __init__(self, pool, layout_id, force: bool = False):
        self.kv_cache_manager = type("M", (), {"block_pool": pool})()
        self._gms_kv_layout_id = layout_id
        if force:
            kvi._last_snapshot = 0.0
