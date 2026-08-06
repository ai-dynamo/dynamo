# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the failover prefix-index log: step barrier, tombstones, replay.

These cover the log's semantics without a GPU or a real ``BlockPool``:

* ADD records are staged at schedule time and only published at the step barrier, so a
  crash can never leave the log claiming bytes the forward pass never wrote.
* Eviction tombstones retire a mapping, so replay cannot resurrect a block the primary
  already reclaimed for other content.
* Replay applies events in order, which is what makes cache/evict/re-cache cycles
  reconstruct the primary's final index rather than a union of everything ever cached.
"""

from __future__ import annotations

import importlib

import pytest

kvi = importlib.import_module("gpu_memory_service.integrations.vllm.kv_index_persist")

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _FakeBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.block_hash = None
        self.block_hash_num_tokens = None
        self.ref_cnt = 0

    def set_block_hash(self, block_hash, num_tokens=None):
        self.block_hash = block_hash
        self.block_hash_num_tokens = num_tokens

    def reset_hash(self):
        self.block_hash = None


class _FakeQueue:
    """Just enough FreeKVCacheBlockQueue to observe ordering."""

    def __init__(self, blocks):
        self._q = list(blocks)

    def remove(self, block):
        self._q.remove(block)

    def append(self, block):
        self._q.append(block)

    def ids(self):
        return [b.block_id for b in self._q]


class _FakePool:
    def __init__(self, num_blocks: int = 16):
        self.blocks = [_FakeBlock(i) for i in range(num_blocks)]
        self.null_block = self.blocks[0]
        self.free_block_queue = _FakeQueue(self.blocks[1:])
        self.cached_block_hash_to_block = {}
        self.cached_block_hashes_by_block = {}

    def _insert_block_hash(self, block_hash, block, num_tokens):
        if block.block_hash is None:
            block.set_block_hash(block_hash, num_tokens)
        else:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(
                block_hash
            )
        self.cached_block_hash_to_block[block_hash] = block

    def installed(self):
        """hash bytes -> block_id, for whatever replay put in the index."""
        return {
            bytes(h): b.block_id for h, b in self.cached_block_hash_to_block.items()
        }


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "kv_index.log"
    monkeypatch.setattr(kvi, "_log_path", str(path))
    monkeypatch.setattr(kvi, "_staged", [], raising=False)
    return str(path)


def _add(h: bytes, block_id: int, num_tokens: int = 16):
    return (kvi._OP_ADD, h, block_id, num_tokens)


def _delete(h: bytes, block_id: int):
    return (kvi._OP_DEL, h, block_id, None)


def test_staged_adds_are_not_published_until_the_step_barrier(log_path):
    """A crash between schedule and forward must not leave a record on the log."""
    kvi._stage_records([_add(b"h1", 1)])

    import os

    assert not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    kvi.flush_staged_records()

    pool = _FakePool()
    kvi.rehydrate_block_pool(pool)
    assert pool.installed() == {b"h1": 1}


def test_flush_is_idempotent_when_nothing_is_staged(log_path):
    kvi.flush_staged_records()
    kvi._stage_records([_add(b"h1", 1)])
    kvi.flush_staged_records()
    kvi.flush_staged_records()

    pool = _FakePool()
    kvi.rehydrate_block_pool(pool)
    assert pool.installed() == {b"h1": 1}


def test_eviction_tombstone_prevents_resurrecting_a_reclaimed_block(log_path):
    """Without the DEL, replay would hand a standby a HIT on overwritten bytes."""
    kvi._append_records([_add(b"h1", 1), _add(b"h2", 2)])
    kvi._append_records([_delete(b"h1", 1)])  # block 1 reclaimed for other content

    pool = _FakePool()
    kvi.rehydrate_block_pool(pool)

    assert pool.installed() == {b"h2": 2}
    assert pool.blocks[1].block_hash is None


def test_replay_applies_events_in_order_for_cache_evict_recache(log_path):
    """A hash re-cached on a different block must land on the NEW block."""
    kvi._append_records([_add(b"h1", 1)])
    kvi._append_records([_delete(b"h1", 1)])
    kvi._append_records([_add(b"h1", 5)])

    pool = _FakePool()
    kvi.rehydrate_block_pool(pool)

    assert pool.installed() == {b"h1": 5}


def test_stale_tombstone_does_not_retire_a_newer_mapping(log_path):
    """A DEL naming an older block must not cancel a later ADD of the same hash."""
    kvi._append_records([_add(b"h1", 1)])
    kvi._append_records([_add(b"h1", 5)])  # re-cached elsewhere
    kvi._append_records([_delete(b"h1", 1)])  # old block finally evicted

    pool = _FakePool()
    kvi.rehydrate_block_pool(pool)

    assert pool.installed() == {b"h1": 5}


def test_truncated_tail_record_is_dropped(log_path):
    kvi._append_records([_add(b"h1", 1), _add(b"h2", 2)])
    with open(log_path, "ab") as f:
        f.write(b"\x40\x00\x00\x00partial")  # length prefix promising more than exists

    pool = _FakePool()
    kvi.rehydrate_block_pool(pool)

    assert pool.installed() == {b"h1": 1, b"h2": 2}


def test_rehydrated_blocks_move_to_the_free_queue_tail(log_path):
    """Cached-but-free blocks must be handed out LAST, or the next request clobbers them."""
    kvi._append_records([_add(b"h1", 1), _add(b"h2", 2), _add(b"h3", 3)])

    pool = _FakePool(num_blocks=8)
    assert pool.free_block_queue.ids() == [1, 2, 3, 4, 5, 6, 7]

    kvi.rehydrate_block_pool(pool)

    ids = pool.free_block_queue.ids()
    assert ids[:4] == [4, 5, 6, 7], f"uncached blocks must come first, got {ids}"
    # Descending block_id at the tail: the prefix's LAST blocks evict first, which
    # leaves a usable leading prefix.
    assert ids[4:] == [3, 2, 1], f"expected descending rehydrated tail, got {ids}"


def test_replay_never_disturbs_a_block_the_standby_is_already_using(log_path):
    kvi._append_records([_add(b"h1", 1), _add(b"h2", 2)])

    pool = _FakePool()
    pool.blocks[1].ref_cnt = 1  # in flight on the standby

    kvi.rehydrate_block_pool(pool)

    assert pool.installed() == {b"h2": 2}
    assert pool.blocks[1].block_hash is None


def test_absent_log_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(kvi, "_log_path", str(tmp_path / "missing.log"))
    pool = _FakePool()
    kvi.rehydrate_block_pool(pool)
    assert pool.installed() == {}
