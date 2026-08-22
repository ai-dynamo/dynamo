# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The store tracks vLLM's prefix-cache index exactly.

These drive a *real* ``vllm.v1.core.block_pool.BlockPool`` -- it is constructible
with three ints and no GPU, no config, and no engine -- so they test the actual
code paths rather than a hand-rolled fake. No GPU required.

The load-bearing claim under test is completeness: vLLM changes a block's index
entry at exactly three sites, so overriding those three mirrors every change.
"""

from __future__ import annotations

import collections
import hashlib
import random

import pytest

pytest.importorskip("vllm")

from gpu_memory_service.integrations.vllm.kv_cache import block_pool  # noqa: E402
from gpu_memory_service.kv_cache import MmapBlockIndexStore, Refusal  # noqa: E402
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]

N_BLOCKS = 32
HASH_BLOCK_SIZE = 16
IDENTITY = hashlib.sha256(b"test-ruler").digest()


def key_for(tag: str, group: int = 0):
    return make_block_hash_with_group_id(
        BlockHash(hashlib.sha256(tag.encode()).digest()), group
    )


def indexed_blocks(pool) -> set[int]:
    return {b.block_id for b in pool.blocks if b.block_hash is not None}


def free_queue_order(pool) -> list[int]:
    order, node = [], pool.free_block_queue.fake_free_list_head.next_free_block
    tail = pool.free_block_queue.fake_free_list_tail
    while node is not None and node is not tail:
        order.append(node.block_id)
        node = node.next_free_block
    return order


@pytest.fixture
def rig(tmp_path):
    """A real BlockPool with the store attached."""
    pool = BlockPool(
        num_gpu_blocks=N_BLOCKS, enable_caching=True, hash_block_size=HASH_BLOCK_SIZE
    )
    store = MmapBlockIndexStore.create(
        str(tmp_path / "index.store"), N_BLOCKS, IDENTITY
    )
    block_pool.attach(pool, store)
    return pool, store


# ---------------------------------------------------------------------------
# the invariant
# ---------------------------------------------------------------------------


def test_store_tracks_the_index_across_a_scripted_trace(rig):
    """The core claim: after every operation the store equals vLLM's index.

    Drives the real allocate / label / free / touch / connector-evict / reset
    paths. Equality (not just subset) is assertable because this trace stays in
    scope: one KV cache group, a distinct key per block.
    """
    pool, store = rig
    rng = random.Random(1234)
    held: list = []
    seen = collections.Counter()
    counter = 0
    live_before: set[int] = set()

    ops = (
        ["alloc"] * 5
        + ["label"] * 7
        + ["free"] * 4
        + ["touch"]
        + ["evict"] * 2
        + ["reset"]
    )

    for step in range(1500):
        op = rng.choice(ops)

        if op == "alloc" and pool.free_block_queue.num_free_blocks > 4:
            store.on_schedule()
            got = pool.get_new_blocks(rng.randint(1, 4))
            # get_new_blocks retracts a label whenever it recycles a cached block
            seen["recycled"] += sum(1 for b in got if b.block_id in live_before)
            held.extend(got)
            seen["alloc"] += 1
        elif op == "label":
            candidates = [b for b in held if b.block_hash is None and not b.is_null]
            if candidates:
                counter += 1
                pool._insert_block_hash(
                    key_for(f"k{counter}"),
                    rng.choice(candidates),
                    HASH_BLOCK_SIZE * (counter % 7 + 1),
                )
                seen["label"] += 1
            store.on_complete()
        elif op == "free" and held:
            n = rng.randint(1, len(held))
            pool.free_blocks(held[:n])
            held = held[n:]
            seen["free"] += 1
        elif op == "touch":
            block = rng.choice(pool.blocks[1:])
            if block.ref_cnt == 0 and not block.is_null:
                pool.touch([block])
                held.append(block)
                seen["touch"] += 1
        elif op == "evict":
            victims = sorted(
                b.block_id for b in pool.blocks[1:] if b.block_hash is not None
            )
            if victims:
                pool.evict_blocks({rng.choice(victims)})
                seen["evict_blocks"] += 1
        elif op == "reset":
            seen["reset_ok" if pool.reset_prefix_cache() else "reset_refused"] += 1

        live_before = store.live()
        assert live_before == indexed_blocks(pool), (
            f"divergence at step {step} after {op}: "
            f"store={sorted(live_before)} vllm={sorted(indexed_blocks(pool))}"
        )

    # The trace is only evidence for paths it actually took.
    print(f"\ntrace coverage: {dict(sorted(seen.items()))}")
    for path in ("label", "recycled", "evict_blocks", "reset_ok", "reset_refused"):
        assert seen[path] >= 5, f"trace under-exercised {path}: {dict(seen)}"


def test_connector_eviction_path_retracts(rig):
    """``evict_blocks`` is the connector's 'these bytes went bad' channel.

    A design that watched block *handout* instead of label changes would miss
    this path entirely -- no block leaves the free queue here.
    """
    pool, store = rig
    store.on_schedule()
    (block,) = pool.get_new_blocks(1)
    pool._insert_block_hash(key_for("a"), block, HASH_BLOCK_SIZE)
    store.on_complete()
    assert store.live() == {block.block_id}

    pool.evict_blocks({block.block_id})
    assert store.live() == set()
    assert block.block_hash is None


# ---------------------------------------------------------------------------
# the batch counters
# ---------------------------------------------------------------------------


def test_counters_withhold_entries_until_their_batch_completes(rig):
    """vLLM labels blocks during schedule(), before the KV exists."""
    pool, store = rig
    store.on_schedule()  # S=1
    (block,) = pool.get_new_blocks(1)
    pool._insert_block_hash(key_for("a"), block, HASH_BLOCK_SIZE)

    assert store.live() == {block.block_id}
    assert list(store.usable()) == [], "entry usable before its batch completed"

    store.on_complete()  # E=1
    assert [b for b, _, _ in store.usable()] == [block.block_id]


def test_entries_from_an_unreaped_batch_are_never_usable(rig):
    """An engine killed mid-batch must not publish that batch's labels."""
    pool, store = rig
    store.on_schedule()
    (early,) = pool.get_new_blocks(1)
    pool._insert_block_hash(key_for("early"), early, HASH_BLOCK_SIZE)
    store.on_complete()

    store.on_schedule()  # this batch never completes -- SIGKILL here
    (late,) = pool.get_new_blocks(1)
    pool._insert_block_hash(key_for("late"), late, HASH_BLOCK_SIZE)

    trusted = {b for b, _, _ in store.usable()}
    assert trusted == {early.block_id}
    assert late.block_id not in trusted


# ---------------------------------------------------------------------------
# reset_prefix_cache
# ---------------------------------------------------------------------------


def test_failed_reset_does_not_clear_the_store(rig):
    """The reset refuses while any block is held, and drops no label."""
    pool, store = rig
    store.on_schedule()
    blocks = pool.get_new_blocks(2)
    pool._insert_block_hash(key_for("a"), blocks[0], HASH_BLOCK_SIZE)
    store.on_complete()

    assert pool.reset_prefix_cache() is False  # blocks[1] still held
    assert store.live() == {blocks[0].block_id}


def test_successful_reset_clears_the_store(rig):
    pool, store = rig
    store.on_schedule()
    blocks = pool.get_new_blocks(2)
    pool._insert_block_hash(key_for("a"), blocks[0], HASH_BLOCK_SIZE)
    store.on_complete()
    pool.free_blocks(blocks)

    assert pool.reset_prefix_cache() is True
    assert store.live() == set()


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def _populate(pool, n=5):
    pool._dyn_store.on_schedule()
    blocks = pool.get_new_blocks(n)
    for i, block in enumerate(blocks):
        pool._insert_block_hash(key_for(f"k{i}"), block, HASH_BLOCK_SIZE * (i + 1))
    pool._dyn_store.on_complete()
    pool.free_blocks(blocks)
    return blocks


def test_restore_installs_the_predecessors_entries(rig, tmp_path):
    pool, store = rig
    blocks = _populate(pool)
    expected = {b.block_id: b.block_hash for b in blocks}

    successor = BlockPool(N_BLOCKS, True, HASH_BLOCK_SIZE)
    reopened, reason = MmapBlockIndexStore.open(
        store.path, identity=IDENTITY, num_blocks=N_BLOCKS
    )
    assert reason is Refusal.OK
    installed = block_pool.restore(successor, reopened)

    assert {b.block_id for b in installed} == set(expected)
    for block_id, key in expected.items():
        assert successor.blocks[block_id].block_hash == key
        assert successor.cached_block_hash_to_block.get_one_block(key) is (
            successor.blocks[block_id]
        )


def test_restored_blocks_move_to_the_free_queue_tail(rig):
    """Without this the first request served overwrites the restored prefix."""
    pool, store = rig
    blocks = _populate(pool, n=5)
    ids = {b.block_id for b in blocks}

    successor = BlockPool(N_BLOCKS, True, HASH_BLOCK_SIZE)
    reopened, _ = MmapBlockIndexStore.open(
        store.path, identity=IDENTITY, num_blocks=N_BLOCKS
    )
    assert len(block_pool.restore(successor, reopened)) == len(ids)

    order = free_queue_order(successor)
    assert set(order[-len(ids) :]) == ids, "restored blocks are not at the tail"
    # deepest match nearest the head, so it is evicted first and the shallow
    # leading prefix -- reusable by more requests -- survives longest
    depths = [successor.blocks[b].block_hash_num_tokens for b in order[-len(ids) :]]
    assert depths == sorted(depths, reverse=True)
    # and the next block handed out is NOT one of the restored ones
    assert successor.free_block_queue.popleft().block_id not in ids


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda kw: kw.update(identity=hashlib.sha256(b"other").digest()),
            Refusal.IDENTITY,
        ),
        (lambda kw: kw.update(num_blocks=N_BLOCKS + 1), Refusal.NUM_BLOCKS),
    ],
)
def test_open_refuses_on_each_header_gate(rig, mutate, expected):
    pool, store = rig
    _populate(pool)
    kwargs = dict(identity=IDENTITY, num_blocks=N_BLOCKS)
    mutate(kwargs)

    reopened, reason = MmapBlockIndexStore.open(store.path, **kwargs)
    assert reopened is None
    assert reason is expected


def test_missing_store_is_not_an_error(tmp_path):
    reopened, reason = MmapBlockIndexStore.open(
        str(tmp_path / "nope.store"),
        identity=IDENTITY,
        num_blocks=N_BLOCKS,
    )
    assert reopened is None and reason is Refusal.ABSENT


def test_truncated_store_is_refused(rig):
    pool, store = rig
    _populate(pool)
    with open(store.path, "r+b") as f:
        f.truncate(4096 + 64 * 4)
    reopened, reason = MmapBlockIndexStore.open(
        store.path, identity=IDENTITY, num_blocks=N_BLOCKS
    )
    assert reopened is None and reason is Refusal.TRUNCATED


# ---------------------------------------------------------------------------
# differential: replay must reproduce what the native install produces
# ---------------------------------------------------------------------------


def test_restore_reproduces_the_native_install(rig):
    """Fails loudly on any vLLM change to ``_insert_block_hash``'s effects.

    Builds the same index two ways -- through vLLM's own path, and through
    replay -- and compares the resulting structures field by field.
    """
    pool, store = rig
    native = _populate(pool, n=6)

    successor = BlockPool(N_BLOCKS, True, HASH_BLOCK_SIZE)
    reopened, _ = MmapBlockIndexStore.open(
        store.path, identity=IDENTITY, num_blocks=N_BLOCKS
    )
    block_pool.restore(successor, reopened)

    for block in native:
        mine = successor.blocks[block.block_id]
        assert mine.block_hash == block.block_hash
        assert mine.block_hash_num_tokens == block.block_hash_num_tokens
        assert mine.ref_cnt == block.ref_cnt == 0

    assert set(pool.cached_block_hash_to_block._cache) == set(
        successor.cached_block_hash_to_block._cache
    )
    # single group => the multi-name side table stays empty on both sides
    assert pool.cached_block_hashes_by_block == {}
    assert successor.cached_block_hashes_by_block == {}


# ---------------------------------------------------------------------------
# durability of a single record
# ---------------------------------------------------------------------------


def test_torn_record_reads_back_as_unusable(rig):
    """The stamp is written last, so a half-written record is simply absent."""
    pool, store = rig
    store.on_schedule()
    (block,) = pool.get_new_blocks(1)

    # Simulate a crash between the key write and the stamp write.
    store._rec["key"][block.block_id][:36] = 7
    store._rec["key_len"][block.block_id] = 36
    store._rec["num_tokens"][block.block_id] = HASH_BLOCK_SIZE
    store.on_complete()

    assert block.block_id not in {b for b, _, _ in store.usable()}
