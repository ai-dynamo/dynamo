# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keeping the store in step with vLLM's prefix cache, and putting it back.

vLLM changes a block's index entry at exactly three sites, so overriding those
three on the pool instance tracks every change with no scanning. Record *after*
vLLM, drop *before* it -- an interrupted step then leaves the store claiming
less than vLLM does, never more.

    attach(pool, store, skip_reset)   mirror vLLM's changes into the store
    restore(pool, store)              install a predecessor's entries
"""

from __future__ import annotations

import logging
from typing import Callable

from gpu_memory_service.kv_cache.interface import BlockIndexStore

logger = logging.getLogger(__name__)


def attach(
    pool, store: BlockIndexStore, skip_reset: Callable[[], bool] = lambda: False
) -> None:
    """Mirror every index change on this ``BlockPool`` instance into ``store``.

    A runtime subclass swap rather than a patch on the class: it composes with
    whatever ``BlockPool`` subclass is already in play and is scoped to the one
    object we were handed. ``BlockPool`` has no injection point -- one hardcoded
    construction site, no factory, no config field -- so this is the least
    invasive option available.

    Re-attaching on a later wake only re-points the store.

    ``skip_reset`` answers "ignore this one": vLLM drops its whole index when an
    engine sleeps, on the assumption the KV is going away. Under a committed GMS
    layout it is not, so those entries are still true. Outside that case a reset
    means the operator dropped the cache on purpose and the entries are dead.
    """
    pool._dyn_store = store
    pool._dyn_skip_reset = skip_reset
    if getattr(pool, "_dyn_mirrored", False):
        return
    base = type(pool)

    def _insert_block_hash(self, key, block, num_tokens):
        # vLLM first: recording after means a crash mid-way simply never
        # recorded the entry.
        base._insert_block_hash(self, key, block, num_tokens)
        actual = block.block_hash
        if actual is None or actual != key:
            # An early return declined the insert, or this block already carried
            # a different primary name (out of scope: >1 KV cache group). Never
            # guess -- drop it and stay a subset of what vLLM believes.
            self._dyn_store.drop(block.block_id)
            return
        self._dyn_store.record(
            block.block_id, bytes(actual), block.block_hash_num_tokens
        )

    def _remove_cached_block_hashes(self, block):
        # Drop first: a crash mid-way leaves us with fewer claims, not more.
        self._dyn_store.drop(block.block_id)
        return base._remove_cached_block_hashes(self, block)

    def reset_prefix_cache(self):
        # Only react when vLLM says it actually happened -- the reset refuses
        # while any block is in use, and mirroring a refusal would empty the
        # store on every poll of a busy engine.
        ok = base.reset_prefix_cache(self)
        if ok and not self._dyn_skip_reset():
            self._dyn_store.drop_all()
        return ok

    pool.__class__ = type(
        f"Mirrored{base.__name__}",
        (base,),
        {
            "_insert_block_hash": _insert_block_hash,
            "_remove_cached_block_hashes": _remove_cached_block_hashes,
            "reset_prefix_cache": reset_prefix_cache,
        },
    )
    pool._dyn_mirrored = True


def restore(pool, store: BlockIndexStore) -> list:
    """Install a predecessor's entries into the pool this engine just adopted."""
    from vllm.v1.core.kv_cache_utils import get_group_id

    installed = []
    for block_id, block_hash, num_tokens in store.usable():
        if block_id <= 0 or block_id >= len(pool.blocks):
            continue
        if get_group_id(block_hash) != 0:
            continue
        block = pool.blocks[block_id]
        if block.is_null or block.ref_cnt != 0:
            continue
        if block.block_hash is not None or block.block_hash_num_tokens is not None:
            continue  # already in use on this engine; never disturb it
        block.set_block_hash(block_hash, num_tokens=num_tokens)
        pool.cached_block_hash_to_block.insert(block_hash, block)
        installed.append(block)

    _requeue_to_tail(pool, installed)
    logger.info("[kv_cache] restored %d entries", len(installed))
    return installed


def _requeue_to_tail(pool, blocks: list) -> None:
    """Move restored blocks to the free-queue tail.

    Mandatory, not an optimisation. vLLM keeps "hand out uncached blocks first"
    purely by queue *position*, and naming a block does not move it. On a freshly
    woken standby every block is unnamed and queued in id order, so a restored
    prefix sits at the head and the first request served is handed its own blocks
    and overwrites them -- the cache would not survive one request.

    Sorted descending by ``num_tokens`` so the deepest match sits nearest the
    head and is evicted first, leaving the shallow leading prefix -- which more
    requests can reuse -- alive longest. Same order ``free_blocks`` produces.
    """
    if not blocks:
        return
    for block in blocks:
        pool.free_block_queue.remove(block)
    blocks.sort(key=lambda b: (b.block_hash_num_tokens or 0), reverse=True)
    pool.free_block_queue.append_n(blocks)
