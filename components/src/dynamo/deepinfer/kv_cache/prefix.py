# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefix cache over DeepInfer's block allocator, with SWA state snapshots.

DeepInfer keeps its allocator (refcounted blocks; pin = ref + 1). On top sit
vLLM's chained block hashes, LRU retention and KV events. A cached prefix of K
tokens (a multiple of the 128-token block) is the chain of blocks
``[0, K/128)``, each pinned once by the cache, plus a snapshot of the request's
state slot taken when its prefill reached K: the sliding-window rings cannot be
rebuilt from the blocks.

- ``admit``: find the deepest snapshot at K <= L - KEEP, allocate with the
  K/128 shared blocks (``try_allocate_seq_with_prefix`` pins them), restore the
  snapshot into the new state slot, start prefill at K, and set
  ``chunk_checkpoints`` so prefill stops where snapshots are wanted.
- ``on_post_schedule``: when a prompt chunk starts at a checkpoint, snapshot the
  state slot, pin the new blocks of ``[0, K/128)`` and emit ``BlockStored``.
- ``on_finished`` / ``release_hold``: drop the request's references and free it.
- Eviction: exact LRU over cached leaf blocks that no live request holds; unpin,
  free the snapshot, emit ``BlockRemoved``.

Imported lazily by rank workers: no ``dynamo._core``; torch only when the
engine's state tensor is a torch tensor.
"""

from __future__ import annotations

import heapq
import importlib
import itertools
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .base import PREFILL_FREE_FRACTION, Admission, CacheStats, HeldKV, KVEvent
from .noop import NoopCacheManager
from .vllm.hashing import get_hash_fn_by_name
from .vllm.kv_cache_utils import (
    BlockHash,
    hash_block_tokens,
    init_none_hash,
    maybe_convert_block_hash,
)

logger = logging.getLogger(__name__)

# vLLM's default prefix_caching_hash_algo (pickle + SHA-256): ~5 us per block,
# where sha256_cbor costs ~27 us. Dynamo hashes the event token ids itself, so
# the value is only an opaque parent-chain id on the wire.
_HASH_ALGO = "sha256"


@dataclass(eq=False)
class _Block:
    """A cached full block; the cache holds one pin on ``block_id``."""

    block_hash: BlockHash
    block_id: int
    parent: Optional[_Block]
    # Chain position: the block holds tokens [index * 128, (index + 1) * 128).
    index: int
    children: int = 0
    # Live requests whose block table holds block_id.
    users: int = 0
    # Snapshot slot with the state after this block, or -1.
    snapshot: int = -1
    last_used: int = 0
    # Priority of this block's current eviction-heap entry (None: not queued).
    heap_key: Optional[int] = None


@dataclass(eq=False)
class _Request:
    seq_id: int
    seq: Any
    tokens: np.ndarray
    # Chained block hashes of the prompt, computed on demand.
    hashes: list[BlockHash] = field(default_factory=list)
    # Cached blocks this request's block table holds, root first.
    held: dict[BlockHash, _Block] = field(default_factory=dict)
    # Prompt positions still to snapshot, ascending.
    checkpoints: deque[int] = field(default_factory=deque)


def _finished(seq: Any) -> bool:
    return seq.status.name.startswith("FINISHED")


def _external(block: _Block) -> int:
    return int(maybe_convert_block_hash(block.block_hash))


def _copy(dst: Any, src: Any) -> None:
    if isinstance(dst, np.ndarray):
        dst[...] = src
    else:
        # Ordered on the current (default) stream with the prefill kernels.
        dst.copy_(src, non_blocking=True)


class _SnapshotPool:
    """Preallocated slots for the stored part of a state slot (DeepInfer's
    ``state_size_to_disk`` bytes: the producer layers' sliding-window rings)."""

    def __init__(self, cache_engine: Any, num_slots: int, device: str) -> None:
        if num_slots <= 0:
            raise ValueError(f"need at least one snapshot slot, got {num_slots}")
        if device not in ("gpu", "host"):
            raise ValueError(f"snapshot device must be gpu or host, got {device!r}")
        # DeepInfer: classmethods of GPUCacheInterface (cache_engine.gpu_cache);
        # the CPU fake keeps them on the cache engine.
        self._io = (
            cache_engine
            if hasattr(cache_engine, "select_state_slice_to_store")
            else cache_engine.gpu_cache
        )
        self._states = cache_engine.gpu_states
        self.nbytes = int(cache_engine.state_size_to_disk)
        if self.nbytes <= 0:
            raise ValueError("the prefix cache needs a model with SWA state")
        self._host = device == "host"
        shape = (num_slots, self.nbytes)
        if isinstance(self._states, np.ndarray):
            self._buf = np.zeros(shape, dtype=self._states.dtype)
        else:
            torch = importlib.import_module("torch")
            if self._host:
                self._buf = torch.empty(
                    shape, dtype=self._states.dtype, pin_memory=True
                )
            else:
                self._buf = torch.empty(
                    shape, dtype=self._states.dtype, device=self._states.device
                )
        self._free = list(range(num_slots - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> Optional[int]:
        return self._free.pop() if self._free else None

    def release(self, slot: int) -> None:
        self._free.append(slot)

    def save(self, slot: int, state_id: int, pos: int) -> None:
        # With DEEPINFER_PREFILL_STORE_EXTRA_STATE off (enforced at launch) this
        # is a view of the slot's first state_size_to_disk bytes.
        src = self._io.select_state_slice_to_store(state_id, pos, self._states)
        _copy(self._buf[slot], src)

    def load(self, slot: int, state_id: int, pos: int) -> None:
        if self._host:
            # DeepInfer's loader would block the thread on the pinned H2D copy.
            _copy(self._states[state_id, : self.nbytes], self._buf[slot])
        else:
            self._io.load_state_from_store_to_prefill_cache(
                state_id, pos, self._buf[slot], self._states
            )


class PrefixCacheManager(NoopCacheManager):
    """Hash-chain prefix cache with SWA snapshots (see the module docstring)."""

    def __init__(
        self,
        engine: Any,
        *,
        snapshot_interval: int = 32768,
        snapshot_slots: int = 256,
        snapshot_device: str = "gpu",
        free_watermark: float = 0.05,
    ) -> None:
        super().__init__(engine)
        if not engine.scheduler_config.chunked_prefill:
            raise ValueError("the prefix cache needs chunked_prefill=True")
        bs = self.block_size
        self.keep_tokens = int(
            os.environ.get("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS") or 140
        )
        self.snapshot_interval = max(0, snapshot_interval) // bs * bs
        total = int(engine.cache_config.num_gpu_blocks)
        # Retained blocks count as used by DeepInfer's 90% prefill gate, so the
        # cache always leaves at least that headroom free.
        self.floor_blocks = math.ceil(
            max(free_watermark, PREFILL_FREE_FRACTION) * total
        )
        self._hash_fn = get_hash_fn_by_name(_HASH_ALGO)
        init_none_hash(self._hash_fn)
        self._pool = _SnapshotPool(
            engine.driver_worker.cache_engine, snapshot_slots, snapshot_device
        )
        self._blocks: dict[BlockHash, _Block] = {}
        # Eviction candidates: (key, tiebreak, block); stale entries are skipped.
        self._heap: list[tuple[int, int, _Block]] = []
        self._tiebreak = itertools.count()
        self._clock = itertools.count(1)
        # Cached blocks no live request holds (an upper bound on what eviction frees).
        self._num_unused = 0
        # Keyed by seq_id (request ids can repeat across retries) and by HeldKV.
        self._requests: dict[int, _Request] = {}
        self._holds: dict[int, _Request] = {}
        # The last request admit() turned away, kept for its hashes.
        self._retry: Optional[_Request] = None
        self._events: list[KVEvent] = []
        self._query_tokens = 0
        self._hit_tokens = 0
        logger.info(
            "DeepInfer prefix cache: keep=%d interval=%d snapshots=%d x %d B on %s, "
            "free floor=%d/%d blocks",
            self.keep_tokens,
            self.snapshot_interval,
            snapshot_slots,
            self._pool.nbytes,
            snapshot_device,
            self.floor_blocks,
            total,
        )

    # -- KVCacheManager -------------------------------------------------------

    def admit(self, seq: Any, token_ids: np.ndarray) -> Admission:
        seq_id = int(seq.seq_id)
        num_tokens = len(token_ids)
        req = self._retry
        if req is None or req.seq_id != seq_id:
            req = _Request(seq_id, seq, token_ids)
        self._retry = None
        chain = self._lookup(req, self._last_checkpoint(num_tokens) // self.block_size)
        scheduler = self.engine.scheduler
        hit = len(chain) * self.block_size
        if chain:
            # A hit allocates now (shared blocks plus restored state), so leave
            # DeepInfer its prefill and decode headroom. A miss allocates later,
            # inside DeepInfer's scheduler (see before_step).
            # DeepInfer gives a miss its state slot only at the head of its FCFS
            # queue, which never skips the head: keep a slot free for every
            # waiting miss, or the hits queued behind one take the last slot
            # and nothing ever frees one.
            slotless = sum(
                1 for s in scheduler.waiting if s.state_id < 0 and not _finished(s)
            )
            if self.engine.state_manager.get_num_free_blocks() <= slotless:
                self._retry = req
                return Admission(admitted=False)
            for block in chain:
                self._acquire(req, block)
            need = -(-(num_tokens + 1) // self.block_size) - len(chain)
            want = need + self.floor_blocks + len(scheduler.running)
            self._make_room(want, want)
            if self.engine.block_manager.get_num_free_blocks() < want or not (
                scheduler.try_allocate_seq_with_prefix(seq, [b.block_id for b in chain])
            ):
                self._release(req)
                self._retry = req
                return Admission(admitted=False)
            self._pool.load(chain[-1].snapshot, seq.state_id, hit)
            seq.num_cached_tokens = hit
            seq.cached_hit_tokens = hit
            logger.debug(
                "Prefix hit for %s: %d of %d tokens", seq.request_id, hit, num_tokens
            )
        req.checkpoints = deque(self._checkpoints(hit, num_tokens))
        if req.checkpoints:
            seq.chunk_checkpoints = list(req.checkpoints)
        self._requests[seq_id] = req
        self._query_tokens += num_tokens
        self._hit_tokens += hit
        scheduler.add_sequence(seq)
        return Admission(admitted=True, hit_tokens=hit)

    def before_step(self) -> None:
        scheduler = self.engine.scheduler
        waiting = scheduler.waiting
        if len(self._requests) > len(waiting) + len(scheduler.running):
            # Seqs DeepInfer finished without finalize_seq (preempted, ignored).
            self._reap()
        if not self._num_unused:
            return
        head = next((s for s in waiting if not _finished(s)), None)
        if head is None:
            return
        # DeepInfer prefills the head only while usage is at most 90%, and
        # allocates its blocks then (none yet on a miss): evict for both.
        need = (
            0 if len(head.block_table) else -(-(head.get_len() + 1) // self.block_size)
        )
        self._make_room(need + self.floor_blocks, max(need, self.floor_blocks))

    def on_post_schedule(self, scheduler_outputs: Any, engine: Any) -> None:
        if not self._requests or not scheduler_outputs.prompt_run:
            return
        for seq in scheduler_outputs.scheduled_seqs:
            req = self._requests.get(seq.seq_id)
            if req is None or not req.checkpoints:
                continue
            # The chunk about to run starts at num_cached_tokens; the state slot
            # holds exactly the tokens before it.
            pos = int(seq.num_cached_tokens)
            while req.checkpoints and req.checkpoints[0] < pos:
                req.checkpoints.popleft()
            if req.checkpoints and req.checkpoints[0] == pos:
                req.checkpoints.popleft()
                self._capture(seq, req, pos)

    def on_finished(self, seq: Any) -> None:
        req = self._requests.pop(seq.seq_id, None)
        if req is not None:
            self._release(req)
        super().on_finished(seq)
        self._make_room(self.floor_blocks, self.floor_blocks)

    def ensure_free(self, num_blocks: int) -> int:
        return self._make_room(num_blocks, num_blocks)

    def take_events(self) -> list[KVEvent]:
        events, self._events = self._events, []
        return events

    def stats(self) -> CacheStats:
        stats = super().stats()
        # Blocks only the cache holds are evictable: report them as free, as
        # vLLM does (the router's overload threshold reads this number).
        stats.used_blocks = max(0, stats.used_blocks - self._num_unused)
        if self._query_tokens:
            stats.hit_rate = self._hit_tokens / self._query_tokens
        return stats

    def hold_for_transfer(self, seq: Any) -> HeldKV:
        req = self._requests.pop(seq.seq_id, None)
        held = super().hold_for_transfer(seq)
        if req is not None:
            # The hold pins these blocks as well; keep them out of eviction.
            self._holds[id(held)] = req
        return held

    def release_hold(self, held: HeldKV) -> None:
        super().release_hold(held)
        req = self._holds.pop(id(held), None)
        if req is not None:
            self._release(req)
        self._make_room(self.floor_blocks, self.floor_blocks)

    # -- positions ------------------------------------------------------------

    def _last_checkpoint(self, num_tokens: int) -> int:
        """Deepest resumable position: consumer (YOCO) layers rebuild their
        state from the last KEEP tokens, so those must be recomputed."""
        return (
            max(0, num_tokens - self.keep_tokens) // self.block_size * self.block_size
        )

    def _checkpoints(self, hit: int, num_tokens: int) -> list[int]:
        last = self._last_checkpoint(num_tokens)
        points = {last} if last else set()
        if self.snapshot_interval:
            points.update(range(self.snapshot_interval, last, self.snapshot_interval))
        return sorted(p for p in points if p > hit)

    # -- hashing and lookup ---------------------------------------------------

    def _block_hash(self, req: _Request, index: int) -> BlockHash:
        hashes, bs = req.hashes, self.block_size
        while len(hashes) <= index:
            i = len(hashes)
            hashes.append(
                hash_block_tokens(
                    self._hash_fn,
                    hashes[-1] if hashes else None,
                    req.tokens[i * bs : (i + 1) * bs].tolist(),
                )
            )
        return hashes[index]

    def _lookup(self, req: _Request, max_blocks: int) -> list[_Block]:
        """The longest cached chain of the prompt that ends at a snapshot."""
        chain: list[_Block] = []
        best = 0
        for i in range(max_blocks):
            block = self._blocks.get(self._block_hash(req, i))
            if block is None:
                break
            chain.append(block)
            if block.snapshot >= 0:
                best = i + 1
        return chain[:best]

    # -- capture --------------------------------------------------------------

    def _capture(self, seq: Any, req: _Request, pos: int) -> None:
        """Cache the prompt prefix [0, pos): pin its blocks, snapshot the state."""
        bs = self.block_size
        num = pos // bs
        last = self._blocks.get(self._block_hash(req, num - 1))
        slot = None
        if last is None or last.snapshot < 0:
            # Evicts before the walk below, so it cannot drop blocks it registers.
            slot = self._snapshot_slot()
            if slot is None:
                logger.debug("No snapshot slot for %s at %d", seq.request_id, pos)
                return
        table = seq.block_table
        parent: Optional[_Block] = None
        new: list[_Block] = []
        for i in range(num):
            h = req.hashes[i]
            block = self._blocks.get(h)
            if block is None:
                block = _Block(h, int(table[i]), parent, i)
                self._blocks[h] = block
                self._num_unused += 1
                if parent is not None:
                    parent.children += 1
                new.append(block)
            # Another request may have cached the same tokens in other blocks.
            if block.block_id == table[i]:
                self._acquire(req, block)
            parent = block
        assert parent is not None
        if new:
            self.engine.block_manager.pin_blocks([b.block_id for b in new])
            first = new[0]
            self._events.append(
                KVEvent(
                    kind="stored",
                    block_hashes=[_external(b) for b in new],
                    parent_block_hash=(
                        _external(first.parent) if first.parent is not None else None
                    ),
                    token_ids=req.tokens[first.index * bs : pos].tolist(),
                    block_size=bs,
                )
            )
        if slot is None:
            return
        self._pool.save(slot, int(seq.state_id), pos)
        parent.snapshot = slot
        parent.last_used = next(self._clock)
        self._maybe_push(parent)

    def _snapshot_slot(self) -> Optional[int]:
        slot = self._pool.alloc()
        if slot is not None:
            return slot
        # Preempted requests may still pin snapshotted blocks.
        self._reap()
        ids: list[int] = []
        hashes: list[int] = []
        while slot is None:
            block = self._pop_evictable()
            if block is None:
                break
            self._drop(block, ids, hashes)
            slot = self._pool.alloc()
        self._flush_evicted(ids, hashes)
        return slot

    # -- references -----------------------------------------------------------

    def _acquire(self, req: _Request, block: _Block) -> None:
        if block.block_hash in req.held:
            return
        req.held[block.block_hash] = block
        if block.users == 0:
            self._num_unused -= 1
        block.users += 1

    def _release(self, req: _Request) -> None:
        tick = next(self._clock)
        for block in reversed(req.held.values()):
            block.users -= 1
            block.last_used = tick
            if block.users == 0:
                self._num_unused += 1
                self._maybe_push(block)
        req.held.clear()

    def _reap(self) -> None:
        """Release requests DeepInfer freed without ``finalize_seq``: a preempted
        seq is freed inside ``schedule()`` (``Scheduler::_preempt``)."""
        for seq_id, req in list(self._requests.items()):
            seq = req.seq
            if _finished(seq) and not len(seq.block_table):
                del self._requests[seq_id]
                self._release(req)

    # -- eviction -------------------------------------------------------------

    def _maybe_push(self, block: _Block) -> None:
        if block.users or block.children:
            return
        # A leaf without a snapshot cannot serve a hit: it goes first.
        key = block.last_used if block.snapshot >= 0 else 0
        block.heap_key = key
        heapq.heappush(self._heap, (key, next(self._tiebreak), block))
        if len(self._heap) > 2 * len(self._blocks) + 1024:
            self._heap = [
                (b.heap_key, next(self._tiebreak), b)
                for b in self._blocks.values()
                if b.heap_key is not None
            ]
            heapq.heapify(self._heap)

    def _pop_evictable(self) -> Optional[_Block]:
        while self._heap:
            key, _, block = heapq.heappop(self._heap)
            if key != block.heap_key:
                continue
            block.heap_key = None
            if block.users == 0 and block.children == 0:
                return block
        return None

    def _drop(self, block: _Block, ids: list[int], hashes: list[int]) -> None:
        del self._blocks[block.block_hash]
        self._num_unused -= 1
        if block.snapshot >= 0:
            self._pool.release(block.snapshot)
            block.snapshot = -1
        ids.append(block.block_id)
        hashes.append(_external(block))
        parent = block.parent
        if parent is not None:
            parent.children -= 1
            self._maybe_push(parent)

    def _flush_evicted(self, ids: list[int], hashes: list[int]) -> None:
        if ids:
            self.engine.block_manager.unpin_blocks(ids)
            self._events.append(KVEvent(kind="removed", block_hashes=hashes))

    def _make_room(self, want: int, at_least: int) -> int:
        """Evict LRU blocks until ``want`` blocks are free; do nothing when even
        evicting every unheld block would leave fewer than ``at_least``."""
        block_manager = self.engine.block_manager
        free = int(block_manager.get_num_free_blocks())
        if free >= want:
            return 0
        if free + self._num_unused < at_least:
            self._reap()
            if free + self._num_unused < at_least:
                return 0
        ids: list[int] = []
        hashes: list[int] = []
        # Each unheld block's only reference is the cache's pin.
        while free + len(ids) < want:
            block = self._pop_evictable()
            if block is None:
                break
            self._drop(block, ids, hashes)
        self._flush_evicted(ids, hashes)
        return int(block_manager.get_num_free_blocks()) - free
