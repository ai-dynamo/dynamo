# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefix cache over DeepInfer's block allocator, with SWA state snapshots and
an optional host-DRAM tier.

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

Host tier (``host_bytes``): eviction demotes a GPU leaf that can still serve a
hit (it has a snapshot or cached blocks below it) into a pinned host row
instead of dropping it, and emits nothing: the block is still a hit. A host
block sits only below GPU blocks or host blocks, never above a GPU block: a GPU
block is a leaf for eviction once no child is on GPU, and a host block once it
has no child. ``admit`` shares the GPU part of a hit and copies the host part
into the seq's freshly allocated blocks, which the cache then pins as its own.
A request that recomputes host blocks hands the cache its GPU copy instead.
Host eviction (LRU over host leaves) emits ``BlockRemoved``. Snapshots default
to host memory with the tier, and their slots grow with it.

SSD tier (``ssd_bytes``, below the host tier): a block demoted to host is also
written to a slot of a per-rank file (write-through, from the host row, once its
GPU copy has landed; a write the cap or a full file turns away is queued, root
first, and retried while the block is still on host), and a block with such a
copy leaves host or GPU for the SSD tier without I/O and without an event: the
copy is the block. The chain is GPU blocks, then host, then SSD. ``admit``
serves only the GPU and host part of a hit (up to its last snapshot) and, when
the SSD blocks behind it are worth it, starts a best-effort read of them into
host rows: the request is not delayed, later ones hit from host. A host row
stays reserved while a write reads it. Blocks and snapshots stay on the engine
thread; the I/O threads (``ssd.py``) only move bytes, and their completions are
applied on the engine thread (in ``before_step``, ``admit`` and ``_make_room``).

Imported lazily by rank workers: no ``dynamo._core``; torch only when the
engine's state tensor is a torch tensor.
"""

from __future__ import annotations

import enum
import heapq
import importlib
import itertools
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .base import PREFILL_FREE_FRACTION, Admission, CacheStats, HeldKV, KVEvent
from .host import HostBlockPool, PinnedRows, row_nbytes
from .noop import NoopCacheManager
from .ssd import SSDPool, Ticket
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
# With the host tier and snapshots on host, this share of host_bytes holds
# snapshots (~1.4 MB each against ~29 MB of KV per 32K-token interval).
HOST_SNAPSHOT_SHARE = 16
STATS_LOG_INTERVAL_S = 60.0
# An SSD hit shorter than this many blocks is not read back into host: the
# read would cost more than recomputing it.
SSD_MIN_READ_BLOCKS = 8
SSD_MAX_INFLIGHT_WRITES = 64
# Blocks whose write was skipped wait for a turn, up to this many per write the
# cap allows; the deepest are forgotten first.
SSD_PENDING_WRITES_FACTOR = 4


class Tier(enum.IntEnum):
    """Where a hit reads a block from; chains run GPU, then host, then SSD."""

    GPU = 0
    HOST = 1
    SSD = 2


class SsdState(enum.Enum):
    """The block's SSD copy: none (or none yet), being written, or complete."""

    NONE = 0
    WRITING = 1
    DONE = 2


@dataclass(eq=False)
class _Block:
    """A cached full block, on GPU (the cache holds one pin on ``block_id``),
    in a host row (``host_slot``) or only in the SSD file (``ssd_slot``)."""

    block_hash: BlockHash
    # -1 while the block is off GPU.
    block_id: int
    parent: Optional[_Block]
    # Chain position: the block holds tokens [index * 128, (index + 1) * 128).
    index: int
    tier: Tier = Tier.GPU
    children: int = 0
    # The first child, and a dict of the others once the chain branches (chains
    # are most of the cache, so most blocks have at most one child).
    kid: Optional[_Block] = None
    more_kids: Optional[dict[BlockHash, _Block]] = None
    # Children per tier: a block is a leaf for eviction from a tier once no child
    # is on that tier or above it (children on lower tiers do not need it).
    gpu_children: int = 0
    host_children: int = 0
    # Live requests whose block table holds block_id (while admit promotes a
    # host block: the request that will).
    users: int = 0
    # Snapshot slot with the state after this block, or -1.
    snapshot: int = -1
    # Host tier row holding the block while it is on host, or -1.
    host_slot: int = -1
    # SSD slot of the copy (a backup while the block is on GPU or host): set
    # from the start of the write until the copy is dropped.
    ssd_slot: int = -1
    ssd_state: SsdState = SsdState.NONE
    # The write in flight (ssd_state WRITING): it reads a host row.
    write: Optional[_Write] = None
    # An SSD read of this block is in flight: it must stay put.
    reading: bool = False
    # Read from SSD into host and not served from there yet.
    ssd_warm: bool = False
    last_used: int = 0
    # Priority of this block's current entry in its tier's eviction heap, in the
    # SSD heap and in the heap of SSD-tier snapshot holders (None: not queued).
    heap_key: Optional[int] = None
    ssd_key: Optional[int] = None
    snap_key: Optional[int] = None

    @property
    def on_gpu(self) -> bool:
        return self.tier is Tier.GPU

    def add_kid(self, child: _Block) -> None:
        if self.kid is None:
            self.kid = child
        else:
            if self.more_kids is None:
                self.more_kids = {}
            self.more_kids[child.block_hash] = child

    def remove_kid(self, child: _Block) -> None:
        if self.kid is child:
            self.kid = None
            if self.more_kids:
                _, self.kid = self.more_kids.popitem()
        else:
            assert self.more_kids is not None
            del self.more_kids[child.block_hash]
        if not self.more_kids:
            self.more_kids = None

    def kids(self) -> list[_Block]:
        if self.kid is None:
            return []
        return [self.kid, *(self.more_kids or {}).values()]


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


@dataclass
class _Evicted:
    """One eviction pass, applied by ``_flush``."""

    # GPU blocks to unpin.
    ids: list[int] = field(default_factory=list)
    # Of those, (block_id, host_slot) to copy to host first.
    stores: list[tuple[int, int]] = field(default_factory=list)
    # Of those, the blocks to write to SSD from their host row once copied.
    writes: list[_Block] = field(default_factory=list)
    # External hashes of the blocks that left the cache.
    removed: list[int] = field(default_factory=list)


@dataclass(eq=False)
class _Write:
    """An SSD write in flight: it reads the host ``row`` until it completes."""

    block: _Block
    row: int
    # The block left that row meanwhile: free it when the write is done.
    free_row: bool = False


@dataclass(eq=False)
class _Read:
    """An SSD read in flight: the blocks, the SSD slots read and the host rows
    reserved for them."""

    blocks: list[_Block]
    slots: list[int]
    rows: list[int]


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
        self.row_bytes = row_nbytes(self._states[0, : self.nbytes])
        if self._host:
            # Pinned rows in chunks, clamped to the host's free memory.
            self._rows: Any = PinnedRows(self._states[:1, : self.nbytes], num_slots)
            num_slots = self._rows.count
            if not num_slots:
                raise ValueError("no host memory left for prefix snapshots")
        elif isinstance(self._states, np.ndarray):
            self._buf = np.zeros((num_slots, self.nbytes), dtype=self._states.dtype)
        else:
            torch = importlib.import_module("torch")
            self._buf = torch.empty(
                (num_slots, self.nbytes),
                dtype=self._states.dtype,
                device=self._states.device,
            )
        self.num_slots = num_slots
        self._free = list(range(num_slots - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def _slot(self, slot: int) -> Any:
        return self._rows.row(slot) if self._host else self._buf[slot]

    def alloc(self) -> Optional[int]:
        return self._free.pop() if self._free else None

    def release(self, slot: int) -> None:
        self._free.append(slot)

    def save(self, slot: int, state_id: int, pos: int) -> None:
        # With DEEPINFER_PREFILL_STORE_EXTRA_STATE off (enforced at launch) this
        # is a view of the slot's first state_size_to_disk bytes.
        src = self._io.select_state_slice_to_store(state_id, pos, self._states)
        _copy(self._slot(slot), src)

    def load(self, slot: int, state_id: int, pos: int) -> None:
        if self._host:
            # DeepInfer's loader would block the thread on the pinned H2D copy.
            _copy(self._states[state_id, : self.nbytes], self._slot(slot))
        else:
            self._io.load_state_from_store_to_prefill_cache(
                state_id, pos, self._slot(slot), self._states
            )


class PrefixCacheManager(NoopCacheManager):
    """Hash-chain prefix cache with SWA snapshots (see the module docstring)."""

    def __init__(
        self,
        engine: Any,
        *,
        snapshot_interval: int = 32768,
        snapshot_slots: int = 256,
        snapshot_device: Optional[str] = None,
        free_watermark: float = 0.05,
        host_bytes: int = 0,
        ssd_dir: Optional[str] = None,
        ssd_bytes: int = 0,
        ssd_rank: int = 0,
        ssd_share: float = 1.0,
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
        cache_engine = engine.driver_worker.cache_engine
        host_bytes = max(0, int(host_bytes))
        if snapshot_device is None:
            snapshot_device = "host" if host_bytes else "gpu"
        if host_bytes and snapshot_device == "host":
            # Snapshots come out of the same host budget.
            states = cache_engine.gpu_states
            snapshot_bytes = row_nbytes(
                states[0, : int(cache_engine.state_size_to_disk)]
            )
            snapshot_slots = max(
                snapshot_slots, host_bytes // HOST_SNAPSHOT_SHARE // snapshot_bytes
            )
        self._pool = _SnapshotPool(cache_engine, snapshot_slots, snapshot_device)
        self._host: Optional[HostBlockPool] = None
        if host_bytes:
            if snapshot_device == "host":
                host_bytes -= self._pool.num_slots * self._pool.row_bytes
            self._host = HostBlockPool(
                cache_engine.gpu_blocks,
                max(0, host_bytes) // row_nbytes(cache_engine.gpu_blocks[0]),
            )
            if not self._host.capacity:
                logger.warning("No host memory left for the prefix cache's host tier")
                self._host = None
        self._ssd: Optional[SSDPool] = None
        ssd_bytes = max(0, int(ssd_bytes))
        if ssd_bytes and self._host is not None and ssd_dir:
            self._ssd = self._open_ssd(
                ssd_dir,
                ssd_bytes,
                ssd_rank,
                row_nbytes(cache_engine.gpu_blocks[0]),
                ssd_share,
            )
        elif ssd_bytes:
            logger.warning("The prefix cache's SSD tier needs the host tier and a dir")
        self._blocks: dict[BlockHash, _Block] = {}
        # Eviction candidates: (key, tiebreak, block); stale entries are skipped.
        self._heap: list[tuple[int, int, _Block]] = []
        self._host_heap: list[tuple[int, int, _Block]] = []
        # Blocks with a complete SSD copy, and the SSD-tier leaves among them
        # that hold a snapshot (what a full snapshot pool drops first).
        self._ssd_heap: list[tuple[int, int, _Block]] = []
        self._ssd_snap_heap: list[tuple[int, int, _Block]] = []
        # Host blocks waiting for an SSD write (cap reached or no slot free):
        # (chain index, tiebreak, block, host row), ancestors first. Not pinned:
        # an entry is checked when it is popped.
        self._pending_writes: list[tuple[int, int, _Block, int]] = []
        # Blocks in the host and SSD tiers.
        self._num_host = 0
        self._num_ssd = 0
        self._tiebreak = itertools.count()
        self._clock = itertools.count(1)
        # GPU blocks no live request holds (an upper bound on what eviction frees).
        self._num_unused = 0
        # Keyed by seq_id (request ids can repeat across retries) and by HeldKV.
        self._requests: dict[int, _Request] = {}
        self._holds: dict[int, _Request] = {}
        # The last request admit() turned away, kept for its hashes.
        self._retry: Optional[_Request] = None
        self._events: list[KVEvent] = []
        self._query_tokens = 0
        self._hit_tokens = 0
        # Host tier counters: hit tokens served from host, blocks copied to
        # host, copied back, taken back from a request that recomputed them,
        # and dropped from host.
        self.host_hit_tokens = 0
        self.host_stores = 0
        self.host_loads = 0
        self.host_rehomed = 0
        self.host_evictions = 0
        # SSD tier counters: blocks written, read back into host, slots reclaimed
        # (backup copies cleared and SSD-tier blocks dropped), failed I/O
        # operations, and hit tokens served from host rows an SSD read filled.
        self.ssd_stores = 0
        self.ssd_loads = 0
        self.ssd_evictions = 0
        self.ssd_errors = 0
        self.ssd_hit_tokens = 0
        self._last_log = time.monotonic()
        logger.info(
            "DeepInfer prefix cache: keep=%d interval=%d snapshots=%d x %d B on %s, "
            "free floor=%d/%d blocks, host tier=%s, ssd tier=%s",
            self.keep_tokens,
            self.snapshot_interval,
            self._pool.num_slots,
            self._pool.row_bytes,
            snapshot_device,
            self.floor_blocks,
            total,
            (
                f"{self._host.capacity} blocks ({self._host.nbytes / 2**30:.1f} GiB)"
                if self._host is not None
                else "off"
            ),
            (
                f"{self._ssd.capacity} blocks ({self._ssd.nbytes / 2**30:.1f} GiB)"
                if self._ssd is not None
                else "off"
            ),
        )

    def _open_ssd(
        self, ssd_dir: str, ssd_bytes: int, rank: int, row_bytes: int, share: float
    ) -> Optional[SSDPool]:
        """The rank's SSD file; None when no room is left. A hit ends at a
        snapshot, so no more blocks than the snapshots (each resumes at most one
        interval) can ever serve are worth keeping."""
        per_snapshot = self.snapshot_interval // self.block_size
        pool = SSDPool(
            os.path.join(ssd_dir, f"rank{rank}.kv"),
            row_bytes,
            ssd_bytes,
            max_blocks=self._pool.num_slots * per_snapshot if per_snapshot else None,
            share=share,
            max_inflight_writes=SSD_MAX_INFLIGHT_WRITES,
        )
        return pool if pool.capacity else None

    # -- KVCacheManager -------------------------------------------------------

    def admit(self, seq: Any, token_ids: np.ndarray) -> Admission:
        self._poll_ssd()
        seq_id = int(seq.seq_id)
        num_tokens = len(token_ids)
        req = self._retry
        if req is None or req.seq_id != seq_id:
            req = _Request(seq_id, seq, token_ids)
        self._retry = None
        chain, ssd_run = self._lookup(
            req, self._last_checkpoint(num_tokens) // self.block_size
        )
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
            # Held, host blocks stay put while _make_room evicts.
            for block in chain:
                self._acquire(req, block)
            # The GPU part is shared; the host part goes into new blocks.
            shared = next((i for i, b in enumerate(chain) if not b.on_gpu), len(chain))
            need = -(-(num_tokens + 1) // self.block_size) - shared
            want = need + self.floor_blocks + len(scheduler.running)
            self._make_room(want, want)
            if self.engine.block_manager.get_num_free_blocks() < want or not (
                scheduler.try_allocate_seq_with_prefix(
                    seq, [b.block_id for b in chain[:shared]]
                )
            ):
                self._release(req)
                self._retry = req
                return Admission(admitted=False)
            if shared < len(chain):
                self._promote(chain[shared:], seq.block_table[shared : len(chain)])
                self.host_hit_tokens += (len(chain) - shared) * self.block_size
            self._pool.load(chain[-1].snapshot, seq.state_id, hit)
            seq.num_cached_tokens = hit
            seq.cached_hit_tokens = hit
            logger.debug(
                "Prefix hit for %s: %d of %d tokens (%d from host)",
                seq.request_id,
                hit,
                num_tokens,
                (len(chain) - shared) * self.block_size,
            )
        if ssd_run:
            # Best effort: this request goes on without them.
            self._read_ssd(ssd_run)
        req.checkpoints = deque(self._checkpoints(hit, num_tokens))
        if req.checkpoints:
            seq.chunk_checkpoints = list(req.checkpoints)
        self._requests[seq_id] = req
        self._query_tokens += num_tokens
        self._hit_tokens += hit
        scheduler.add_sequence(seq)
        return Admission(admitted=True, hit_tokens=hit)

    def before_step(self) -> None:
        self._poll_ssd()
        scheduler = self.engine.scheduler
        waiting = scheduler.waiting
        if len(self._requests) > len(waiting) + len(scheduler.running):
            # Seqs DeepInfer finished without finalize_seq (preempted, ignored).
            self._reap()
        self._log_stats()
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

    def close(self) -> None:
        """Wait for the SSD I/O in flight: it reads and writes host rows that
        are unpinned and unmapped once the cache is gone."""
        if self._ssd is not None:
            self._ssd.close()
            self._ssd = None
            self._pending_writes.clear()

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

    def _lookup(
        self, req: _Request, max_blocks: int
    ) -> tuple[list[_Block], list[_Block]]:
        """The longest cached chain of the prompt that ends at a snapshot, in two
        parts: what ``admit`` serves now (GPU blocks, then host blocks, up to a
        snapshot), and the blocks behind it up to the last snapshot of the
        whole chain (which then continues on SSD)."""
        chain: list[_Block] = []
        served = best = 0
        for i in range(max_blocks):
            block = self._blocks.get(self._block_hash(req, i))
            # Tiers only go down the chain: GPU, host, SSD.
            if block is None or (chain and block.tier < chain[-1].tier):
                break
            chain.append(block)
            if block.snapshot >= 0:
                best = i + 1
                if block.tier is not Tier.SSD:
                    served = i + 1
        return chain[:served], chain[served:best]

    # -- host tier ------------------------------------------------------------

    def _promote(self, blocks: list[_Block], block_ids: Any) -> None:
        """Copy held host blocks into the seq's new GPU blocks and cache those."""
        assert self._host is not None
        ids = [int(i) for i in block_ids]
        self._host.load([b.host_slot for b in blocks], ids)
        # The cache's own reference, as for a capture.
        self.engine.block_manager.pin_blocks(ids)
        for block, block_id in zip(blocks, ids):
            # The load is queued on the stream before any later store into the row.
            self._release_row(block)
            block.block_id = block_id
            self._move(block, Tier.GPU)
            if block.ssd_warm:
                block.ssd_warm = False
                self.ssd_hit_tokens += self.block_size
        self.host_loads += len(blocks)

    def _rehome(self, block: _Block, block_id: int) -> None:
        """A request recomputed an off-GPU block into ``block_id``: cache that."""
        assert self._host is not None
        if block.tier is Tier.HOST:
            self._release_row(block)
        block.block_id, block.ssd_warm = block_id, False
        self._move(block, Tier.GPU)
        self._num_unused += 1
        self.host_rehomed += 1

    def _release_row(self, block: _Block) -> None:
        """Give up the block's host row; a write in flight still reads it, and
        frees it when done."""
        assert self._host is not None
        row, block.host_slot = block.host_slot, -1
        write = block.write
        if write is not None and write.row == row:
            write.free_row = True
        else:
            self._host.release(row)

    def _move(self, block: _Block, tier: Tier) -> None:
        """Change the block's tier and its parent's child counts."""
        parent = block.parent
        if parent is not None:
            if block.tier is Tier.GPU:
                parent.gpu_children -= 1
            elif block.tier is Tier.HOST:
                parent.host_children -= 1
            if tier is Tier.GPU:
                parent.gpu_children += 1
            elif tier is Tier.HOST:
                parent.host_children += 1
        self._num_host += (tier is Tier.HOST) - (block.tier is Tier.HOST)
        self._num_ssd += (tier is Tier.SSD) - (block.tier is Tier.SSD)
        block.tier = tier
        # A queued entry was for the old tier's heap.
        block.heap_key = None
        if parent is not None:
            self._maybe_push(parent)

    def _drop(self, block: _Block, out: _Evicted) -> None:
        """Take the block out of the cache: its host row, snapshot and SSD slot
        go back to their pools (a slot an I/O in flight uses, when it is done)."""
        del self._blocks[block.block_hash]
        if block.tier is Tier.HOST:
            self._release_row(block)
            self._num_host -= 1
        elif block.tier is Tier.SSD:
            self._num_ssd -= 1
        if block.snapshot >= 0:
            self._pool.release(block.snapshot)
            block.snapshot = -1
        if block.ssd_slot >= 0 and block.write is None and not block.reading:
            # Nothing to give back once the cache is closed: the file is gone.
            if self._ssd is not None:
                self._ssd.release(block.ssd_slot)
            block.ssd_slot = -1
        block.heap_key = block.ssd_key = block.snap_key = None
        out.removed.append(_external(block))
        parent = block.parent
        if parent is not None:
            parent.children -= 1
            parent.remove_kid(block)
            if block.tier is Tier.GPU:
                parent.gpu_children -= 1
            elif block.tier is Tier.HOST:
                parent.host_children -= 1
            self._maybe_push(parent)

    def _host_slot(self, out: _Evicted) -> Optional[int]:
        assert self._host is not None
        slot = self._host.alloc()
        while slot is None:
            victim = self._pop_evictable(gpu=False)
            if victim is None:
                return None
            self._evict_host(victim, out)
            slot = self._host.alloc()
        return slot

    def _evict_host(self, block: _Block, out: _Evicted) -> None:
        """Free the host row of a leaf: a block with a complete SSD copy moves to
        the SSD tier (no I/O, no event), any other leaves the cache. That takes
        what is chained behind it on SSD along (the block has no copy to hang
        it on, and nothing could reach it any more), so such a block never pins
        its row."""
        assert self._host is not None
        self.host_evictions += 1
        if block.ssd_state is SsdState.DONE:
            self._release_row(block)
            block.ssd_warm = False
            self._move(block, Tier.SSD)
            self._maybe_push(block)
        else:
            if block.kid is not None:
                self._drop_subtree(block.kids(), out)
            self._drop(block, out)

    # -- SSD tier -------------------------------------------------------------

    def _can_write(self, out: _Evicted) -> bool:
        ssd = self._ssd
        return ssd is not None and (
            ssd.inflight_writes + len(out.writes) < ssd.max_inflight_writes
        )

    def _ssd_slot(self, out: _Evicted) -> Optional[int]:
        """A free SSD slot, reclaiming the coldest copy when none is free; None
        (skip the write) when nothing can be."""
        assert self._ssd is not None
        slot = self._ssd.alloc()
        while slot is None:
            victim = self._pop_ssd()
            if victim is None:
                return None
            self._evict_ssd(victim, out)
            slot = self._ssd.alloc()
        return slot

    def _evict_ssd(self, block: _Block, out: _Evicted) -> None:
        """Reclaim the SSD slot of a complete copy. The backup of a block that
        is still on GPU or host is cleared (the block stays cached, no event);
        a block that is only on SSD leaves the cache."""
        assert self._ssd is not None
        self.ssd_evictions += 1
        if block.tier is Tier.SSD:
            self._drop(block, out)
            return
        self._ssd.release(block.ssd_slot)
        block.ssd_slot, block.ssd_state, block.ssd_key = -1, SsdState.NONE, None

    def _write_through(self, block: _Block, row: int, out: _Evicted) -> None:
        """Queue the SSD copy of a block just demoted to host ``row``, unless it
        has one. ``_drain_writes`` starts it at the end of the eviction pass, so
        a burst writes its ancestors first. The row is not pinned: the entry is
        checked (``_write_pending``) when it is popped."""
        if block.ssd_state is SsdState.NONE:
            self._queue_write(block, row)

    def _start_write(self, block: _Block, row: int, out: _Evicted) -> bool:
        """Take an SSD slot for the block and add its write to ``out``; False
        when no slot can be had."""
        slot = self._ssd_slot(out)
        if slot is None:
            return False
        block.ssd_slot, block.ssd_state = slot, SsdState.WRITING
        block.write = _Write(block, row)
        out.writes.append(block)
        return True

    def _queue_write(self, block: _Block, row: int) -> None:
        """Queue a demoted block's write for ``_drain_writes``, root first."""
        heapq.heappush(
            self._pending_writes, (block.index, next(self._tiebreak), block, row)
        )

    def _write_pending(self, entry: tuple[int, int, _Block, int]) -> bool:
        """The queued block is still cached, on the same host row and without an
        SSD copy."""
        _, _, block, row = entry
        return (
            self._blocks.get(block.block_hash) is block
            and block.tier is Tier.HOST
            and block.host_slot == row
            and block.ssd_state is SsdState.NONE
        )

    def _drain_writes(self, out: _Evicted) -> None:
        """Start the queued writes the cap allows, root first. What is left
        waits for a later poll, bounded: the deepest entries are forgotten
        (stale ones first)."""
        if self._ssd is None:
            return
        pending = self._pending_writes
        while pending and self._can_write(out):
            entry = heapq.heappop(pending)
            if not self._write_pending(entry):
                continue
            if not self._start_write(entry[2], entry[3], out):
                # No slot to be had: try again at the next poll.
                heapq.heappush(pending, entry)
                break
        bound = SSD_PENDING_WRITES_FACTOR * self._ssd.max_inflight_writes
        if len(pending) > bound:
            live = [e for e in pending if self._write_pending(e)]
            pending[:] = heapq.nsmallest(bound, live)

    def _poll_ssd(self) -> None:
        """Apply the SSD I/O that completed since the last step."""
        if self._ssd is None:
            return
        out = _Evicted()
        for ticket in self._ssd.poll():
            if ticket.error is not None:
                self.ssd_errors += 1
                logger.warning("SSD %s failed: %r", ticket.kind, ticket.error)
            if ticket.kind == "store":
                self._write_done(ticket)
            else:
                self._read_done(ticket, out)
        self._drain_writes(out)
        self._flush(out, [])

    def _write_done(self, ticket: Ticket) -> None:
        assert self._host is not None and self._ssd is not None
        write: _Write = ticket.tag
        block = write.block
        block.write = None
        if write.free_row:
            self._host.release(write.row)
        live = self._blocks.get(block.block_hash) is block
        if ticket.error is not None or not live:
            self._ssd.release(block.ssd_slot)
            block.ssd_slot, block.ssd_state = -1, SsdState.NONE
        else:
            block.ssd_state = SsdState.DONE
            self.ssd_stores += 1
        if live:
            # Evictable from host again, and a backup copy to reclaim.
            self._maybe_push(block)

    def _read_ssd(self, run: list[_Block]) -> None:
        """Start reading the SSD blocks of ``run`` into host rows (best effort:
        the request that found them goes on without them)."""
        if self._ssd is None or self._host is None:
            return
        run = self._live(run)
        # Host blocks between the served part and the SSD blocks stay: a read
        # under a parent that left host would be of no use.
        held = [b for b in run if b.tier is Tier.HOST]
        for block in held:
            block.users += 1
        try:
            self._start_read(run)
        finally:
            for block in held:
                block.users -= 1
                self._maybe_push(block)

    def _start_read(self, run: list[_Block]) -> None:
        """Read the SSD blocks of ``run`` into host rows. Rows are taken until
        the host has none to give; the read is then cut back to the deepest
        snapshot they cover and the unused rows go back. A read that could not
        reach its first snapshot even with every evictable host block taken is
        skipped before anything is evicted for it."""
        assert self._ssd is not None and self._host is not None
        todo = [b for b in run if b.tier is Tier.SSD and not b.reading]
        if len(todo) < SSD_MIN_READ_BLOCKS:
            return
        # Rows needed to reach the first snapshot worth reading.
        need = min_need = 0
        held = 0
        for block in run:
            held += block.tier is Tier.HOST
            need += block.tier is Tier.SSD and not block.reading
            if not min_need and block.snapshot >= 0 and need >= SSD_MIN_READ_BLOCKS:
                min_need = need
        if not min_need or min_need > self._host.capacity:
            return
        # Loose bound: free rows plus every host block the run does not hold.
        if min_need > self._host.num_free + self._num_host - held:
            return
        out = _Evicted()
        rows: list[int] = []
        for _ in todo:  # the last snapshot of ``run`` needs all of them
            row = self._host_slot(out)
            if row is None:
                break
            rows.append(row)
        # Defensive: evicting for the rows may have dropped blocks of the run.
        run = self._live(run)
        end = need = 0
        for i, block in enumerate(run):
            need += block.tier is Tier.SSD and not block.reading
            if need > len(rows):
                break
            if block.snapshot >= 0:
                end = i + 1
        todo = [b for b in run[:end] if b.tier is Tier.SSD and not b.reading]
        keep = len(todo) if len(todo) >= SSD_MIN_READ_BLOCKS else 0
        for row in rows[keep:]:
            self._host.release(row)
        rows, todo = rows[:keep], todo[:keep]
        try:
            if todo:
                for block in todo:
                    # Held against eviction until the read is done.
                    block.reading = True
                slots = [b.ssd_slot for b in todo]
                try:
                    self._ssd.load(
                        slots,
                        [self._host.row(r) for r in rows],
                        # Rows freed since (by a promotion or an eviction) may
                        # still be the target of a copy queued before this read.
                        after=self._host.fence(),
                        tag=_Read(todo, slots, rows),
                    )
                except Exception:
                    logger.exception("SSD load submission failed")
                    for block in todo:
                        block.reading = False
                    for row in rows:
                        self._host.release(row)
        finally:
            self._flush(out, [])

    def _live(self, run: list[_Block]) -> list[_Block]:
        """The part of ``run`` that is still cached and still ends at a snapshot:
        evictions since the lookup (``_make_room``) may have dropped blocks and
        their SSD slots."""
        end = 0
        for i, block in enumerate(run):
            if self._blocks.get(block.block_hash) is not block or (
                block.tier is Tier.SSD and block.ssd_slot < 0
            ):
                break
            if block.snapshot >= 0:
                end = i + 1
        return run[:end]

    def _read_done(self, ticket: Ticket, out: _Evicted) -> None:
        assert self._host is not None and self._ssd is not None
        read: _Read = ticket.tag
        failed: list[_Block] = []
        for block, slot, row in zip(read.blocks, read.slots, read.rows):
            block.reading = False
            if self._blocks.get(block.block_hash) is not block:
                if block.ssd_slot >= 0:
                    self._ssd.release(block.ssd_slot)
                    block.ssd_slot = -1
                self._host.release(row)
            elif block.tier is not Tier.SSD or block.ssd_slot != slot:
                self._host.release(row)
                self._maybe_push(block)
            elif ticket.error is not None:
                self._host.release(row)
                failed.append(block)
            elif block.parent is not None and block.parent.tier is Tier.SSD:
                self._host.release(row)
                self._maybe_push(block)
            else:
                block.host_slot, block.ssd_warm = row, True
                block.last_used = next(self._clock)
                self._move(block, Tier.HOST)
                self._maybe_push(block)
                self.ssd_loads += 1
        if failed:
            self._drop_subtree(failed, out)

    def _drop_subtree(self, roots: list[_Block], out: _Evicted) -> None:
        """Drop SSD-tier blocks and everything chained behind them (leaves first):
        a block whose parent is gone could never be reached again."""
        seen: set[_Block] = set()
        stack = [(block, False) for block in reversed(roots)]
        while stack:
            block, expanded = stack.pop()
            if expanded:
                assert block.tier is Tier.SSD
                self._drop(block, out)
            elif block not in seen:
                seen.add(block)
                stack.append((block, True))
                stack.extend((kid, False) for kid in block.kids())

    def _log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_log < STATS_LOG_INTERVAL_S or not self._query_tokens:
            return
        self._last_log = now
        logger.info(
            "DeepInfer prefix cache: hit rate %.3f (%d of %d prompt tokens, %d "
            "from host), %d blocks cached on GPU, %d of %d on host, host stores "
            "%d, loads %d, re-homed %d, evictions %d",
            self._hit_tokens / self._query_tokens,
            self._hit_tokens,
            self._query_tokens,
            self.host_hit_tokens,
            len(self._blocks) - self._num_host - self._num_ssd,
            self._num_host,
            self._host.capacity if self._host else 0,
            self.host_stores,
            self.host_loads,
            self.host_rehomed,
            self.host_evictions,
        )
        if self._ssd is not None:
            logger.info(
                "DeepInfer prefix cache SSD: %d blocks only on SSD, %d of %d slots "
                "used, %d hit tokens warmed by SSD reads, stores %d, loads %d, "
                "evictions %d, errors %d",
                self._num_ssd,
                self._ssd.capacity - self._ssd.num_free,
                self._ssd.capacity,
                self.ssd_hit_tokens,
                self.ssd_stores,
                self.ssd_loads,
                self.ssd_evictions,
                self.ssd_errors,
            )

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
        pinned: list[int] = []
        for i in range(num):
            h = req.hashes[i]
            block = self._blocks.get(h)
            if block is None:
                block = _Block(h, int(table[i]), parent, i)
                self._blocks[h] = block
                self._num_unused += 1
                if parent is not None:
                    parent.children += 1
                    parent.gpu_children += 1
                    parent.add_kid(block)
                new.append(block)
                pinned.append(block.block_id)
            elif not block.on_gpu:
                self._rehome(block, int(table[i]))
                pinned.append(block.block_id)
            # Another request may have cached the same tokens in other blocks.
            if block.block_id == table[i]:
                self._acquire(req, block)
            parent = block
        assert parent is not None
        if pinned:
            self.engine.block_manager.pin_blocks(pinned)
        if new:
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
        out = _Evicted()
        kept: list[_Block] = []
        while slot is None:
            # Snapshots of blocks that only the SSD holds go first: a demoted
            # block keeps its own, so evicting those would free nothing.
            block = self._pop_ssd_snapshot() if self._ssd else None
            if block is not None:
                self._evict_ssd(block, out)
                slot = self._pool.alloc()
                continue
            # The coldest snapshots are on host; a demoted block keeps its own.
            block = self._pop_evictable(gpu=False) if self._host else None
            if block is not None:
                self._evict_host(block, out)
            else:
                block = self._pop_evictable(gpu=True)
                if block is None:
                    break
                if not self._evict_gpu(block, out, demote=False):
                    kept.append(block)
            slot = self._pool.alloc()
        self._drain_writes(out)
        self._flush(out, kept)
        return slot

    # -- references -----------------------------------------------------------

    def _acquire(self, req: _Request, block: _Block) -> None:
        if block.block_hash in req.held:
            return
        req.held[block.block_hash] = block
        if block.users == 0 and block.on_gpu:
            self._num_unused -= 1
        block.users += 1

    def _release(self, req: _Request) -> None:
        tick = next(self._clock)
        for block in reversed(req.held.values()):
            block.users -= 1
            block.last_used = tick
            if block.users == 0:
                if block.on_gpu:
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
        if block.tier is Tier.SSD:
            if not block.children and not block.reading and block.snapshot >= 0:
                self._push(self._ssd_snap_heap, block, "snap_key", None)
        elif not block.users:
            if block.tier is Tier.GPU:
                if not block.gpu_children:
                    self._push(self._heap, block, "heap_key", Tier.GPU)
            elif self._host_leaf(block):
                self._push(self._host_heap, block, "heap_key", Tier.HOST)
        if block.ssd_state is SsdState.DONE and self._ssd_victim(block):
            self._push(self._ssd_heap, block, "ssd_key", None)

    @staticmethod
    def _host_leaf(block: _Block) -> bool:
        """A host block that can leave host: nothing above it needs it (its
        children, if any, are on SSD) and no write reads its row."""
        return (
            not block.gpu_children
            and not block.host_children
            and block.ssd_state is not SsdState.WRITING
        )

    @staticmethod
    def _ssd_victim(block: _Block) -> bool:
        """A complete SSD copy whose slot can be reclaimed, unless a read uses it:
        a backup when no child is on SSD (the SSD children would go with the
        block once it left host), an SSD-only block when it is a leaf."""
        if block.reading:
            return False
        if block.tier is Tier.SSD:
            return not block.children
        return block.children == block.gpu_children + block.host_children

    def _push(
        self,
        heap: list[tuple[int, int, _Block]],
        block: _Block,
        attr: str,
        tier: Optional[Tier],
    ) -> None:
        # A leaf that cannot serve a hit (no snapshot, nothing cached below)
        # goes first.
        key = block.last_used if block.snapshot >= 0 or block.children else 0
        setattr(block, attr, key)
        heapq.heappush(heap, (key, next(self._tiebreak), block))
        if len(heap) > 2 * len(self._blocks) + 1024:
            heap[:] = [
                (k, next(self._tiebreak), b)
                for b in self._blocks.values()
                if (k := getattr(b, attr)) is not None
                and (tier is None or b.tier is tier)
            ]
            heapq.heapify(heap)

    def _pop_evictable(self, *, gpu: bool) -> Optional[_Block]:
        heap = self._heap if gpu else self._host_heap
        tier = Tier.GPU if gpu else Tier.HOST
        while heap:
            key, _, block = heapq.heappop(heap)
            if key != block.heap_key or block.tier is not tier:
                continue
            block.heap_key = None
            if block.users == 0 and (
                not block.gpu_children if gpu else self._host_leaf(block)
            ):
                return block
        return None

    def _pop_ssd(self) -> Optional[_Block]:
        """The coldest complete SSD copy whose slot can be reclaimed."""
        heap = self._ssd_heap
        while heap:
            key, _, block = heapq.heappop(heap)
            if key != block.ssd_key:
                continue
            block.ssd_key = None
            if block.ssd_state is SsdState.DONE and self._ssd_victim(block):
                return block
        return None

    def _pop_ssd_snapshot(self) -> Optional[_Block]:
        """The coldest SSD-only leaf that holds a snapshot."""
        heap = self._ssd_snap_heap
        while heap:
            key, _, block = heapq.heappop(heap)
            if key != block.snap_key:
                continue
            block.snap_key = None
            if (
                block.tier is Tier.SSD
                and not block.children
                and not block.reading
                and block.snapshot >= 0
            ):
                return block
        return None

    def _evict_gpu(self, block: _Block, out: _Evicted, *, demote: bool = True) -> bool:
        """Take an unheld GPU leaf off the GPU: to the host tier when it can
        still serve a hit, else to the SSD tier when a complete copy is there,
        else out of the cache. False (nothing changed) when it can do none: the
        blocks below it need it."""
        slot = None
        servable = block.snapshot >= 0 or block.children
        if demote and self._host is not None and servable:
            slot = self._host_slot(out)
        # The SSD copy is the block: no I/O. Not under a child on host, which
        # would then sit above its parent; not for a snapshot slot (it is kept).
        to_ssd = (
            slot is None
            and demote
            and servable
            and block.ssd_state is SsdState.DONE
            and not block.host_children
        )
        if slot is None and block.children and not to_ssd:
            return False
        self._num_unused -= 1
        out.ids.append(block.block_id)
        if slot is not None:
            out.stores.append((block.block_id, slot))
            block.block_id, block.host_slot = -1, slot
            self._move(block, Tier.HOST)
            self.host_stores += 1
            if self._ssd is not None:
                self._write_through(block, slot, out)
            self._maybe_push(block)
        elif to_ssd:
            block.block_id = -1
            self._move(block, Tier.SSD)
            self._maybe_push(block)
        else:
            self._drop(block, out)
        return True

    def _flush(self, out: _Evicted, kept: list[_Block]) -> None:
        event = None
        if out.stores:
            assert self._host is not None
            # Before the blocks are freed: the copy is queued ahead of their reuse.
            event = self._host.store(
                [i for i, _ in out.stores], [s for _, s in out.stores]
            )
        if out.writes:
            assert self._host is not None and self._ssd is not None
            # Each write waits for the copy that filled its row (a queued one
            # waits for every copy queued so far).
            if event is None:
                event = self._host.fence()
            for block in out.writes:
                assert block.write is not None
                self._ssd.store(
                    [self._host.row(block.write.row)],
                    [block.ssd_slot],
                    after=event,
                    tag=block.write,
                )
            out.writes.clear()
        if out.ids:
            self.engine.block_manager.unpin_blocks(out.ids)
        if out.removed:
            self._events.append(KVEvent(kind="removed", block_hashes=out.removed))
        for block in kept:
            self._maybe_push(block)

    def _make_room(self, want: int, at_least: int) -> int:
        """Evict LRU blocks until ``want`` blocks are free; do nothing when even
        evicting every unheld block would leave fewer than ``at_least``."""
        self._poll_ssd()
        block_manager = self.engine.block_manager
        free = int(block_manager.get_num_free_blocks())
        if free >= want:
            return 0
        if free + self._num_unused < at_least:
            self._reap()
            if free + self._num_unused < at_least:
                return 0
        out = _Evicted()
        kept: list[_Block] = []
        # Each unheld block's only reference is the cache's pin.
        while free + len(out.ids) < want:
            block = self._pop_evictable(gpu=True)
            if block is None:
                break
            if not self._evict_gpu(block, out):
                kept.append(block)
        self._drain_writes(out)
        self._flush(out, kept)
        return int(block_manager.get_num_free_blocks()) - free
