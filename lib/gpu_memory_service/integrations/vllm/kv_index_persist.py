# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry the vLLM prefix-cache index across a shadow failover.

Committing the KV layout (see the GMS server's ``commit_layout``) makes the *bytes*
outlive an engine. A prefix-cache HIT also needs the *index* -- ``block_hash ->
block_id`` -- which lives in the scheduler's ``BlockPool`` in process RAM and cannot be
regenerated from the bytes, because the hash is over token ids rather than content. So a
standby that adopts a layout has correct KV it cannot find.

This carries the index the same way the layout is carried: written by the active engine,
read by whoever adopts next.

An index entry is a name-to-location binding that carries an unstated claim -- *the bytes
at this block are the bytes this hash denotes*. Nothing in the memory records the hash, so
the claim can quietly stop holding. Replaying a snapshot re-asserts every binding in it,
which is only sound with evidence that the claim still holds. Two artifacts provide it:

  * a periodic **snapshot** of the index, which proposes bindings, and
  * a per-block **generation counter**, which vetoes them.

A counter is bumped whenever a block leaves the free queue, which is the one gate a block
must pass through to become writable. The snapshot records each block's generation
alongside its hash, and replay installs a record only if the counter has not moved since.
So the snapshot is allowed to be stale -- a block cached after it was taken is merely
absent, costing a MISS -- while a block reused after it was taken is refused. Loss of the
counters, of their file, or of a matching header installs nothing rather than everything.

Snapshots are taken at the step barrier. vLLM only hashes prefixes whose tokens are
confirmed computed (under async scheduling it explicitly lags by the in-flight count), and
a hashed prefix is only ever appended to, never rewritten in place. The bump precedes the
forward pass in the same step on the same thread, so the blocks a step could touch are
always a subset of the blocks it already bumped.

The snapshot also carries the layout identity it belongs to and is refused on mismatch.
That gate has to come first: ``block_id`` is an index into KV tensors of a particular
shape, so under a different geometry the same number names different bytes, and the
generation check would be comparing the wrong entries.

Determinism: both engines must hash identically, so ``PYTHONHASHSEED`` has to be pinned.
Otherwise vLLM's ``NONE_HASH`` (the root of the block-hash chain) is ``os.urandom(32)``
per process and no replayed hash ever matches. That degrades to a MISS rather than a
wrong answer, but it silently removes the entire benefit.

Opt in with ``GMS_KV_INDEX_PATH``; unset, everything here self-disables.
"""

from __future__ import annotations

import logging
import os
import struct
import time

import numpy as np

logger = logging.getLogger(__name__)

_MAGIC = b"GMSKVIX2"
_GEN_MAGIC = b"GMSKVGN1"
# The counter array starts here so it is page- and word-aligned regardless of how long the
# layout id is.
_GEN_DATA_OFFSET = 4096
_GEN_DTYPE = np.uint64

# Seconds between index snapshots. Cost is O(num_gpu_blocks): ~2.6 ms at 8k blocks and
# ~8 ms at 40k, so a snapshot per second is well under 1% of serving time. Raising it
# only widens the window of *additions* that are lost on a crash, which costs recompute.
_SNAPSHOT_INTERVAL_S = float(os.getenv("GMS_KV_SNAPSHOT_INTERVAL", "1.0") or 1.0)

_log_path: str | None = None
_last_snapshot = 0.0


def _path() -> str | None:
    global _log_path
    if _log_path is None:
        _log_path = os.getenv("GMS_KV_INDEX_PATH")
    return _log_path


def is_enabled() -> bool:
    return bool(_path())


def _gen_path() -> str:
    return _path() + ".gen"


# ---------------------------------------------------------------------------
# generations
# ---------------------------------------------------------------------------

# A shared mapping rather than a file we append to: a store to a dirty page needs no
# syscall, cannot tear, and survives the process, which is the only durability this needs.
# The engine's own pages die with the node, and so does the GMS daemon holding them, so
# there is nothing to reattach after a node loss and nothing for an ``fsync`` to buy.
_gen: np.ndarray | None = None
_gen_layout: str | None = None
# Set when a block was handed out while the array was unavailable. The counters can no
# longer account for every reuse, so nothing may be replayed against them.
_gen_blind = False


def open_generations(num_blocks: int, layout_id: str = "") -> bool:
    """Map the counter array, creating or resetting it if it does not describe this pool.

    ``layout_id`` is empty on a fresh engine, which does not learn its layout identity
    until the workers have adopted. Passing it later via :func:`bind_generations` stamps
    the header without disturbing counters that a previous engine left behind.
    """
    global _gen, _gen_layout, _gen_blind
    if not is_enabled():
        return False
    path = _gen_path()
    try:
        header = _read_gen_header(path)
    except Exception:
        header = None

    if header is None or header[1] != num_blocks:
        _create_generations(path, num_blocks, layout_id)
        header = (layout_id, num_blocks)
    elif layout_id and header[0] and header[0] != layout_id:
        # Counters describing a different set of pages. Their values mean nothing here.
        _create_generations(path, num_blocks, layout_id)
        header = (layout_id, num_blocks)

    _gen = np.memmap(
        path, dtype=_GEN_DTYPE, mode="r+", offset=_GEN_DATA_OFFSET, shape=(num_blocks,)
    )
    _gen_layout = header[0]
    _gen_blind = False
    return True


def bind_generations(layout_id: str) -> None:
    """Stamp the layout identity onto an array that was opened before it was known."""
    global _gen_layout
    if _gen is None or not layout_id:
        return
    if _gen_layout == layout_id:
        return
    if _gen_layout:
        # Opened against another layout; open_generations resets rather than reinterprets.
        open_generations(len(_gen), layout_id)
        return
    _write_gen_header(_gen_path(), layout_id, len(_gen))
    _gen_layout = layout_id


def bump(block_ids) -> None:
    """Record that these blocks became writable. Ids are distinct, so no accumulation."""
    global _gen_blind
    if _gen is None:
        if is_enabled():
            _gen_blind = True
        return
    try:
        _gen[block_ids] += 1
    except Exception as e:  # never let bookkeeping break serving
        _gen_blind = True
        logger.warning("[GMS] KV generation bump failed, replay disabled: %s", e)


def bump_all() -> None:
    """Invalidate every record. Used when the index is dropped without the bytes moving.

    ``reset_prefix_cache`` is the case that matters: it exists so an RLHF flow can drop a
    cache computed under stale weights. The bytes are untouched, so every counter still
    matches and a snapshot taken before the reset would replay cleanly -- reinstating
    exactly the cache the operator asked to discard.
    """
    if _gen is None:
        return
    _gen[:] += 1


def _read_gen_header(path: str) -> tuple[str, int] | None:
    with open(path, "rb") as f:
        head = f.read(_GEN_DATA_OFFSET)
    if len(head) < _GEN_DATA_OFFSET or head[: len(_GEN_MAGIC)] != _GEN_MAGIC:
        return None
    off = len(_GEN_MAGIC)
    lid_len, num_blocks = struct.unpack_from("<HI", head, off)
    off += 6
    return head[off : off + lid_len].decode(), num_blocks


def _write_gen_header(path: str, layout_id: str, num_blocks: int) -> None:
    lid = layout_id.encode()
    header = _GEN_MAGIC + struct.pack("<HI", len(lid), num_blocks) + lid
    header += b"\0" * (_GEN_DATA_OFFSET - len(header))
    with open(path, "r+b") as f:
        f.write(header)


def _create_generations(path: str, num_blocks: int, layout_id: str) -> None:
    lid = layout_id.encode()
    header = _GEN_MAGIC + struct.pack("<HI", len(lid), num_blocks) + lid
    header += b"\0" * (_GEN_DATA_OFFSET - len(header))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(b"\0" * (num_blocks * np.dtype(_GEN_DTYPE).itemsize))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# on-disk form
# ---------------------------------------------------------------------------

_REC = struct.Struct("<HiIQ")


def write_snapshot(block_pool, layout_id: str, path: str) -> int:
    """Write a point-in-time picture of the index, then swap it in atomically.

    Reads ``BlockPool.blocks`` -- a public list whose entries carry everything needed --
    rather than the index container, so this depends on no private vLLM structure.
    """
    if _gen is None:
        raise RuntimeError("generation counters unavailable")
    gens = _gen  # local: one bounds-checked numpy read per record

    body = bytearray()
    count = 0
    for blk in block_pool.blocks:
        bh = blk.block_hash
        if bh is None:
            continue
        h = bytes(bh)
        body += _REC.pack(
            len(h),
            blk.block_id,
            blk.block_hash_num_tokens or 0,
            int(gens[blk.block_id]),
        )
        body += h
        count += 1

    lid = layout_id.encode()
    header = _MAGIC + struct.pack("<HI", len(lid), count) + lid

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    # Atomic swap: a reader either sees the whole previous snapshot or the whole new
    # one, never a partial write, so no torn reads and no reader-side locking.
    os.replace(tmp, path)
    return count


def read_snapshot(path: str) -> tuple[str, list[tuple[bytes, int, int, int]]]:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < len(_MAGIC) + 6 or data[: len(_MAGIC)] != _MAGIC:
        raise ValueError("not a GMS KV index snapshot")
    off = len(_MAGIC)
    lid_len, count = struct.unpack_from("<HI", data, off)
    off += 6
    layout_id = data[off : off + lid_len].decode()
    off += lid_len

    out = []
    for _ in range(count):
        hlen, block_id, ntok, gen = _REC.unpack_from(data, off)
        off += _REC.size
        out.append((data[off : off + hlen], block_id, ntok, gen))
        off += hlen
    return layout_id, out


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def maybe_snapshot(scheduler) -> None:
    """Snapshot the index if the interval has elapsed. Called at the step barrier."""
    global _last_snapshot
    if not is_enabled() or _gen is None:
        return

    # Flush unconditionally once the engine goes quiescent. Snapshots ride on steps, and
    # steps only happen while there is work, so an engine that caches a prefix and then
    # idles would otherwise never persist it -- which is exactly the state a failover
    # finds it in. This fires once per idle transition, not repeatedly.
    try:
        idle = not scheduler.has_requests()
    except Exception:
        idle = False

    now = time.monotonic()
    if not idle and now - _last_snapshot < _SNAPSHOT_INTERVAL_S:
        return
    _last_snapshot = now

    layout_id = getattr(scheduler, "_gms_kv_layout_id", None)
    if not layout_id:
        return  # no committed layout to describe

    try:
        pool = scheduler.kv_cache_manager.block_pool
        n = write_snapshot(pool, layout_id, _path())
        logger.debug("[GMS] KV index snapshot: %d entries", n)
    except Exception as e:  # never let persistence break serving
        logger.warning("[GMS] KV index snapshot failed: %s", e)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def replay_index(block_pool, layout_id: str) -> int:
    """Install a prior engine's index onto the layout this engine just adopted."""
    snap_path = _path()
    if not snap_path or not os.path.exists(snap_path):
        logger.info("[GMS] no KV index snapshot at %s; nothing to replay", snap_path)
        return 0

    if _gen is None or _gen_blind or _gen_layout != layout_id:
        # Without counters covering every handout there is no evidence any record still
        # holds, and a record is only trustworthy while some evidence vetoes it.
        logger.warning(
            "[GMS] KV generation counters unusable (open=%s blind=%s layout=%s); "
            "refusing to replay",
            _gen is not None,
            _gen_blind,
            (_gen_layout or "")[:16],
        )
        return 0

    snap_layout_id, records = read_snapshot(snap_path)
    if snap_layout_id != layout_id:
        # The snapshot describes a different set of pages. ``block_id`` indexes tensors of
        # a particular shape, so replaying it would read adopted memory under the wrong
        # geometry: real bytes, wrong ruler.
        logger.warning(
            "[GMS] KV index snapshot is for layout %s but this engine adopted %s; "
            "refusing to replay",
            snap_layout_id[:16],
            layout_id[:16],
        )
        return 0

    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    null_id = block_pool.null_block.block_id
    installed = []
    reused = 0
    for h, block_id, ntok, gen in records:
        if block_id == null_id or block_id < 0 or block_id >= len(block_pool.blocks):
            continue
        if int(_gen[block_id]) != gen:
            # Handed out since the snapshot, so its bytes are someone else's now.
            reused += 1
            continue
        block = block_pool.blocks[block_id]
        if block.block_hash is not None or block.ref_cnt != 0:
            continue  # already in use on this engine; never disturb it
        block_pool._insert_block_hash(BlockHashWithGroupId(h), block, ntok)
        installed.append(block)

    _requeue_to_tail(block_pool, installed)
    logger.info(
        "[GMS] KV index replayed: %d entries installed (%d in snapshot, %d reused since)",
        len(installed),
        len(records),
        reused,
    )
    return len(installed)


def _requeue_to_tail(block_pool, blocks: list) -> None:
    """Move newly-cached blocks to the tail of the free queue.

    vLLM keeps "hand out uncached blocks first" purely by queue *position*: ``free_blocks``
    prepends uncached and appends cached, and ``get_new_blocks`` pops the head. Marking a
    block cached does not move it, so on a freshly woken standby -- where every block is
    unhashed and queued in id order -- the reused prefix would sit at the head and the
    next request served would be handed its blocks and overwrite them.

    Appended in descending block id so the prefix's last blocks are evicted first,
    leaving a usable leading prefix, mirroring how vLLM frees a request's blocks.
    """
    moved = 0
    for block in sorted(blocks, key=lambda b: b.block_id, reverse=True):
        try:
            block_pool.free_block_queue.remove(block)
            block_pool.free_block_queue.append(block)
            moved += 1
        except Exception as e:
            logger.warning("[GMS] KV index requeue failed: %s", e)
            break
    if moved:
        logger.debug("[GMS] KV index: requeued %d blocks to free-queue tail", moved)


def replay_after_wake(engine_core) -> bool:
    """Replay before the engine can schedule anything.

    Hooked on ``EngineCore.wake_up`` because that is the one point satisfying all three
    requirements: it runs in the process that owns the ``BlockPool``, *after*
    ``model_executor.wake_up()`` has returned (a blocking collective, so every rank has
    finished reattaching its shard), and *before* scheduling resumes.

    Not ``resume_scheduler``, which looks like the natural place and is not reliable:
    ``EngineCoreProc`` overrides it and returns early while ``engines_running``, so the
    base method the hook would sit on is simply not reached. That is a race rather than a
    configuration, and it silently skips the replay.

    Deferring past this point is a correctness bug rather than a latency one: until the
    index is replayed every adopted block still looks free and unhashed, so the first
    request served is handed the reused prefix's own blocks and overwrites them.
    """
    if not is_enabled():
        return False

    try:
        state = engine_core.collective_rpc("gms_kv_takeover_state")
    except Exception as e:
        logger.warning("[GMS] could not query workers for KV takeover state (%s)", e)
        return False

    # Every rank must have adopted. The index is engine-wide but the bytes are per-rank,
    # so a partial adoption would be correct for the ranks that adopted and garbage for
    # the one that did not -- silent, partial corruption.
    if not state or not all(adopted for adopted, _ in state):
        return False

    # The layout identity is the per-rank hashes in rank order: each rank adopted its own
    # device's allocations, and the snapshot is only valid against all of them together.
    layout_id = "|".join(h for _, h in state)

    try:
        scheduler = engine_core.scheduler
        scheduler._gms_kv_layout_id = layout_id
        bind_generations(layout_id)
        replay_index(scheduler.kv_cache_manager.block_pool, layout_id)
    except Exception as e:
        logger.warning("[GMS] KV index replay failed: %s", e)
        return False
    return True


def note_layout_id(engine_core) -> None:
    """Record this engine's layout identity so snapshots can be stamped with it."""
    if not is_enabled():
        return
    try:
        state = engine_core.collective_rpc("gms_kv_takeover_state")
        if state:
            layout_id = "|".join(h for _, h in state)
            engine_core.scheduler._gms_kv_layout_id = layout_id
            bind_generations(layout_id)
    except Exception as e:
        logger.warning("[GMS] could not record KV layout id (%s)", e)


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------


def enable_kv_index_persistence() -> None:
    """Install the hooks. Invoked by vLLM as a ``vllm.general_plugins`` entry point.

    ``EngineCore.__init__`` loads general plugins before it builds the scheduler, and it
    does so in the scheduler's own process regardless of TP -- unlike importing the GMS
    worker module, which at TP>1 happens only in the worker children.
    """
    if not is_enabled():
        return

    _install_scheduler_subclass()
    _patch_wake()
    logger.info(
        "[GMS] KV index persistence enabled (snapshot=%s, interval=%.1fs)",
        _path(),
        _SNAPSHOT_INTERVAL_S,
    )


def _install_scheduler_subclass() -> None:
    """Derive from whatever scheduler vLLM chose, rather than replacing it.

    Setting ``scheduler_config.scheduler_cls`` would work but costs more than it looks:
    ``get_scheduler_cls`` only returns ``AsyncScheduler`` while that field is unset, so
    naming a class here silently disables async scheduling. Wrapping the resolver keeps
    whatever base the config selected -- async, sync, or someone else's subclass.
    """
    from vllm.config.scheduler import SchedulerConfig

    if getattr(SchedulerConfig, "_gms_kv_patched", False):
        return
    orig_get = SchedulerConfig.get_scheduler_cls
    derived: dict[type, type] = {}

    def get_scheduler_cls(self):
        base = orig_get(self)
        if base not in derived:
            derived[base] = _make_scheduler_cls(base)
        return derived[base]

    SchedulerConfig.get_scheduler_cls = get_scheduler_cls
    SchedulerConfig._gms_kv_patched = True


def _make_scheduler_cls(base: type) -> type:
    class GmsKvScheduler(base):  # type: ignore[misc, valid-type]
        """Snapshots the prefix index and keeps the generation counters honest."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                _install_block_pool_hooks(self.kv_cache_manager.block_pool)
            except Exception as e:
                logger.warning("[GMS] KV generation hooks not installed: %s", e)

        def update_from_output(self, *args, **kwargs):
            out = super().update_from_output(*args, **kwargs)
            maybe_snapshot(self)
            return out

    GmsKvScheduler.__name__ = f"GmsKv{base.__name__}"
    GmsKvScheduler.__qualname__ = GmsKvScheduler.__name__
    return GmsKvScheduler


def _install_block_pool_hooks(pool) -> None:
    """Bump on handout, invalidate on cache reset.

    The bump belongs on the free queue rather than on ``get_new_blocks``: leaving the
    queue is what makes a block writable, and ``get_new_blocks`` is not the only caller --
    sink blocks are popped directly by the sliding-window manager. ``remove`` is left
    alone deliberately, since ``touch`` uses it to reclaim a block whose contents are kept.
    """
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue

    queue_cls = type(pool.free_block_queue)
    if not getattr(queue_cls, "_gms_kv_derived", False):

        class _GmsFreeQueue(queue_cls):  # type: ignore[misc, valid-type]
            _gms_kv_derived = True

            def popleft(self):
                block = super().popleft()
                bump([block.block_id])
                return block

            def popleft_n(self, n):
                blocks = super().popleft_n(n)
                if blocks:
                    bump([b.block_id for b in blocks])
                return blocks

        assert issubclass(queue_cls, FreeKVCacheBlockQueue)
        pool.free_block_queue.__class__ = _GmsFreeQueue

    pool_cls = type(pool)
    if not getattr(pool_cls, "_gms_kv_derived", False):

        class _GmsBlockPool(pool_cls):  # type: ignore[misc, valid-type]
            _gms_kv_derived = True

            def reset_prefix_cache(self) -> bool:
                ok = super().reset_prefix_cache()
                if ok:
                    bump_all()
                return ok

        assert issubclass(pool_cls, BlockPool)
        pool.__class__ = _GmsBlockPool

    open_generations(len(pool.blocks))


def _patch_wake() -> None:
    """Replay on takeover, and learn our layout identity on a fresh start."""
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_gms_kv_patched", False):
        return
    orig_wake_up = EngineCore.wake_up

    def wake_up(self, *args, **kwargs):
        out = orig_wake_up(self, *args, **kwargs)
        try:
            # A partial wake leaves some allocations asleep, so the KV may not be resident
            # yet. This is the same condition vLLM uses before resuming scheduling.
            if self.model_executor.is_sleeping:
                return out
            if not replay_after_wake(self):
                # Not a takeover: still record the layout id so our own snapshots carry it.
                note_layout_id(self)
        except Exception as e:
            logger.warning("[GMS] KV index wake hook failed: %s", e)
        return out

    EngineCore.wake_up = wake_up
    EngineCore._gms_kv_patched = True
