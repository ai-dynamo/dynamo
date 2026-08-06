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

Two mechanisms, because the two directions of staleness are not symmetric:

  * ADDs are captured by periodically **snapshotting** the index. A block cached since
    the last snapshot is simply absent, which costs a MISS and a recompute.
  * DELs are **streamed immediately** on eviction. A block evicted since the last
    snapshot is still *present* in it, and replaying that entry is a HIT on memory that
    has since been overwritten. Losing a deletion is a correctness bug, so deletions
    cannot wait for the next snapshot.

The snapshot is taken after the step barrier, so every entry in it describes a write that
has completed. It also carries the layout identity it belongs to, and is refused on
mismatch -- a snapshot only describes the pages it was taken against.

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

logger = logging.getLogger(__name__)

_MAGIC = b"GMSKVIX1"
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


def _del_path() -> str:
    return _path() + ".del"


# ---------------------------------------------------------------------------
# on-disk form
# ---------------------------------------------------------------------------


def write_snapshot(block_pool, layout_id: str, path: str) -> int:
    """Write a point-in-time picture of the index, then swap it in atomically.

    Reads ``BlockPool.blocks`` -- a public list whose entries carry everything needed --
    rather than the index container, so this depends on no private vLLM structure.
    """
    body = bytearray()
    count = 0
    for blk in block_pool.blocks:
        bh = blk.block_hash
        if bh is None:
            continue
        h = bytes(bh)
        body += struct.pack(
            "<HiI", len(h), blk.block_id, blk.block_hash_num_tokens or 0
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


def read_snapshot(path: str) -> tuple[str, list[tuple[bytes, int, int]]]:
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
        hlen, block_id, ntok = struct.unpack_from("<HiI", data, off)
        off += 10
        out.append((data[off : off + hlen], block_id, ntok))
        off += hlen
    return layout_id, out


def append_deletions(hashes: list[bytes]) -> None:
    """Record evictions immediately. Opened per call so the bytes reach the page cache
    without us holding a descriptor across the engine's lifetime."""
    if not hashes:
        return
    buf = bytearray()
    for h in hashes:
        buf += struct.pack("<H", len(h)) + h
    with open(_del_path(), "ab") as f:
        f.write(buf)


def read_deletions(path: str) -> set[bytes]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return set()
    out, off = set(), 0
    while off + 2 <= len(data):
        (hlen,) = struct.unpack_from("<H", data, off)
        off += 2
        if off + hlen > len(data):
            break  # torn tail from a crash mid-append; the rest is still usable
        out.add(data[off : off + hlen])
        off += hlen
    return out


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def maybe_snapshot(scheduler) -> None:
    """Snapshot the index if the interval has elapsed. Called at the step barrier.

    Runs after the step's model output exists, so every entry describes a write that has
    landed: the KV writes precede the sampler on the same CUDA stream, and having the
    output in hand means the stream reached that point.
    """
    global _last_snapshot
    if not is_enabled():
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
        # Deletions before this snapshot are already reflected in it (an evicted block
        # has no hash), so the accumulated list can be dropped. Truncating after the
        # write is safe because both run on the scheduler thread, so no eviction can
        # interleave.
        open(_del_path(), "wb").close()
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

    snap_layout_id, records = read_snapshot(snap_path)
    if snap_layout_id != layout_id:
        # The snapshot describes a different set of pages. Replaying it would point
        # hashes at memory that never held those prefixes.
        logger.warning(
            "[GMS] KV index snapshot is for layout %s but this engine adopted %s; "
            "refusing to replay",
            snap_layout_id[:16],
            layout_id[:16],
        )
        return 0

    deleted = read_deletions(_del_path())
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    null_id = block_pool.null_block.block_id
    installed = []
    for h, block_id, ntok in records:
        if h in deleted:
            continue
        if block_id == null_id or block_id < 0 or block_id >= len(block_pool.blocks):
            continue
        block = block_pool.blocks[block_id]
        if block.block_hash is not None or block.ref_cnt != 0:
            continue  # already in use on this engine; never disturb it
        block_pool._insert_block_hash(BlockHashWithGroupId(h), block, ntok)
        installed.append(block)

    _requeue_to_tail(block_pool, installed)
    logger.info(
        "[GMS] KV index replayed: %d entries installed (%d in snapshot, %d deleted since)",
        len(installed),
        len(records),
        len(deleted),
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

    Hooked on ``resume_scheduler`` because that is the one point satisfying all three
    requirements at any TP: it runs in the process that owns the ``BlockPool``, *after*
    ``model_executor.wake_up()`` has returned (a blocking collective, so every rank has
    finished reattaching its shard), and *before* scheduling resumes.

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
            engine_core.scheduler._gms_kv_layout_id = "|".join(h for _, h in state)
    except Exception as e:
        logger.warning("[GMS] could not record KV layout id (%s)", e)


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------


def enable_kv_index_persistence() -> None:
    """Install the hooks. Invoked by vLLM as a ``vllm.general_plugins`` entry point.

    ``EngineCore.__init__`` loads general plugins, so this lands in the scheduler's
    process regardless of TP -- unlike importing the GMS worker module, which at TP>1
    happens only in the worker child processes, leaving the scheduler unpatched.
    """
    if not is_enabled():
        return

    _patch_eviction()
    _patch_step_barrier()
    _patch_wake()
    logger.info(
        "[GMS] KV index persistence enabled (snapshot=%s, interval=%.1fs)",
        _path(),
        _SNAPSHOT_INTERVAL_S,
    )


def _patch_eviction() -> None:
    """Stream deletions. The only capture that cannot wait for the next snapshot."""
    from vllm.v1.core.block_pool import BlockPool

    if getattr(BlockPool, "_gms_kv_patched", False):
        return
    orig_evict = BlockPool._maybe_evict_cached_block

    def _maybe_evict_cached_block(self, block, *args, **kwargs):
        # Read the hashes before delegating: the original resets them.
        hashes = []
        try:
            if block.block_hash is not None:
                hashes.append(bytes(block.block_hash))
            hashes.extend(
                bytes(h)
                for h in self.cached_block_hashes_by_block.get(block.block_id, ())
            )
        except Exception:
            hashes = []
        evicted = orig_evict(self, block, *args, **kwargs)
        if evicted and hashes:
            try:
                append_deletions(hashes)
            except Exception as e:
                logger.warning("[GMS] KV index deletion record failed: %s", e)
        return evicted

    BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block
    BlockPool._gms_kv_patched = True


def _patch_step_barrier() -> None:
    """Snapshot after a step's output exists, so every entry describes a landed write."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_gms_kv_patched", False):
        return
    orig_update = Scheduler.update_from_output

    def update_from_output(self, *args, **kwargs):
        out = orig_update(self, *args, **kwargs)
        maybe_snapshot(self)
        return out

    Scheduler.update_from_output = update_from_output
    Scheduler._gms_kv_patched = True


def _patch_wake() -> None:
    """Replay on takeover, and learn our layout identity on a fresh start."""
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_gms_kv_patched", False):
        return
    orig_resume = EngineCore.resume_scheduler

    def resume_scheduler(self, *args, **kwargs):
        try:
            if not replay_after_wake(self):
                # Not a takeover: still record the layout id so our own snapshots carry it.
                note_layout_id(self)
        except Exception as e:
            logger.warning("[GMS] KV index wake hook failed: %s", e)
        return orig_resume(self, *args, **kwargs)

    EngineCore.resume_scheduler = resume_scheduler
    EngineCore._gms_kv_patched = True
