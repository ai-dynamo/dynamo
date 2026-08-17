# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry the vLLM prefix-cache index across an engine failover.

GMS can already make the KV *bytes* outlive the engine that wrote them
(``commit_layout()``). This carries the *index*: the ``block_hash -> block_id``
binding that lives in the scheduler's ``BlockPool`` and cannot be regenerated
from the bytes, because the hash is over token ids rather than content.

The design is a write-through mirror rather than a periodic snapshot. vLLM has
exactly three sites that change a block's label -- ``_insert_block_hash`` names
one, ``_remove_cached_block_hashes`` un-names one, ``reset_prefix_cache``
un-names all -- so wrapping those three on one object mirrors every change with
no gaps and no scan.

The invariant, from which everything else follows:

    At every instant a successor could read it, the mirror is a SUBSET of the
    labels that are actually true.

Subset means every possible error is a missing label, and a missing label is a
cache miss. No arrangement of failures yields a *wrong* label.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import struct

import numpy as np

logger = logging.getLogger(__name__)

MAGIC = b"DYNKVIX1"
VERSION = 1
HEADER_BYTES = 4096
# sha256 digest + 4-byte group id = 36; xxhash128 + 4 = 20. 40 covers both.
KEY_BYTES = 40

_OFF_VERSION = 8
_OFF_NUM_BLOCKS = 12
_OFF_COMMITTED = 16
_OFF_IDENTITY = 24
_DIGEST_BYTES = 32


def mirror_path() -> str | None:
    """Where the mirror lives, or None when the feature is off (the default)."""
    return os.environ.get("GMS_KV_INDEX_PATH") or None


# vLLM's ``sleep(level>=1)`` clears the prefix cache (core.py:874) because it
# assumes the KV is about to be discarded. Under GMS with a committed layout it
# is not -- the server keeps the pages, so the labels stay true. A reset caused
# by a sleep must therefore NOT wipe the mirror, while an RLHF weight-update
# reset must. Nothing inside reset_prefix_cache can tell them apart.
#
# The flag covers the whole sleep->wake window rather than the sleep() call:
# EngineCoreProc.pause_scheduler (core.py:1764) may defer _reset_caches into an
# idle callback that fires after sleep() has already returned.
_IN_SLEEP = False


def _in_sleep() -> bool:
    return _IN_SLEEP

# One slot per block, indexed by block_id -- a flat array, not a map.
_REC = np.dtype(
    [
        ("stamp", "<u8"),  # 0 == invalid; else the S in flight at publish
        ("num_tokens", "<u4"),  # 0 encodes None
        ("key_len", "<u4"),
        ("key", np.uint8, (KEY_BYTES,)),
        ("_rsv", "<u8"),
    ]
)
assert _REC.itemsize == 64, _REC.itemsize


def identity_digest(**fields) -> bytes:
    """Hash the 'ruler': everything that defines what a block id measures.

    A label says "block 457". That only means something if both engines measure
    blocks the same way, so a mismatch here is a refusal rather than a migration.
    """
    blob = json.dumps(fields, sort_keys=True, default=repr).encode()
    return hashlib.sha256(blob).digest()


class MirrorFile:
    """The artifact: a fixed-size shared mapping, header plus one slot per block."""

    def __init__(self, path: str, mm: mmap.mmap, num_blocks: int):
        self.path = path
        self.num_blocks = num_blocks
        self._mm = mm
        # Writable views straight onto the mapping: a publish is a store, not a
        # syscall. Nothing is fsync'd -- the pages this file describes die with
        # the node, so there is nothing an fsync would buy.
        self._committed = np.ndarray(
            (1,), dtype="<u8", buffer=mm, offset=_OFF_COMMITTED
        )
        self._rec = np.ndarray(
            (num_blocks,), dtype=_REC, buffer=mm, offset=HEADER_BYTES
        )

    # ---- header ----

    @property
    def committed(self) -> int:
        return int(self._committed[0])

    def set_committed(self, value: int) -> None:
        self._committed[0] = value

    # ---- writer ----

    def publish(self, block_id: int, key: bytes, num_tokens: int | None, stamp: int):
        """Record a label. Stamp is written LAST, always 0 -> n.

        A block can only be re-named after being retracted (``set_block_hash``
        asserts the slot is empty), so a record torn by SIGKILL always reads
        back ``stamp == 0`` and is ignored. That is why no checksum is needed.
        """
        n = len(key)
        self._rec["key"][block_id][:n] = np.frombuffer(key, dtype=np.uint8)
        self._rec["key_len"][block_id] = n
        self._rec["num_tokens"][block_id] = num_tokens or 0
        self._rec["stamp"][block_id] = stamp

    def invalidate(self, block_id: int) -> None:
        self._rec["stamp"][block_id] = 0

    def invalidate_all(self) -> None:
        self._rec["stamp"][:] = 0

    def retain_only(self, block_ids) -> None:
        """Drop every record we did not install.

        A refused record keeps a stamp that is *currently* above the watermark,
        but our fence restarts at that watermark and counts up -- so a stale
        stamp would eventually fall below it and start looking trustworthy.
        We also now own the pool, so those blocks are ours to overwrite.
        """
        keep = np.zeros(self.num_blocks, dtype=bool)
        if block_ids:
            keep[list(block_ids)] = True
        self._rec["stamp"][~keep] = 0

    # ---- reader ----

    def live_block_ids(self) -> set[int]:
        """Blocks currently claimed, ignoring the fence. For invariant tests."""
        return set(np.nonzero(self._rec["stamp"] > 0)[0].tolist())

    def trusted(self):
        """Yield ``(block_id, key, num_tokens)`` for labels the fence admits."""
        stamps = self._rec["stamp"]
        idx = np.nonzero((stamps > 0) & (stamps <= self.committed))[0]
        for block_id in idx.tolist():
            n = int(self._rec["key_len"][block_id])
            key = bytes(self._rec["key"][block_id][:n])
            nt = int(self._rec["num_tokens"][block_id])
            yield block_id, key, (nt or None)

    def close(self) -> None:
        del self._committed, self._rec
        self._mm.close()

    # ---- lifecycle ----

    @staticmethod
    def _map(path: str, num_blocks: int) -> "MirrorFile":
        f = open(path, "r+b")
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        finally:
            f.close()
        return MirrorFile(path, mm, num_blocks)

    @classmethod
    def create(cls, path: str, num_blocks: int, identity: bytes) -> "MirrorFile":
        size = HEADER_BYTES + num_blocks * _REC.itemsize
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.truncate(size)
            f.write(MAGIC)
            f.write(struct.pack("<II", VERSION, num_blocks))
            f.write(struct.pack("<Q", 0))  # committed
            f.write(identity.ljust(_DIGEST_BYTES, b"\0")[:_DIGEST_BYTES])
        os.replace(tmp, path)
        return cls._map(path, num_blocks)

    @classmethod
    def open_for_replay(
        cls, path: str, *, identity: bytes, num_blocks: int
    ) -> tuple["MirrorFile | None", str]:
        """Admit the mirror, or refuse it with a reason.

        All-or-nothing gates live here; the per-record fence is applied later by
        :meth:`trusted`. Any refusal degrades to a cold cache, never to a wrong
        answer.
        """
        if not os.path.exists(path):
            return None, "absent"
        try:
            with open(path, "rb") as f:
                head = f.read(HEADER_BYTES)
            actual_size = os.path.getsize(path)
        except OSError as e:
            logger.warning("[kvidx] cannot read mirror %s: %s", path, e)
            return None, "unreadable"

        if len(head) < HEADER_BYTES or head[: len(MAGIC)] != MAGIC:
            return None, "magic"
        (version, blocks) = struct.unpack_from("<II", head, _OFF_VERSION)
        if version != VERSION:
            return None, "version"
        if blocks != num_blocks:
            return None, "num_blocks"
        if actual_size != HEADER_BYTES + num_blocks * _REC.itemsize:
            return None, "truncated"
        if head[_OFF_IDENTITY : _OFF_IDENTITY + _DIGEST_BYTES] != identity:
            return None, "identity"
        return cls._map(path, num_blocks), "ok"


class Fence:
    """The publication fence: two integers.

    vLLM attaches labels during ``schedule()``, *before* the KV is computed, so
    the index always contains bindings whose bytes do not exist yet. Batches are
    reaped strictly FIFO, so ``update_from_output`` call #n reaps ``schedule()``
    call #n: stamping a label with S and trusting it only once E has passed is
    exactly the condition "the bytes exist".

    Self-correcting: an engine that dies mid-batch never advances E to those
    stamps, so they are permanently untrusted.
    """

    def __init__(self, mirror: MirrorFile, start: int = 0):
        self._mirror = mirror
        self.S = start
        self.E = start
        mirror.set_committed(start)

    def on_schedule(self) -> None:
        self.S += 1

    def on_update(self) -> None:
        self.E += 1
        self._mirror.set_committed(self.E)


def install_writer(pool, mirror: MirrorFile, fence: Fence) -> None:
    """Mirror every label change on this ``BlockPool`` instance.

    A runtime subclass swap rather than a patch on the class: it composes with
    whatever ``BlockPool`` subclass is already in play, and it is scoped to the
    one object we were handed.
    """
    if getattr(pool, "_dyn_mirrored", False):
        return
    base = type(pool)

    def _insert_block_hash(self, key, block, num_tokens):
        # vLLM first: publishing after means a crash mid-way simply never
        # recorded the label.
        base._insert_block_hash(self, key, block, num_tokens)
        label = block.block_hash
        if label is None or label != key:
            # Either an early return declined the insert, or this block already
            # carried a different primary name (out of scope: >1 KV cache
            # group). Never guess -- retract and stay a subset.
            self._dyn_mirror.invalidate(block.block_id)
            self._dyn_out_of_scope += 1
            return
        self._dyn_mirror.publish(
            block.block_id, bytes(label), block.block_hash_num_tokens, self._dyn_fence.S
        )

    def _remove_cached_block_hashes(self, block):
        # Retract first: a crash mid-way leaves us with fewer claims, not more.
        self._dyn_mirror.invalidate(block.block_id)
        return base._remove_cached_block_hashes(self, block)

    def reset_prefix_cache(self):
        # Only react when vLLM says it actually happened: the reset refuses
        # whenever any block is in use, and dropping the mirror on a refusal
        # would zero it on every poll of a busy engine.
        ok = base.reset_prefix_cache(self)
        if ok and not _in_sleep():
            # Outside sleep this means the operator dropped the index on purpose
            # -- an RLHF weight update -- so the labels are genuinely dead.
            self._dyn_mirror.invalidate_all()
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
    pool._dyn_mirror = mirror
    pool._dyn_fence = fence
    pool._dyn_out_of_scope = 0
    pool._dyn_mirrored = True


def replay(pool, mirror: MirrorFile) -> list:
    """Install a predecessor's labels onto the pool this engine just adopted."""
    from vllm.v1.core.kv_cache_utils import get_group_id

    installed = []
    for block_id, key, num_tokens in mirror.trusted():
        if block_id <= 0 or block_id >= len(pool.blocks):
            continue
        if get_group_id(key) != 0:
            continue
        block = pool.blocks[block_id]
        if block.is_null or block.ref_cnt != 0:
            continue
        if block.block_hash is not None or block.block_hash_num_tokens is not None:
            continue  # already in use on this engine; never disturb it
        block.set_block_hash(key, num_tokens=num_tokens)
        pool.cached_block_hash_to_block.insert(key, block)
        installed.append(block)

    _requeue_to_tail(pool, installed)
    logger.info("[kvidx] replayed %d labels", len(installed))
    return installed


def _probe_adopted(worker) -> bool:
    """Did THIS rank inherit the previous engine's KV pages?

    Runs in the worker process via ``collective_rpc``. Module-level so it is
    picklable to a spawned worker.

    Reads the flag ``GMSWorker.wake_up`` recorded rather than the live GMS
    grant: ``commit_layout()`` regrants a *creating* writer to ``RW_DATA`` as
    well, so by the time this runs both cases look identical from the grant.
    """
    return bool(getattr(worker, "_gms_kv_adopted", False))


def _identity(engine_core, num_blocks: int) -> bytes:
    """The ruler: everything that decides what a block id measures.

    A mismatch means two engines would read the same label as different bytes,
    which is the one failure mode here that is a wrong answer rather than a miss.
    """
    import vllm
    from vllm.v1.core import kv_cache_utils

    cfg = engine_core.vllm_config
    return identity_digest(
        vllm=vllm.__version__,
        model=cfg.model_config.model,
        block_size=cfg.cache_config.block_size,
        cache_dtype=str(cfg.cache_config.cache_dtype),
        tp=cfg.parallel_config.tensor_parallel_size,
        num_blocks=num_blocks,
        # Root of the block-hash chain: os.urandom per process unless
        # PYTHONHASHSEED is pinned, so this also catches an unpinned seed.
        none_hash=bytes(kv_cache_utils.NONE_HASH).hex(),
        hash_algo=str(cfg.cache_config.prefix_caching_hash_algo),
    )


def on_wake_up(engine_core) -> None:
    """Take over the index, then start recording. The whole integration.

    Runs after vLLM's ``wake_up`` returns -- the only point that is both after
    every rank has re-attached its KV memory (``model_executor.wake_up`` is a
    blocking collective) and before the scheduler can hand out a block.
    """
    path = mirror_path()
    if not path or engine_core.model_executor.is_sleeping:
        return

    pool = engine_core.scheduler.kv_cache_manager.block_pool
    num_blocks = len(pool.blocks)
    identity = _identity(engine_core, num_blocks)

    # Every rank, or nobody: the index is engine-wide but the bytes are
    # per-rank, so a partial adoption is correct on some ranks and garbage on
    # others -- the shape that produces plausible-looking wrong output.
    try:
        states = engine_core.collective_rpc(_probe_adopted)
        adopted = bool(states) and all(states)
    except Exception as e:
        logger.warning("[kvidx] takeover probe failed (%s); not replaying", e)
        adopted = False

    mirror, reason, installed, live, trusted = None, "not_adopted", [], 0, 0
    # vLLM's own index as we find it, BEFORE we touch anything. Sleep clears it,
    # so this must be 0 -- replay is only doing real work if it rebuilds from
    # nothing rather than riding on labels that survived in process memory.
    pool_labeled_before = sum(1 for b in pool.blocks if b.block_hash is not None)
    if adopted:
        mirror, reason = MirrorFile.open_for_replay(
            path, identity=identity, num_blocks=num_blocks
        )
        if mirror is None:
            logger.info("[kvidx] not replaying (%s); starting a fresh mirror", reason)
        else:
            # Counted before replay so a harness can tell the stages apart:
            # live==0 means the mirror was wiped, trusted==0 means the fence
            # rejected everything, installed==0 with trusted>0 means the pool
            # refused them.
            live = len(mirror.live_block_ids())
            trusted = sum(1 for _ in mirror.trusted())
            installed = replay(pool, mirror)
            mirror.retain_only({b.block_id for b in installed})

    _record_takeover(
        path,
        adopted=adopted,
        reason=reason,
        pool_labeled_before=pool_labeled_before,
        live=live if adopted and mirror is not None else 0,
        trusted=trusted if adopted and mirror is not None else 0,
        installed=len(installed),
        num_blocks=num_blocks,
        committed=mirror.committed if mirror is not None else 0,
    )

    if mirror is None:
        # Either we built a fresh pool -- so whatever the old mirror describes
        # no longer exists -- or it was refused. Same action either way.
        mirror = MirrorFile.create(path, num_blocks, identity)

    # Continue the predecessor's stamp space: inherited labels stay valid and
    # ours land strictly above them.
    fence = Fence(mirror, start=mirror.committed)
    install_writer(pool, mirror, fence)
    _install_fence(engine_core.scheduler, fence)
    pool._dyn_mirror = mirror
    pool._dyn_fence = fence

    global _IN_SLEEP
    _IN_SLEEP = False  # back to serving: a reset now means what it says


def _record_takeover(path: str, **fields) -> None:
    """Append one line per wake to ``<mirror>.status.jsonl``.

    A fail-closed design that fails *silently* decays into fail-open the first
    time someone inverts a condition, and "refused" has to be distinguishable
    from "never ran". This is the surface a harness asserts on; the engine
    process's logging config is not reliably ours to configure.
    """
    try:
        with open(f"{path}.status.jsonl", "a") as f:
            f.write(json.dumps(fields) + "\n")
    except OSError:
        pass


def _install_fence(scheduler, fence: Fence) -> None:
    """Wrap schedule/update_from_output once; re-point the fence on later wakes."""
    scheduler._dyn_fence = fence
    if getattr(scheduler, "_dyn_fenced", False):
        return
    schedule, update = scheduler.schedule, scheduler.update_from_output

    def _schedule(*args, **kwargs):
        scheduler._dyn_fence.on_schedule()
        return schedule(*args, **kwargs)

    def _update_from_output(*args, **kwargs):
        out = update(*args, **kwargs)
        scheduler._dyn_fence.on_update()
        return out

    scheduler.schedule = _schedule
    scheduler.update_from_output = _update_from_output
    scheduler._dyn_fenced = True


def enable_kv_index() -> None:
    """Entry point for ``vllm.general_plugins``. Self-disables when unset.

    Patches ``wake_up`` and nothing else. Notably NOT ``EngineCore.__init__``:
    plugins are loaded by its first statement, so a wrapper installed from here
    could never affect the frame already executing.
    """
    if not mirror_path():
        return
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_dyn_kvidx_patched", False):
        return
    original_wake, original_sleep = EngineCore.wake_up, EngineCore.sleep

    def wake_up(self, *args, **kwargs):
        result = original_wake(self, *args, **kwargs)
        try:
            on_wake_up(self)
        except Exception as e:  # persistence must never break serving
            logger.warning("[kvidx] takeover failed (%s); continuing cold", e)
        return result

    def sleep(self, *args, **kwargs):
        global _IN_SLEEP
        _IN_SLEEP = True  # cleared by the wake, not by this call returning
        return original_sleep(self, *args, **kwargs)

    EngineCore.wake_up = wake_up
    EngineCore.sleep = sleep
    EngineCore._dyn_kvidx_patched = True
    logger.info("[kvidx] enabled; mirror at %s", mirror_path())


def _requeue_to_tail(pool, blocks: list) -> None:
    """Move restored blocks to the free-queue tail.

    Mandatory, not an optimisation. vLLM keeps "hand out uncached blocks first"
    purely by queue *position*, and naming a block does not move it. On a freshly
    woken standby every block is unnamed and queued in id order, so a restored
    prefix sits at the head and the very first request served is handed its own
    blocks and overwrites them -- the cache would not survive one request.

    Descending ``num_tokens`` puts the deepest prefix furthest from the head, so
    the shallow end is evicted first and a usable leading prefix survives. Same
    convention vLLM uses in ``free_blocks``.
    """
    if not blocks:
        return
    for block in blocks:
        pool.free_block_queue.remove(block)
    blocks.sort(key=lambda b: (b.block_hash_num_tokens or 0), reverse=True)
    pool.free_block_queue.append_n(blocks)
