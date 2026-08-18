# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry the vLLM prefix-cache index across an engine failover.

GMS can already make the KV *bytes* outlive the engine that wrote them
(``commit_layout()``). This carries the *index*: the ``block_hash -> block_id``
binding that lives in the scheduler's ``BlockPool`` and cannot be regenerated
from the bytes, because the hash is over token ids rather than content. Without
it a standby adopts correct KV that it cannot find, and re-prefills anyway.

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

Known gap: that invariant assumes every engine adopting these pages
participates. An engine running with ``DYN_GMS_PERSIST_KV`` but *without* this
feature adopts the pages, overwrites blocks and leaves nothing observable
behind, so a later participant would replay labels describing bytes that engine
changed. Closing it needs a writer epoch on the GMS handshake; until then, do
not mix participating and non-participating engines over one pool.

Enabled by setting ``GMS_KV_INDEX_PATH``; off by default. Installed through the
``vllm.general_plugins`` entry point in ``setup.py``, which wraps
``EngineCore.wake_up`` and ``EngineCore.sleep`` and nothing else.
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

# Everything else is internal, exported only for the tests.
__all__ = ["enable_kv_index"]

MAGIC = b"DYNKVIX1"
VERSION = 1
HEADER_BYTES = 4096
# sha256 digest + 4-byte group id = 36; xxhash128 + 4 = 20. 40 covers both.
KEY_BYTES = 40

_OFF_VERSION = 8
_OFF_COMMITTED = 16
_OFF_IDENTITY = 24
_DIGEST_BYTES = 32

# NOTE: the sleep flag now lives on the scheduler (``_dyn_sleeping``), not in
# module scope -- one engine's sleep no longer speaks for the whole process.
# One slot per block, indexed by block_id -- a flat array, not a map.
_REC = np.dtype(
    [
        ("stamp", "<u8"),  # 0 == invalid; else `scheduled` at publish
        ("num_tokens", "<u4"),  # 0 encodes None
        ("key_len", "<u4"),
        ("key", np.uint8, (KEY_BYTES,)),
        ("_rsv", "<u8"),
    ]
)
assert _REC.itemsize == 64, _REC.itemsize


def mirror_path() -> str | None:
    """Where the mirror lives, or None when the feature is off (the default)."""
    return os.environ.get("GMS_KV_INDEX_PATH") or None


class MirrorFile:
    """The artifact: a fixed-size shared mapping, header plus one slot per block.

    Also carries the publication watermark. vLLM attaches a label during
    ``schedule()`` -- and, under async scheduling, inside ``update_from_output``
    -- in both cases before the KV it names is certainly on the device. Batches
    are reaped strictly FIFO, so counting schedules against completions is
    enough: a label stamped with ``scheduled`` becomes trustworthy once
    ``committed`` reaches it. An engine that dies mid-batch never advances
    ``committed`` that far, so those labels stay untrusted forever.
    """

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
        # Continue the predecessor's stamp space: inherited labels stay valid
        # and ours land strictly above them.
        self.scheduled = self.committed

    # ---- publication watermark ----

    @property
    def committed(self) -> int:
        return int(self._committed[0])

    def on_schedule(self) -> None:
        self.scheduled += 1

    def on_update(self) -> None:
        self._committed[0] = self.committed + 1

    # ---- writer ----

    def publish(self, block_id: int, key: bytes, num_tokens: int | None) -> None:
        """Record a label. The stamp is written LAST, always 0 -> n.

        A block can only be re-named after being retracted (``set_block_hash``
        asserts the slot is empty), so a record torn by SIGKILL always reads back
        ``stamp == 0`` and is ignored. That is why no checksum is needed.
        """
        n = len(key)
        self._rec["key"][block_id][:n] = np.frombuffer(key, dtype=np.uint8)
        self._rec["key_len"][block_id] = n
        self._rec["num_tokens"][block_id] = num_tokens or 0
        self._rec["stamp"][block_id] = self.scheduled

    def invalidate(self, block_id: int) -> None:
        self._rec["stamp"][block_id] = 0

    def invalidate_all(self) -> None:
        self._rec["stamp"][:] = 0

    def retain_only(self, block_ids) -> None:
        """Drop every record we did not install.

        A refused record keeps a stamp above the current watermark, but our
        watermark counts up from there -- so a stale stamp would eventually fall
        below it and start looking trustworthy. We also own the pool now, so
        those blocks are ours to overwrite.
        """
        keep = np.zeros(self.num_blocks, dtype=bool)
        if block_ids:
            keep[list(block_ids)] = True
        self._rec["stamp"][~keep] = 0

    # ---- reader ----

    def live_block_ids(self) -> set[int]:
        """Blocks currently claimed, ignoring the watermark. Test-only: the
        subset invariant is asserted against this."""
        return set(np.nonzero(self._rec["stamp"] > 0)[0].tolist())

    def trusted(self):
        """Yield ``(block_id, key, num_tokens)`` for labels whose KV exists."""
        stamps = self._rec["stamp"]
        for block_id in np.nonzero((stamps > 0) & (stamps <= self.committed))[
            0
        ].tolist():
            n = int(self._rec["key_len"][block_id])
            key = bytes(self._rec["key"][block_id][:n])
            yield block_id, key, (int(self._rec["num_tokens"][block_id]) or None)

    # ---- lifecycle ----

    @staticmethod
    def _map(path: str, num_blocks: int) -> "MirrorFile":
        with open(path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
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

        All-or-nothing gates live here; the per-record watermark is applied later
        by :meth:`trusted`. Any refusal degrades to a cold cache.
        """
        if not os.path.exists(path):
            return None, "absent"
        try:
            with open(path, "rb") as f:
                head = f.read(HEADER_BYTES)
            actual_size = os.path.getsize(path)
        except OSError as e:
            logger.warning("[kv_index] cannot read mirror %s: %s", path, e)
            return None, "unreadable"

        if len(head) < HEADER_BYTES or head[: len(MAGIC)] != MAGIC:
            return None, "magic"
        version, blocks = struct.unpack_from("<II", head, _OFF_VERSION)
        if version != VERSION:
            return None, "version"
        if blocks != num_blocks:
            return None, "num_blocks"
        if actual_size != HEADER_BYTES + num_blocks * _REC.itemsize:
            return None, "truncated"
        if head[_OFF_IDENTITY : _OFF_IDENTITY + _DIGEST_BYTES] != identity:
            return None, "identity"
        return cls._map(path, num_blocks), "ok"


def install_writer(pool, mirror: MirrorFile) -> None:
    """Mirror every label change on this ``BlockPool`` instance.

    A runtime subclass swap rather than a patch on the class: it composes with
    whatever ``BlockPool`` subclass is already in play, and is scoped to the one
    object we were handed. Re-arming on a later wake only re-points the mirror.
    """
    pool._dyn_mirror = mirror
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
            return
        self._dyn_mirror.publish(
            block.block_id, bytes(label), block.block_hash_num_tokens
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
        if ok and not _sleeping_for(self):
            # Outside a sleep this means the operator dropped the index on
            # purpose -- an RLHF weight update -- so the labels are dead.
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
    logger.info("[kv_index] replayed %d labels", len(installed))
    return installed


def _requeue_to_tail(pool, blocks: list) -> None:
    """Move restored blocks to the free-queue tail.

    Mandatory, not an optimisation. vLLM keeps "hand out uncached blocks first"
    purely by queue *position*, and naming a block does not move it. On a freshly
    woken standby every block is unnamed and queued in id order, so a restored
    prefix sits at the head and the very first request served is handed its own
    blocks and overwrites them -- the cache would not survive one request.

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


def _probe_adopted(path: str, world_size: int) -> bool:
    """Did EVERY rank inherit its KV pages?

    The scheduler process has no handle on the executor, so it cannot ask the
    workers directly. Each rank drops a one-byte answer next to the mirror at
    wake (``_publish_kv_adoption`` in worker.py) and we read them here.

    Consumed, not just read. A file left behind is a *stale* answer, and a stale
    "1" is the one input to this whole design that turns a miss into a wrong
    answer -- it would replay a predecessor's labels onto pages this engine
    allocated itself. Deleting after each read makes absence the resting state,
    so a rank that fails to publish reads as "no" rather than as whatever the
    previous engine said.

    Missing or unreadable is therefore a NO, and so is anything short of
    unanimity: the index is engine-wide but the bytes are per-rank.
    """
    answers = []
    for rank in range(world_size):
        rank_path = f"{path}.rank{rank}"
        try:
            with open(rank_path) as f:
                answers.append(f.read(1) == "1")
        except OSError:
            answers.append(False)
        try:
            os.unlink(rank_path)
        except OSError:
            # Could not consume it, so we cannot promise the next wake sees a
            # fresh answer. Refuse now rather than trust it later.
            logger.warning("[kv_index] could not consume %s; refusing", rank_path)
            answers.append(False)
    return bool(answers) and all(answers)


def _identity(cfg, num_blocks: int) -> bytes:
    """Hash the ruler: everything that decides what a block id measures.

    A label says "block 457". That only means something if both engines measure
    blocks the same way, so a mismatch is a refusal rather than a migration --
    the one failure mode here that would be a wrong answer rather than a miss.
    """
    import vllm
    from vllm.v1.core import kv_cache_utils

    fields = {
        "vllm": vllm.__version__,
        "model": cfg.model_config.model,
        "block_size": cfg.cache_config.block_size,
        "cache_dtype": str(cfg.cache_config.cache_dtype),
        "tp": cfg.parallel_config.tensor_parallel_size,
        "num_blocks": num_blocks,
        # Root of the block-hash chain: os.urandom per process unless
        # PYTHONHASHSEED is pinned, so this also catches an unpinned seed.
        "none_hash": bytes(kv_cache_utils.NONE_HASH).hex(),
        "hash_algo": str(cfg.cache_config.prefix_caching_hash_algo),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).digest()


def take_over(scheduler) -> None:
    """Take over the index, then start recording. The whole integration.

    Called from ``set_pause_state(UNPAUSED)``, which vLLM reaches from
    ``EngineCore.wake_up`` only after ``model_executor.wake_up()`` has returned
    -- a blocking collective, so every rank has re-attached -- and before the
    scheduler can hand out a block. vLLM also gates that call on the executor
    being fully awake, so partial wakes never reach us.
    """
    path = mirror_path()
    if not path:
        return

    pool = scheduler.kv_cache_manager.block_pool
    num_blocks = len(pool.blocks)
    identity = _identity(scheduler.vllm_config, num_blocks)
    # vLLM's own index as we find it. Sleep clears it, so this is normally 0;
    # it is what distinguishes "replay rebuilt the index" from "labels happened
    # to survive in process memory".
    labelled_before = sum(1 for b in pool.blocks if b.block_hash is not None)

    world_size = getattr(scheduler.parallel_config, "world_size", 1)
    adopted = _probe_adopted(path, world_size)

    mirror, reason, installed = None, "not_adopted", []
    if adopted:
        mirror, reason = MirrorFile.open_for_replay(
            path, identity=identity, num_blocks=num_blocks
        )
        if mirror is None:
            logger.info("[kv_index] not replaying (%s); starting fresh", reason)
        else:
            installed = replay(pool, mirror)
            mirror.retain_only({b.block_id for b in installed})

    _record_takeover(
        path,
        adopted=adopted,
        reason=reason,
        labelled_before=labelled_before,
        installed=len(installed),
    )

    if mirror is None:
        # Either we built a fresh pool -- so whatever the old mirror describes
        # no longer exists -- or it was refused. Same action either way, and it
        # is what makes a stale mirror inert for everyone after us.
        mirror = MirrorFile.create(path, num_blocks, identity)

    pool._dyn_scheduler = scheduler
    install_writer(pool, mirror)
    scheduler._dyn_mirror = mirror


def _record_takeover(path: str, **fields) -> None:
    """Append one line per wake to ``<mirror>.status.jsonl``.

    A fail-closed design that fails *silently* decays into fail-open the first
    time someone inverts a condition, and "refused" has to be distinguishable
    from "never ran". The engine process's logging config is not reliably ours.
    """
    try:
        with open(f"{path}.status.jsonl", "a") as f:
            f.write(json.dumps(fields) + "\n")
    except OSError as e:
        logger.debug("[kv_index] could not record takeover: %s", e)


def _disarm(path: str | None) -> None:
    """Remove the mirror so no successor can trust it."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError as e:
        logger.warning("[kv_index] could not disarm mirror %s: %s", path, e)


class KvIndexMixin:
    """Scheduler behaviour for the prefix-index mirror.

    Mix in *ahead* of a vLLM scheduler:

        class KvIndexScheduler(KvIndexMixin, AsyncScheduler): ...

    ``set_pause_state`` is the interesting one: both the sleep and the wake
    transition run through it, in the scheduler's own process, at exactly the
    moments the takeover needs. ``EngineCore.pause_scheduler`` sets PAUSED_*
    before clearing the prefix cache, and ``resume_scheduler`` sets UNPAUSED
    after ``model_executor.wake_up()`` has returned -- a blocking collective, so
    every rank has re-attached -- and before scheduling resumes.
    """

    def set_pause_state(self, pause_state) -> None:
        from vllm.v1.core.sched.interface import PauseState

        if pause_state != PauseState.UNPAUSED:
            # Entering a sleep. vLLM is about to clear its own prefix index on
            # the assumption the KV is discarded; under a committed GMS layout
            # it is not, so the mirror must survive that clear.
            self._dyn_sleeping = True
            super().set_pause_state(pause_state)
            return

        super().set_pause_state(pause_state)
        if not getattr(self, "_dyn_sleeping", False):
            return  # not a wake from a sleep -- nothing to take over
        self._dyn_sleeping = False
        try:
            take_over(self)
        except Exception as e:  # persistence must never break serving
            logger.warning("[kv_index] takeover failed (%s); continuing cold", e)
            _disarm(mirror_path())

    def schedule(self, *args, **kwargs):
        mirror = getattr(self, "_dyn_mirror", None)
        if mirror is not None:
            mirror.on_schedule()
        return super().schedule(*args, **kwargs)

    def update_from_output(self, *args, **kwargs):
        out = super().update_from_output(*args, **kwargs)
        mirror = getattr(self, "_dyn_mirror", None)
        if mirror is not None:
            mirror.on_update()
        return out


def _make_scheduler_cls(base: type) -> type:
    """Derive from whatever scheduler vLLM (or Dynamo) already chose."""
    return type(f"KvIndex{base.__name__}", (KvIndexMixin, base), {})


def _sleeping_for(pool) -> bool:
    """The BlockPool hook asks its scheduler whether we are mid-sleep."""
    sched = getattr(pool, "_dyn_scheduler", None)
    return bool(sched is not None and getattr(sched, "_dyn_sleeping", False))


def enable_kv_index() -> None:
    """Entry point for ``vllm.general_plugins``. Self-disables when unset.

    This is the *fallback* path, for when nobody named a scheduler explicitly.
    The supported route is ``--scheduler-cls`` naming one of the classes in
    ``schedulers.py``; this wrap only says "if no one chose for us, choose well"
    by deriving from whatever base vLLM or Dynamo already settled on.
    """
    if not mirror_path():
        return
    from vllm.config.scheduler import SchedulerConfig

    if getattr(SchedulerConfig, "_dyn_kv_index_patched", False):
        return
    original = SchedulerConfig.get_scheduler_cls
    derived: dict[type, type] = {}

    def get_scheduler_cls(self):
        base = original(self)
        if base not in derived:
            derived[base] = _make_scheduler_cls(base)
        return derived[base]

    SchedulerConfig.get_scheduler_cls = get_scheduler_cls
    SchedulerConfig._dyn_kv_index_patched = True
    logger.info("[kv_index] enabled; mirror at %s", mirror_path())
