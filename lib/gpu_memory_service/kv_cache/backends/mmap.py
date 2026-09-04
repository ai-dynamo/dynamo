# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A block index store in a shared file, mapped into memory.

``MAP_SHARED``, so recording an entry is a store to a kernel-owned page rather
than a syscall -- cheap enough to do on every change instead of snapshotting.
Contents survive the process; nothing is ``fsync``'d, because the GPU pages they
describe do not survive the node.

A 4 KiB header, then one 64-byte slot per block, addressed by block id.

    MmapBlockIndexStore.create(path, num_blocks, identity)
                       .open(path, identity=, num_blocks=) -> (store, Refusal)
    MAGIC / VERSION / HEADER_BYTES        the on-disk contract
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
from typing import Iterator

import numpy as np
from gpu_memory_service.kv_cache.interface import Refusal

logger = logging.getLogger(__name__)

MAGIC = b"DYNKVIX1"
VERSION = 1
HEADER_BYTES = 4096
# sha256 digest + 4-byte group id = 36; xxhash128 + 4 = 20. 40 covers both.
KEY_BYTES = 40

_OFF_VERSION = 8
_OFF_COMPLETED = 16
_OFF_IDENTITY = 24
_DIGEST_BYTES = 32

# One slot per block, indexed by block_id -- an array, not a map.
_RECORD_DTYPE = np.dtype(
    [
        ("scheduled_at", "<u8"),  # 0 == empty; else the batch that recorded it
        ("num_tokens", "<u4"),  # 0 encodes None
        ("key_len", "<u4"),
        ("key", np.uint8, (KEY_BYTES,)),
        ("_rsv", "<u8"),
    ]
)
assert _RECORD_DTYPE.itemsize == 64, _RECORD_DTYPE.itemsize


class MmapBlockIndexStore:
    """A :class:`BlockIndexStore` backed by a memory-mapped file."""

    def __init__(self, path: str, mm: mmap.mmap, num_blocks: int):
        self.path = path
        self.num_blocks = num_blocks
        self._mm = mm
        # Writable views straight onto the mapping: recording is a store, not a
        # syscall. `completed` lives in the header so a successor can read it.
        self._completed = np.ndarray(
            (1,), dtype="<u8", buffer=mm, offset=_OFF_COMPLETED
        )
        self._rec = np.ndarray(
            (num_blocks,), dtype=_RECORD_DTYPE, buffer=mm, offset=HEADER_BYTES
        )
        # Continue the predecessor's numbering: inherited entries stay usable
        # and ours land strictly above them.
        self.scheduled = self.completed

    # ---- batch counters ----

    @property
    def completed(self) -> int:
        return int(self._completed[0])

    def on_schedule(self) -> None:
        self.scheduled += 1

    def on_complete(self) -> None:
        self._completed[0] = self.completed + 1

    # ---- writing ----

    def record(self, block_id: int, block_hash: bytes, num_tokens: int | None):
        """``scheduled_at`` is written LAST, always 0 -> n.

        A block is only re-named after being dropped, so a slot torn by SIGKILL
        always reads back 0 and is ignored -- which is why there is no checksum.
        """
        n = len(block_hash)
        self._rec["key"][block_id][:n] = np.frombuffer(block_hash, dtype=np.uint8)
        self._rec["key_len"][block_id] = n
        self._rec["num_tokens"][block_id] = num_tokens or 0
        self._rec["scheduled_at"][block_id] = self.scheduled

    def drop(self, block_id: int) -> None:
        self._rec["scheduled_at"][block_id] = 0

    def drop_all(self) -> None:
        self._rec["scheduled_at"][:] = 0

    def retain(self, block_ids) -> None:
        """Forget everything we did not keep.

        A refused entry carries a batch number above the current watermark, and
        our counter climbs from there -- so it would eventually look usable
        again, over a block we are now free to overwrite.
        """
        keep = np.zeros(self.num_blocks, dtype=bool)
        if block_ids:
            keep[list(block_ids)] = True
        self._rec["scheduled_at"][~keep] = 0

    # ---- reading ----

    def live(self) -> set[int]:
        return set(np.nonzero(self._rec["scheduled_at"] > 0)[0].tolist())

    def usable(self) -> Iterator[tuple[int, bytes, int | None]]:
        at = self._rec["scheduled_at"]
        for block_id in np.nonzero((at > 0) & (at <= self.completed))[0].tolist():
            n = int(self._rec["key_len"][block_id])
            yield (
                block_id,
                bytes(self._rec["key"][block_id][:n]),
                int(self._rec["num_tokens"][block_id]) or None,
            )

    def close(self) -> None:
        del self._completed, self._rec
        self._mm.close()

    # ---- lifecycle ----

    @staticmethod
    def _map(path: str, num_blocks: int) -> "MmapBlockIndexStore":
        with open(path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        return MmapBlockIndexStore(path, mm, num_blocks)

    @classmethod
    def create(
        cls, path: str, num_blocks: int, identity: bytes
    ) -> "MmapBlockIndexStore":
        size = HEADER_BYTES + num_blocks * _RECORD_DTYPE.itemsize
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.truncate(size)
            f.write(MAGIC)
            f.write(struct.pack("<II", VERSION, num_blocks))
            f.write(struct.pack("<Q", 0))  # completed
            f.write(identity.ljust(_DIGEST_BYTES, b"\0")[:_DIGEST_BYTES])
        os.replace(tmp, path)  # atomic: readers see whole-old or whole-new
        return cls._map(path, num_blocks)

    @classmethod
    def open(
        cls, path: str, *, identity: bytes, num_blocks: int
    ) -> tuple["MmapBlockIndexStore | None", Refusal]:
        """Admit the store, or refuse it with a reason.

        All-or-nothing gates, cheapest first; the per-entry counter check is
        applied later by :meth:`usable`. ``IDENTITY`` is the only refusal here
        whose absence would be a wrong answer rather than a miss.
        """
        if not os.path.exists(path):
            return None, Refusal.ABSENT
        try:
            with open(path, "rb") as f:
                head = f.read(HEADER_BYTES)
            actual_size = os.path.getsize(path)
        except OSError as e:
            logger.warning("[kv_cache] cannot read store %s: %s", path, e)
            return None, Refusal.UNREADABLE

        if len(head) < HEADER_BYTES or head[: len(MAGIC)] != MAGIC:
            return None, Refusal.MAGIC
        version, blocks = struct.unpack_from("<II", head, _OFF_VERSION)
        if version != VERSION:
            return None, Refusal.VERSION
        if blocks != num_blocks:
            return None, Refusal.NUM_BLOCKS
        if actual_size != HEADER_BYTES + num_blocks * _RECORD_DTYPE.itemsize:
            return None, Refusal.TRUNCATED
        if head[_OFF_IDENTITY : _OFF_IDENTITY + _DIGEST_BYTES] != identity:
            return None, Refusal.IDENTITY
        return cls._map(path, num_blocks), Refusal.OK
