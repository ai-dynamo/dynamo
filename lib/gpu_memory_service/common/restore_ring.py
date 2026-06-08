# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk-grained restore ring — engine → daemon submission channel for
GMSRadixCache cache-hit restores.

Companion to `common/evict_ring.py` (which handles the spill direction).
Where the spill ring carries per-block records that the daemon turns
into N-layer D2H operations, this ring carries **per-chunk** records
that the daemon expands into (chunk_n_blocks × n_layers) H2D
operations and finishes by signalling a shared 32-bit counter.

The completion signal is a `cuStreamWaitValue32` semaphore, NOT a
cudaEvent — this kills the IPC-event-handle handshake (saving ~5µs
per restore) since the engine and daemon share a small array of
counters via cuMemHostRegister + cuMemHostGetDevicePointer.

Layout (one /dev/shm file, mmap'd by both processes):

  ┌─ Header (64B) ──────────────────────────────────────────────┐
  │  magic        : u32   = 0x47_4D_53_52  ("GMSR")             │
  │  version      : u32   = 1                                    │
  │  capacity     : u32   record count                           │
  │  record_size  : u32   = RECORD_SIZE (512)                    │
  │  head_seq     : u64   producer-only                          │
  │  tail_seq     : u64   consumer-only                          │
  │  drops        : u64   producer-only                          │
  │  padding      : 16B                                          │
  └──────────────────────────────────────────────────────────────┘
  ┌─ Record[0]   (512B) ────────────────────────────────────────┐
  │  seq            : u64   (non-zero ⇒ ready for consumer)     │
  │  op_kind        : u8                                         │
  │  flags          : u8                                         │
  │  _pad           : u16                                        │
  │  counter_slot   : u32   (semaphore array index)              │
  │  counter_target : u32   (value to atomic-store on completion)│
  │  n_blocks       : u32   (number of (src, dest) pairs)        │
  │  _pad2          : u64                                        │
  │  src_engine_id  : char[48]   (NUL-padded)                    │
  │  block_pairs[54]: each (u32 src_blk, u32 dest_blk)           │
  └──────────────────────────────────────────────────────────────┘

Note: layer count is implicit (the daemon already has the
src_engine_id's layer layout cached). All layers of the chunk are
restored before the counter is bumped.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Try to load the native Rust hot path. Falls back to pure Python.
_RUST_RING = None
try:
    import gms_rust_ring as _RUST_RING  # type: ignore[import-not-found]

    logger.debug("restore_ring: using native Rust push/pop")
except Exception:  # noqa: BLE001
    _RUST_RING = None
    logger.debug("restore_ring: native Rust extension not available; pure Python")


def is_rust_accelerated() -> bool:
    """True iff the native Rust hot path is available."""
    return _RUST_RING is not None


MAGIC = 0x47_4D_53_52  # "GMSR"
VERSION = 1
HEADER_SIZE = 64
RECORD_SIZE = 512
ENGINE_ID_MAX_LEN = 48
MAX_BLOCK_PAIRS_PER_RECORD = 54  # 432 bytes / 8 bytes-per-pair

# Op kinds
OP_NOOP = 0
OP_RESTORE_CHUNK = 1
OP_DRAIN = 2

# Header field offsets (little-endian)
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_CAPACITY = 8
_OFF_RECORD_SIZE = 12
_OFF_HEAD_SEQ = 16
_OFF_TAIL_SEQ = 24
_OFF_DROPS = 32

# Record field offsets
_R_SEQ = 0
_R_OP = 8
_R_FLAGS = 9
_R_COUNTER_SLOT = 12
_R_COUNTER_TARGET = 16
_R_N_BLOCKS = 20
_R_ENGINE_ID = 32
_R_BLOCK_PAIRS = 80
_BLOCK_PAIR_STRIDE = 8  # u32 src + u32 dest


def _file_size(capacity: int) -> int:
    return HEADER_SIZE + capacity * RECORD_SIZE


def create_restore_ring(path: str, capacity: int = 4096):
    """Create a fresh restore ring file at `path` and return a writer."""
    if capacity <= 0 or (capacity & (capacity - 1)) != 0:
        raise ValueError(f"capacity must be a power of 2; got {capacity}")
    size = _file_size(capacity)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, size)
        buf = mmap.mmap(
            fd,
            size,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        struct.pack_into(
            "<IIIIQQQ",
            buf,
            0,
            MAGIC,
            VERSION,
            capacity,
            RECORD_SIZE,
            0,
            0,
            0,
        )
    finally:
        os.close(fd)
    return RestoreRingWriter(path, buf, capacity)


def attach_restore_ring_writer(path: str):
    return _attach(path, RestoreRingWriter)


def attach_restore_ring_reader(path: str):
    return _attach(path, RestoreRingReader)


def _attach(path: str, cls):
    fd = os.open(path, os.O_RDWR)
    try:
        size = os.fstat(fd).st_size
        buf = mmap.mmap(
            fd,
            size,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(fd)
    magic, version, capacity, record_size, _, _, _ = struct.unpack_from(
        "<IIIIQQQ",
        buf,
        0,
    )
    if magic != MAGIC:
        buf.close()
        raise ValueError(f"bad magic 0x{magic:x}; not a GMS restore ring")
    if version != VERSION:
        buf.close()
        raise ValueError(f"version mismatch: {version} != {VERSION}")
    if record_size != RECORD_SIZE:
        buf.close()
        raise ValueError(
            f"record_size mismatch: {record_size} != {RECORD_SIZE}",
        )
    return cls(path, buf, capacity)


# ---------------------------------------------------------------------------
# Writer (engine side)
# ---------------------------------------------------------------------------


class RestoreRingWriter:
    """SPSC writer. Engine pushes one record per chunk restore."""

    def __init__(self, path: str, buf: mmap.mmap, capacity: int) -> None:
        self.path = path
        self.buf = buf
        self.capacity = capacity
        self._mask = capacity - 1

    def close(self) -> None:
        if self.buf is not None:
            try:
                self.buf.close()
            except Exception:
                pass
            self.buf = None

    def _read_head(self) -> int:
        return struct.unpack_from("<Q", self.buf, _OFF_HEAD_SEQ)[0]

    def _read_tail(self) -> int:
        return struct.unpack_from("<Q", self.buf, _OFF_TAIL_SEQ)[0]

    def _write_head(self, n: int) -> None:
        struct.pack_into("<Q", self.buf, _OFF_HEAD_SEQ, int(n))

    def _bump_drops(self) -> None:
        cur = struct.unpack_from("<Q", self.buf, _OFF_DROPS)[0]
        struct.pack_into("<Q", self.buf, _OFF_DROPS, cur + 1)

    def push(
        self,
        *,
        src_engine_id: str,
        block_pairs: Iterable[tuple],
        counter_slot: int,
        counter_target: int,
        flags: int = 0,
    ) -> bool:
        """Push one chunk-restore record.

        `block_pairs` is an iterable of `(src_block_id, dest_block_id)`
        tuples — N entries, one per block in the chunk. The daemon
        restores all layers of each source block into the corresponding
        destination block, then atomic-writes `counter_target` to
        the semaphore slot at index `counter_slot`.

        Returns False if the ring was full (drop counter bumped).
        """
        if _RUST_RING is not None:
            # Fast path: native push. Validates lengths internally.
            pairs = list(block_pairs)
            src_blks = [int(p[0]) for p in pairs]
            dest_blks = [int(p[1]) for p in pairs]
            eid_bytes = src_engine_id.encode("utf-8")
            return _RUST_RING.push_record(
                self.buf,
                self.capacity,
                eid_bytes,
                src_blks,
                dest_blks,
                int(counter_slot),
                int(counter_target),
                int(flags) & 0xFF,
            )
        # Slow path: pure-Python fallback (kept for environments
        # without the native extension, e.g. CI without rustc).
        return self._push_python(
            src_engine_id=src_engine_id,
            block_pairs=block_pairs,
            counter_slot=counter_slot,
            counter_target=counter_target,
            flags=flags,
        )

    def _push_python(
        self,
        *,
        src_engine_id: str,
        block_pairs: Iterable[tuple],
        counter_slot: int,
        counter_target: int,
        flags: int = 0,
    ) -> bool:
        pairs = list(block_pairs)
        if len(pairs) > MAX_BLOCK_PAIRS_PER_RECORD:
            raise ValueError(
                f"too many block pairs per record: "
                f"{len(pairs)} > {MAX_BLOCK_PAIRS_PER_RECORD}",
            )
        head = self._read_head()
        tail = self._read_tail()
        if head - tail >= self.capacity:
            self._bump_drops()
            return False

        eid_bytes = src_engine_id.encode("utf-8")
        if len(eid_bytes) >= ENGINE_ID_MAX_LEN:
            raise ValueError(
                f"engine_id too long for restore ring: {len(eid_bytes)} "
                f">= {ENGINE_ID_MAX_LEN}",
            )

        slot_idx = head & self._mask
        base = HEADER_SIZE + slot_idx * RECORD_SIZE
        struct.pack_into("<Q", self.buf, base + _R_SEQ, 0)

        struct.pack_into("<B", self.buf, base + _R_OP, OP_RESTORE_CHUNK)
        struct.pack_into("<B", self.buf, base + _R_FLAGS, int(flags) & 0xFF)
        struct.pack_into(
            "<I",
            self.buf,
            base + _R_COUNTER_SLOT,
            int(counter_slot),
        )
        struct.pack_into(
            "<I",
            self.buf,
            base + _R_COUNTER_TARGET,
            int(counter_target),
        )
        struct.pack_into("<I", self.buf, base + _R_N_BLOCKS, len(pairs))

        eid_padded = eid_bytes.ljust(ENGINE_ID_MAX_LEN, b"\x00")
        self.buf[
            base + _R_ENGINE_ID : base + _R_ENGINE_ID + ENGINE_ID_MAX_LEN
        ] = eid_padded

        for i, (src_blk, dest_blk) in enumerate(pairs):
            struct.pack_into(
                "<II",
                self.buf,
                base + _R_BLOCK_PAIRS + i * _BLOCK_PAIR_STRIDE,
                int(src_blk),
                int(dest_blk),
            )
        for i in range(len(pairs), MAX_BLOCK_PAIRS_PER_RECORD):
            struct.pack_into(
                "<II",
                self.buf,
                base + _R_BLOCK_PAIRS + i * _BLOCK_PAIR_STRIDE,
                0,
                0,
            )

        struct.pack_into("<Q", self.buf, base + _R_SEQ, head + 1)
        self._write_head(head + 1)
        return True

    def stats(self) -> dict:
        return {
            "head": self._read_head(),
            "tail": self._read_tail(),
            "drops": struct.unpack_from("<Q", self.buf, _OFF_DROPS)[0],
            "capacity": self.capacity,
        }


# ---------------------------------------------------------------------------
# Reader (daemon side)
# ---------------------------------------------------------------------------


class RestoreRingReader:
    """SPSC reader. Daemon's consumer thread pops one record per chunk."""

    def __init__(self, path: str, buf: mmap.mmap, capacity: int) -> None:
        self.path = path
        self.buf = buf
        self.capacity = capacity
        self._mask = capacity - 1

    def close(self) -> None:
        if self.buf is not None:
            try:
                self.buf.close()
            except Exception:
                pass
            self.buf = None

    def _read_head(self) -> int:
        return struct.unpack_from("<Q", self.buf, _OFF_HEAD_SEQ)[0]

    def _read_tail(self) -> int:
        return struct.unpack_from("<Q", self.buf, _OFF_TAIL_SEQ)[0]

    def _write_tail(self, n: int) -> None:
        struct.pack_into("<Q", self.buf, _OFF_TAIL_SEQ, int(n))

    def try_pop(self) -> Optional[dict]:
        """Pop one ready record, or return None if the ring is empty.

        Returns a dict with the parsed payload:
          {
            "op": int,
            "flags": int,
            "counter_slot": int,
            "counter_target": int,
            "src_engine_id": str,
            "block_pairs": [(src, dest), ...],
          }
        """
        if _RUST_RING is not None:
            res = _RUST_RING.try_pop_record(self.buf, self.capacity)
            if res is None:
                return None
            op, flags, counter_slot, counter_target, eid_bytes, pairs = res
            return {
                "op": int(op),
                "flags": int(flags),
                "counter_slot": int(counter_slot),
                "counter_target": int(counter_target),
                "src_engine_id": eid_bytes.decode("utf-8", errors="replace"),
                "block_pairs": [(int(s), int(d)) for s, d in pairs],
            }
        return self._try_pop_python()

    def _try_pop_python(self) -> Optional[dict]:
        tail = self._read_tail()
        head = self._read_head()
        if tail >= head:
            return None
        slot_idx = tail & self._mask
        base = HEADER_SIZE + slot_idx * RECORD_SIZE
        seq = struct.unpack_from("<Q", self.buf, base + _R_SEQ)[0]
        if seq != tail + 1:
            return None

        op = struct.unpack_from("<B", self.buf, base + _R_OP)[0]
        flags = struct.unpack_from("<B", self.buf, base + _R_FLAGS)[0]
        counter_slot, counter_target, n_blocks = struct.unpack_from(
            "<III",
            self.buf,
            base + _R_COUNTER_SLOT,
        )
        eid_raw = bytes(
            self.buf[base + _R_ENGINE_ID : base + _R_ENGINE_ID + ENGINE_ID_MAX_LEN]
        )
        nul = eid_raw.find(b"\x00")
        if nul >= 0:
            eid_raw = eid_raw[:nul]
        src_engine_id = eid_raw.decode("utf-8", errors="replace")

        n_blocks = min(int(n_blocks), MAX_BLOCK_PAIRS_PER_RECORD)
        pairs = []
        for i in range(n_blocks):
            src_blk, dest_blk = struct.unpack_from(
                "<II",
                self.buf,
                base + _R_BLOCK_PAIRS + i * _BLOCK_PAIR_STRIDE,
            )
            pairs.append((int(src_blk), int(dest_blk)))

        self._write_tail(tail + 1)
        return {
            "op": int(op),
            "flags": int(flags),
            "counter_slot": int(counter_slot),
            "counter_target": int(counter_target),
            "src_engine_id": src_engine_id,
            "block_pairs": pairs,
        }

    def stats(self) -> dict:
        return {
            "head": self._read_head(),
            "tail": self._read_tail(),
            "capacity": self.capacity,
        }
