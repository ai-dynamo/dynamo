# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore ring — SPSC SHM ring for engine→daemon cache-hit restore.

Chunk-grained: one record describes one prefix-chunk's restore as
N (src_block, dest_block) pairs. The daemon expands and issues
N×L cuMemcpyAsync H2D's, then signals a shared counter via
cuStreamWriteValue32. Engine waits on the counter via
cuStreamWaitValue32.

Layout (one /dev/shm file, mmap'd by both):

  ┌─ Header (64B) ──────────────────────────────────────────────┐
  │  magic        : u32  = 0x47_4D_53_52  ("GMSR")              │
  │  version      : u32  = 1                                     │
  │  capacity     : u32  power-of-2                              │
  │  record_size  : u32  = 512                                   │
  │  head_seq     : u64                                          │
  │  tail_seq     : u64                                          │
  │  drops        : u64                                          │
  │  padding      : 16B                                          │
  └──────────────────────────────────────────────────────────────┘
  ┌─ Record (512B) ─────────────────────────────────────────────┐
  │  seq            : u64                                        │
  │  op_kind        : u8   = 1 (RESTORE_CHUNK)                   │
  │  flags          : u8                                         │
  │  _pad           : u16                                        │
  │  counter_slot   : u32                                        │
  │  counter_target : u32                                        │
  │  n_pairs        : u32                                        │
  │  _pad2          : u64                                        │
  │  src_engine_id  : char[48]  NUL-padded                       │
  │  pairs[54]      : each (u32 src_block, u32 dest_block)       │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import mmap
import os
import struct
from typing import Iterable, Optional

try:
    import gms_rust_ring as _RUST  # type: ignore[import-not-found]
except Exception:
    _RUST = None


MAGIC = 0x5253_4D47  # "GMSR" little-endian
VERSION = 1
HEADER_SIZE = 64
RECORD_SIZE = 512
ENGINE_ID_MAX_LEN = 48
MAX_BLOCK_PAIRS = 54

OP_RESTORE_CHUNK = 1

# Flags byte (8 bits) on each ring record. Currently:
#   bit 0 — FLAG_GDS_DIRECT: bytes come from the storage tier via
#           the backend's `promote_into_gpu` (cuFile read direct
#           into HBM), NOT from host_tier. Daemon consumer
#           branches in `_RestoreConsumer._process`.
#   bit 1 — FLAG_SOURCE_STAGING: src_block in each pair is an opaque
#           staging-restore handle registered with the daemon. Bytes
#           come from StagingTier and are scattered into dest HBM.
# Remaining bits reserved.
FLAG_GDS_DIRECT = 0x01
FLAG_SOURCE_STAGING = 0x02

_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_CAPACITY = 8
_OFF_RECORD_SIZE = 12
_OFF_HEAD = 16
_OFF_TAIL = 24
_OFF_DROPS = 32

_R_SEQ = 0
_R_OP = 8
_R_FLAGS = 9
_R_COUNTER_SLOT = 12
_R_COUNTER_TARGET = 16
_R_N_PAIRS = 20
_R_ENGINE_ID = 32
_R_PAIRS = 80
_PAIR_STRIDE = 8


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def create_ring(path: str, capacity: int = 4096) -> "RestoreRingWriter":
    if not _is_pow2(capacity):
        raise ValueError(f"capacity must be power-of-2: got {capacity}")
    size = HEADER_SIZE + capacity * RECORD_SIZE
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, size)
        buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
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


def attach_writer(path: str) -> "RestoreRingWriter":
    return _attach(path, RestoreRingWriter)


def attach_reader(path: str) -> "RestoreRingReader":
    return _attach(path, RestoreRingReader)


def _attach(path: str, cls):
    fd = os.open(path, os.O_RDWR)
    try:
        size = os.fstat(fd).st_size
        if size < HEADER_SIZE:
            raise ValueError(f"ring file too small: {size} < {HEADER_SIZE}")
        buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    magic, version, capacity, rec_size, _h, _t, _d = struct.unpack_from(
        "<IIIIQQQ",
        buf,
        0,
    )
    error = None
    if magic != MAGIC:
        error = f"bad magic 0x{magic:x}: not a GMSR ring"
    elif version != VERSION:
        error = f"version mismatch: {version} != {VERSION}"
    elif not _is_pow2(capacity):
        error = f"capacity must be power-of-2: got {capacity}"
    elif rec_size != RECORD_SIZE:
        error = f"record_size mismatch: {rec_size} != {RECORD_SIZE}"
    else:
        expected_size = HEADER_SIZE + capacity * rec_size
        if size != expected_size:
            error = f"ring size mismatch: {size} != {expected_size}"
    if error is not None:
        buf.close()
        raise ValueError(error)
    return cls(path, buf, capacity)


class _RingBase:
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

    def stats(self) -> dict:
        return {
            "head": struct.unpack_from("<Q", self.buf, _OFF_HEAD)[0],
            "tail": struct.unpack_from("<Q", self.buf, _OFF_TAIL)[0],
            "drops": struct.unpack_from("<Q", self.buf, _OFF_DROPS)[0],
            "capacity": self.capacity,
        }


def is_rust_accelerated() -> bool:
    """True iff the native Rust extension is loaded and will be used
    for push/pop. False ⇒ pure-Python fallback path is active."""
    return _RUST is not None


class RestoreRingWriter(_RingBase):
    """Engine-side producer. Single thread.

    `push()` auto-selects: Rust extension if available (~1 µs/op),
    pure-Python fallback otherwise (~9 µs/op). Both paths write the
    same on-disk bytes (verified by test_rust_python_parity)."""

    def push(
        self,
        src_engine_id: str,
        block_pairs: Iterable[tuple],
        counter_slot: int,
        counter_target: int,
        flags: int = 0,
    ) -> bool:
        if _RUST is not None:
            pairs = list(block_pairs)
            src = [int(p[0]) for p in pairs]
            dst = [int(p[1]) for p in pairs]
            return _RUST.push_record(
                self.buf,
                self.capacity,
                src_engine_id.encode("utf-8"),
                src,
                dst,
                int(counter_slot),
                int(counter_target),
                int(flags) & 0xFF,
            )
        return self._push_python(
            src_engine_id,
            block_pairs,
            counter_slot,
            counter_target,
            flags,
        )

    def _push_python(
        self,
        src_engine_id: str,
        block_pairs: Iterable[tuple],
        counter_slot: int,
        counter_target: int,
        flags: int = 0,
    ) -> bool:
        pairs = list(block_pairs)
        if len(pairs) > MAX_BLOCK_PAIRS:
            raise ValueError(
                f"too many pairs per record: {len(pairs)} > {MAX_BLOCK_PAIRS}",
            )
        eid_bytes = src_engine_id.encode("utf-8")
        if len(eid_bytes) >= ENGINE_ID_MAX_LEN:
            raise ValueError(
                f"engine_id too long: {len(eid_bytes)} >= {ENGINE_ID_MAX_LEN}",
            )

        head = struct.unpack_from("<Q", self.buf, _OFF_HEAD)[0]
        tail = struct.unpack_from("<Q", self.buf, _OFF_TAIL)[0]
        if head - tail >= self.capacity:
            drops = struct.unpack_from("<Q", self.buf, _OFF_DROPS)[0]
            struct.pack_into("<Q", self.buf, _OFF_DROPS, drops + 1)
            return False

        slot = head & self._mask
        base = HEADER_SIZE + slot * RECORD_SIZE

        struct.pack_into("<Q", self.buf, base + _R_SEQ, 0)
        struct.pack_into("<B", self.buf, base + _R_OP, OP_RESTORE_CHUNK)
        struct.pack_into("<B", self.buf, base + _R_FLAGS, int(flags) & 0xFF)
        struct.pack_into(
            "<III",
            self.buf,
            base + _R_COUNTER_SLOT,
            int(counter_slot),
            int(counter_target),
            len(pairs),
        )
        self.buf[
            base + _R_ENGINE_ID : base + _R_ENGINE_ID + ENGINE_ID_MAX_LEN
        ] = eid_bytes.ljust(ENGINE_ID_MAX_LEN, b"\x00")
        for i, (src_blk, dest_blk) in enumerate(pairs):
            struct.pack_into(
                "<II",
                self.buf,
                base + _R_PAIRS + i * _PAIR_STRIDE,
                int(src_blk),
                int(dest_blk),
            )
        for i in range(len(pairs), MAX_BLOCK_PAIRS):
            struct.pack_into(
                "<II",
                self.buf,
                base + _R_PAIRS + i * _PAIR_STRIDE,
                0,
                0,
            )

        struct.pack_into("<Q", self.buf, base + _R_SEQ, head + 1)
        struct.pack_into("<Q", self.buf, _OFF_HEAD, head + 1)
        return True


class RestoreRingReader(_RingBase):
    def try_pop(self) -> Optional[dict]:
        if _RUST is not None:
            res = _RUST.try_pop_record(self.buf, self.capacity)
            if res is None:
                return None
            op, flags, slot, target, eid_bytes, pairs = res
            return {
                "op": int(op),
                "flags": int(flags),
                "counter_slot": int(slot),
                "counter_target": int(target),
                "src_engine_id": eid_bytes.decode("utf-8", errors="replace"),
                "block_pairs": [(int(s), int(d)) for s, d in pairs],
            }
        return self._try_pop_python()

    def _try_pop_python(self) -> Optional[dict]:
        tail = struct.unpack_from("<Q", self.buf, _OFF_TAIL)[0]
        head = struct.unpack_from("<Q", self.buf, _OFF_HEAD)[0]
        if tail >= head:
            return None
        slot = tail & self._mask
        base = HEADER_SIZE + slot * RECORD_SIZE
        seq = struct.unpack_from("<Q", self.buf, base + _R_SEQ)[0]
        if seq != tail + 1:
            return None

        op = struct.unpack_from("<B", self.buf, base + _R_OP)[0]
        flags = struct.unpack_from("<B", self.buf, base + _R_FLAGS)[0]
        counter_slot, counter_target, n_pairs = struct.unpack_from(
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

        n_pairs = min(int(n_pairs), MAX_BLOCK_PAIRS)
        pairs = []
        for i in range(n_pairs):
            s, d = struct.unpack_from(
                "<II",
                self.buf,
                base + _R_PAIRS + i * _PAIR_STRIDE,
            )
            pairs.append((int(s), int(d)))

        struct.pack_into("<Q", self.buf, _OFF_TAIL, tail + 1)
        return {
            "op": int(op),
            "flags": int(flags),
            "counter_slot": int(counter_slot),
            "counter_target": int(counter_target),
            "src_engine_id": src_engine_id,
            "block_pairs": pairs,
        }
