# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evict ring — SPSC SHM ring for engine→daemon spill requests.

Engine pushes one EvictRecord per evicted block. Daemon consumer
drains in the background and does the actual cuMemcpyAsync D2H.

Layout (one /dev/shm file, mmap'd by both processes):

  ┌─ Header (64B, cache-line aligned) ──────────────────────────┐
  │  magic        : u32  = 0x47_4D_53_45  ("GMSE")              │
  │  version      : u32  = 1                                     │
  │  capacity     : u32  power-of-2 record count                 │
  │  record_size  : u32  = 512                                   │
  │  head_seq     : u64  producer-only writes                    │
  │  tail_seq     : u64  consumer-only writes                    │
  │  drops        : u64  full-queue drop counter                 │
  │  padding      : 16B                                          │
  └──────────────────────────────────────────────────────────────┘
  ┌─ Record (512B) ─────────────────────────────────────────────┐
  │  seq          : u64  (non-zero ⇒ ready for consumer)         │
  │  block_id     : u64                                          │
  │  n_ranges     : u32                                          │
  │  flags        : u32                                          │
  │  ipc_event    : 64B  cudaIpcEventHandle (zeros = none)       │
  │  ranges[24]   : each (u32 layer_idx, u32 size, u64 offset)   │
  └──────────────────────────────────────────────────────────────┘

Concurrency:
  • Single producer (one engine eviction hook thread).
  • Single consumer (daemon's evict ring drain thread).
  • `head_seq` written only by producer; `tail_seq` only by consumer.
  • Payload is written first; seq published last (release ordering).
  • x86-TSO + GIL gives correct ordering in Python; Rust hot path
    uses explicit AtomicU64::store(Release).

Backpressure:
  Ring full → push returns False + bumps `drops`. Caller's policy
  decides (typically: log + skip; the engine's allocator will reuse
  the slot anyway, just no future cache hit).
"""

from __future__ import annotations

import mmap
import os
import struct
from typing import Iterable, Optional

# Try the native Rust hot path. Fall back to pure Python.
try:
    import gms_rust_ring as _RUST  # type: ignore[import-not-found]
except Exception:
    _RUST = None


MAGIC = 0x4754_4D53  # bytes "GMSE" in little-endian
VERSION = 1
HEADER_SIZE = 64
RECORD_SIZE = 512
MAX_RANGES = 24

_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_CAPACITY = 8
_OFF_RECORD_SIZE = 12
_OFF_HEAD = 16
_OFF_TAIL = 24
_OFF_DROPS = 32

_R_SEQ = 0
_R_BLOCK_ID = 8
_R_N_RANGES = 16
_R_FLAGS = 20
_R_IPC_EVENT = 24
_R_RANGES = 88
_RANGE_STRIDE = 16
# Evict-ack fields (placed in the record's tail slack, after the
# 24-range slot region which ends at _R_RANGES + 24*16 = 472):
#   counter_slot   = u32 @ 472   — engine's counter slot for ack
#   counter_target = u32 @ 476   — daemon writes this on success,
#                                  target+1 on failure. 0/0 means
#                                  the engine doesn't want an ack
#                                  (legacy fire-and-forget path).
#   generation     = u64 @ 480   — source block generation for host-tier
#                                  restore validation. 0 means legacy/no check.
_R_COUNTER_SLOT = 472
_R_COUNTER_TARGET = 476
_R_GENERATION = 480


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def create_ring(path: str, capacity: int = 8192) -> "EvictRingWriter":
    """Create a fresh ring file at `path`. Capacity must be power of 2."""
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
    return EvictRingWriter(path, buf, capacity)


def attach_writer(path: str) -> "EvictRingWriter":
    return _attach(path, EvictRingWriter)


def attach_reader(path: str) -> "EvictRingReader":
    return _attach(path, EvictRingReader)


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
        error = f"bad magic 0x{magic:x}: not a GMSE ring"
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


class EvictRingWriter(_RingBase):
    """Engine-side producer. Single thread only.

    Use `push(block_id, ranges, ipc_event_handle)` to enqueue one
    eviction. Returns True on success; False if the ring is full
    (drops counter incremented).
    """

    def push(
        self,
        block_id: int,
        ranges: Iterable[tuple],
        ipc_event_handle: bytes = b"",
        flags: int = 0,
        counter_slot: int = 0,
        counter_target: int = 0,
        generation: int = 0,
    ) -> bool:
        """Push one evict record. Optional `counter_slot`/`counter_target`
        request a host-side completion ack: the daemon writes
        `counter_target` to counter[counter_slot] after the D2H is
        synced, or `counter_target+1` on failure. Pass 0/0 to opt out
        (legacy fire-and-forget path, unsafe if the engine plans to
        reuse the HBM slot before the DMA completes)."""
        ranges_list = list(ranges)
        if len(ranges_list) > MAX_RANGES:
            raise ValueError(
                f"too many ranges per record: {len(ranges_list)} > {MAX_RANGES}",
            )
        if len(ipc_event_handle) not in (0, 64):
            raise ValueError(
                f"ipc_event_handle must be 0 or 64 bytes, got {len(ipc_event_handle)}",
            )

        head = struct.unpack_from("<Q", self.buf, _OFF_HEAD)[0]
        tail = struct.unpack_from("<Q", self.buf, _OFF_TAIL)[0]
        if head - tail >= self.capacity:
            drops = struct.unpack_from("<Q", self.buf, _OFF_DROPS)[0]
            struct.pack_into("<Q", self.buf, _OFF_DROPS, drops + 1)
            return False

        slot = head & self._mask
        base = HEADER_SIZE + slot * RECORD_SIZE

        # Zero the seq first — a half-written record must not be seen.
        struct.pack_into("<Q", self.buf, base + _R_SEQ, 0)
        struct.pack_into(
            "<QII",
            self.buf,
            base + _R_BLOCK_ID,
            int(block_id),
            len(ranges_list),
            int(flags) & 0xFFFFFFFF,
        )
        # ipc_event field — zero-fill then copy.
        self.buf[base + _R_IPC_EVENT : base + _R_IPC_EVENT + 64] = (
            ipc_event_handle.ljust(64, b"\x00") if ipc_event_handle else b"\x00" * 64
        )
        for i, (layer_idx, size, offset) in enumerate(ranges_list):
            struct.pack_into(
                "<IIQ",
                self.buf,
                base + _R_RANGES + i * _RANGE_STRIDE,
                int(layer_idx),
                int(size),
                int(offset),
            )
        # Zero unused range slots (defensive).
        for i in range(len(ranges_list), MAX_RANGES):
            struct.pack_into(
                "<IIQ",
                self.buf,
                base + _R_RANGES + i * _RANGE_STRIDE,
                0,
                0,
                0,
            )
        # Evict-ack fields.
        struct.pack_into(
            "<II",
            self.buf,
            base + _R_COUNTER_SLOT,
            int(counter_slot) & 0xFFFFFFFF,
            int(counter_target) & 0xFFFFFFFF,
        )
        struct.pack_into(
            "<Q",
            self.buf,
            base + _R_GENERATION,
            int(generation) & 0xFFFFFFFFFFFFFFFF,
        )

        # Release: seq AFTER payload. x86-TSO + GIL keeps this ordered.
        struct.pack_into("<Q", self.buf, base + _R_SEQ, head + 1)
        struct.pack_into("<Q", self.buf, _OFF_HEAD, head + 1)
        return True


class EvictRingReader(_RingBase):
    """Daemon-side consumer. Single thread only.

    `try_pop()` returns the next record or None if empty.
    """

    def try_pop(self) -> Optional[dict]:
        tail = struct.unpack_from("<Q", self.buf, _OFF_TAIL)[0]
        head = struct.unpack_from("<Q", self.buf, _OFF_HEAD)[0]
        if tail >= head:
            return None
        slot = tail & self._mask
        base = HEADER_SIZE + slot * RECORD_SIZE
        seq = struct.unpack_from("<Q", self.buf, base + _R_SEQ)[0]
        if seq != tail + 1:
            return None  # producer hasn't released yet

        block_id, n_ranges, flags = struct.unpack_from(
            "<QII",
            self.buf,
            base + _R_BLOCK_ID,
        )
        ipc_event = bytes(self.buf[base + _R_IPC_EVENT : base + _R_IPC_EVENT + 64])
        if ipc_event == b"\x00" * 64:
            ipc_event = b""
        n_ranges = min(int(n_ranges), MAX_RANGES)
        ranges = []
        for i in range(n_ranges):
            li, sz, off = struct.unpack_from(
                "<IIQ",
                self.buf,
                base + _R_RANGES + i * _RANGE_STRIDE,
            )
            ranges.append((int(li), int(sz), int(off)))

        counter_slot, counter_target = struct.unpack_from(
            "<II",
            self.buf,
            base + _R_COUNTER_SLOT,
        )
        (generation,) = struct.unpack_from("<Q", self.buf, base + _R_GENERATION)
        struct.pack_into("<Q", self.buf, _OFF_TAIL, tail + 1)
        return {
            "block_id": int(block_id),
            "ranges": ranges,
            "flags": int(flags),
            "ipc_event": ipc_event,
            "counter_slot": int(counter_slot),
            "counter_target": int(counter_target),
            "generation": int(generation),
        }
