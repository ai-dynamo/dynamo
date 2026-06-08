# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-producer / single-consumer SHM ring buffer for async eviction
submissions from engine → offload daemon.

Replaces the synchronous spill RPC. The engine writes one
`EvictRecord` per evicted block (sub-µs enqueue: a few struct.pack_into
calls + an atomic-style head bump under the GIL); a daemon thread
drains the ring and performs the actual D2H. The RO-lock protocol
(already in place) handles correctness — the engine's allocator wrap
won't reuse a slot while its RO lock is held, regardless of whether
the daemon has finished spilling.

Layout (one file in /dev/shm, mmap'd by both processes):

  ┌─ Header (64B, cache-line aligned) ──────────────────────────┐
  │  magic        : u32   = 0x47_4D_53_45  ("GMSE")             │
  │  version      : u32   = 1                                    │
  │  capacity     : u32   record count                           │
  │  record_size  : u32   = RECORD_SIZE (256)                    │
  │  head_seq     : u64   monotonic; producer-only writes        │
  │  tail_seq     : u64   monotonic; consumer-only writes        │
  │  drops        : u64   producer-only; full-queue drop counter │
  │  padding      : 16B                                          │
  └──────────────────────────────────────────────────────────────┘
  ┌─ Record[0]   (256B) ────────────────────────────────────────┐
  │  seq          : u64   non-zero ⇒ ready for consumer          │
  │  op_kind      : u8                                           │
  │  flags        : u8                                           │
  │  _pad         : u16                                          │
  │  block_id     : u32                                          │
  │  n_ranges     : u32                                          │
  │  _pad2        : u32                                          │
  │  ipc_event    : 64B   raw cuIpcEventHandle (zero ⇒ none)     │
  │  ranges[14]   : each (u32 layer_idx, u32 size, u64 offset)   │
  │  total        : 256B                                         │
  ├─ Record[1] ... Record[capacity-1] ──────────────────────────┤
  └──────────────────────────────────────────────────────────────┘

Concurrency model:
  • Single producer (engine). `head_seq` is bumped only by the
    producer; under CPython's GIL this is naturally atomic.
  • Single consumer (daemon thread). `tail_seq` is bumped only by
    the consumer.
  • A record is "ready" when `record.seq == reserved_seq + 1`. The
    producer writes the payload BEFORE setting seq (release order on
    x86-TSO); the consumer reads seq BEFORE reading payload (acquire).
  • Cross-process: writes to mmap of a tmpfs file are visible to all
    mappers at hardware-cache-coherence boundaries. No msync needed.

Backpressure:
  If `head - tail >= capacity`, `enqueue()` returns False and bumps
  `drops` (visible for monitoring). Caller falls back to sync RPC for
  that one eviction, OR drops the spill (production deployments would
  size the ring to make this never happen).
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


MAGIC = 0x47_4D_53_45  # "GMSE"
VERSION = 1
HEADER_SIZE = 64
# Record layout budget (per-record bytes):
#   seq(8) + op(1)+flags(1)+pad(2) + block_id(4)+n_ranges(4)+pad(4) = 24
#   ipc_event[64]                                                    = 64
#   ranges[N], each 16B                                              = N*16
# RECORD_SIZE = 512 leaves room for 26 ranges (24 used + 2 spare).
# Most engines have ≤24 layer-tensors per block; raise this only if a
# real engine needs more.
RECORD_SIZE = 512
MAX_RANGES_PER_RECORD = 24
IPC_EVENT_HANDLE_LEN = 64

# Op kinds
OP_NOOP = 0
OP_EVICT = 1
OP_DRAIN = 2  # sentinel: daemon should ack and continue draining

# Header field offsets
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_CAPACITY = 8
_OFF_RECORD_SIZE = 12
_OFF_HEAD_SEQ = 16
_OFF_TAIL_SEQ = 24
_OFF_DROPS = 32
# 48..64 reserved

# Record field offsets
_R_SEQ = 0
_R_OP = 8
_R_FLAGS = 9
_R_BLOCK_ID = 12
_R_N_RANGES = 16
_R_IPC_EVENT = 24
_R_RANGES = 88
_RANGE_STRIDE = 16  # u32 + u32 + u64


def _file_size(capacity: int) -> int:
    return HEADER_SIZE + capacity * RECORD_SIZE


def create_ring(path: str, capacity: int = 8192) -> "EvictRingWriter":
    """Create a fresh ring file at `path` and return a writer."""
    if capacity <= 0 or (capacity & (capacity - 1)) != 0:
        raise ValueError(f"capacity must be a power of 2; got {capacity}")
    size = _file_size(capacity)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, size)
        buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        # Initialize header
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
        # Zero out records (already 0 from ftruncate of new file).
    finally:
        os.close(fd)
    return EvictRingWriter(path, buf, capacity)


def attach_ring_writer(path: str) -> "EvictRingWriter":
    """Attach to an existing ring file for writing (single producer)."""
    return _attach(path, EvictRingWriter)


def attach_ring_reader(path: str) -> "EvictRingReader":
    """Attach to an existing ring file for reading (single consumer)."""
    return _attach(path, EvictRingReader)


def _attach(path: str, cls):
    fd = os.open(path, os.O_RDWR)
    try:
        size = os.fstat(fd).st_size
        buf = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    magic, version, capacity, record_size, _, _, _ = struct.unpack_from(
        "<IIIIQQQ",
        buf,
        0,
    )
    if magic != MAGIC:
        buf.close()
        raise ValueError(f"bad magic 0x{magic:x}; not a GMS evict ring")
    if version != VERSION:
        buf.close()
        raise ValueError(f"version mismatch: {version} != {VERSION}")
    if record_size != RECORD_SIZE:
        buf.close()
        raise ValueError(f"record_size mismatch: {record_size} != {RECORD_SIZE}")
    return cls(path, buf, capacity)


# ---------------------------------------------------------------------------
# Writer (producer / engine side)
# ---------------------------------------------------------------------------


class EvictRingWriter:
    """SPSC ring buffer writer. Use from the engine's eviction hook.

    Thread-safe under the CPython GIL with respect to ONE producer
    thread. Do NOT share the writer across producer threads — the head
    bump assumes single-producer (no atomic CAS on Python int operations).
    """

    __slots__ = ("path", "_buf", "capacity", "_lock")

    def __init__(self, path: str, buf: mmap.mmap, capacity: int) -> None:
        self.path = path
        self._buf = buf
        self.capacity = capacity
        # In-process lock guards the head bump even though the GIL
        # makes most struct.pack_into calls atomic — defensive against
        # future use of a free-threaded Python build.
        self._lock = threading.Lock()

    def close(self) -> None:
        try:
            self._buf.close()
        except Exception:  # noqa: BLE001
            pass

    def head(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_HEAD_SEQ)[0]

    def tail(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_TAIL_SEQ)[0]

    def drops(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_DROPS)[0]

    def queue_depth(self) -> int:
        return self.head() - self.tail()

    def enqueue_evict(
        self,
        block_id: int,
        ipc_event_handle: Optional[bytes],
        ranges: Iterable[tuple[int, int, int]],
    ) -> bool:
        """Submit one EVICT record. Returns True on success, False if
        the ring is full (caller can fall back to sync RPC for this one
        eviction). Sub-µs in the happy path."""
        ranges_list = list(ranges)
        if len(ranges_list) > MAX_RANGES_PER_RECORD:
            # Too many ranges for one record — drop and let caller fall
            # back. Real engines have <14 layers; if this fires, the
            # record format needs widening.
            logger.warning(
                "evict_ring: %d ranges exceeds per-record cap %d; dropping",
                len(ranges_list),
                MAX_RANGES_PER_RECORD,
            )
            self._bump_drops()
            return False

        with self._lock:
            head = self.head()
            tail = self.tail()
            if head - tail >= self.capacity:
                self._bump_drops()
                return False
            slot_idx = head % self.capacity
            base = HEADER_SIZE + slot_idx * RECORD_SIZE

            # Payload first (release barrier is provided by the GIL +
            # x86 TSO ordering when we later write seq).
            struct.pack_into(
                "<BBxxII",
                self._buf,
                base + _R_OP,
                OP_EVICT,
                0,
                int(block_id) & 0xFFFFFFFF,
                len(ranges_list),
            )
            ipc_bytes = ipc_event_handle or b""
            if len(ipc_bytes) > IPC_EVENT_HANDLE_LEN:
                ipc_bytes = ipc_bytes[:IPC_EVENT_HANDLE_LEN]
            elif len(ipc_bytes) < IPC_EVENT_HANDLE_LEN:
                ipc_bytes = ipc_bytes.ljust(IPC_EVENT_HANDLE_LEN, b"\x00")
            self._buf[
                base + _R_IPC_EVENT : base + _R_IPC_EVENT + IPC_EVENT_HANDLE_LEN
            ] = ipc_bytes
            roff = base + _R_RANGES
            for i, (layer_idx, size, offset) in enumerate(ranges_list):
                struct.pack_into(
                    "<IIQ",
                    self._buf,
                    roff + i * _RANGE_STRIDE,
                    int(layer_idx) & 0xFFFFFFFF,
                    int(size) & 0xFFFFFFFF,
                    int(offset) & 0xFFFFFFFFFFFFFFFF,
                )
            # Zero unused ranges so the consumer doesn't read stale data
            for i in range(len(ranges_list), MAX_RANGES_PER_RECORD):
                struct.pack_into(
                    "<IIQ",
                    self._buf,
                    roff + i * _RANGE_STRIDE,
                    0,
                    0,
                    0,
                )

            # Publish: write seq LAST (the consumer waits for seq==head+1).
            struct.pack_into("<Q", self._buf, base + _R_SEQ, head + 1)
            # Bump head AFTER publishing the slot.
            struct.pack_into("<Q", self._buf, _OFF_HEAD_SEQ, head + 1)
            return True

    def _bump_drops(self) -> None:
        d = struct.unpack_from("<Q", self._buf, _OFF_DROPS)[0] + 1
        struct.pack_into("<Q", self._buf, _OFF_DROPS, d)


# ---------------------------------------------------------------------------
# Reader (consumer / daemon side)
# ---------------------------------------------------------------------------


class EvictRingReader:
    """SPSC ring buffer reader. Use from the daemon's drain thread.

    `drain_one()` returns at most one ready record (or None if empty).
    `drain_all(callback)` drains until empty, invoking callback per
    record."""

    __slots__ = ("path", "_buf", "capacity")

    def __init__(self, path: str, buf: mmap.mmap, capacity: int) -> None:
        self.path = path
        self._buf = buf
        self.capacity = capacity

    def close(self) -> None:
        try:
            self._buf.close()
        except Exception:  # noqa: BLE001
            pass

    def head(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_HEAD_SEQ)[0]

    def tail(self) -> int:
        return struct.unpack_from("<Q", self._buf, _OFF_TAIL_SEQ)[0]

    def queue_depth(self) -> int:
        return self.head() - self.tail()

    def drain_one(self) -> Optional[dict]:
        """Pop one ready record. Returns dict with keys
        `op`, `block_id`, `ipc_event`, `ranges`, or None if empty."""
        tail = self.tail()
        head = self.head()
        if tail >= head:
            return None
        slot_idx = tail % self.capacity
        base = HEADER_SIZE + slot_idx * RECORD_SIZE
        # Wait until the producer has finished writing this slot. In
        # practice the GIL + head-bump ordering means seq is already
        # set by the time we observe head > tail. Defensive busy-wait
        # bounded to a few iters.
        for _ in range(1024):
            seq = struct.unpack_from("<Q", self._buf, base + _R_SEQ)[0]
            if seq == tail + 1:
                break
        else:
            # Producer hasn't published yet despite head bump — treat
            # as transient, return None and let caller poll again.
            return None
        op, flags = struct.unpack_from("<BB", self._buf, base + _R_OP)
        block_id, n_ranges = struct.unpack_from("<II", self._buf, base + _R_BLOCK_ID)
        ipc_event = bytes(
            self._buf[base + _R_IPC_EVENT : base + _R_IPC_EVENT + IPC_EVENT_HANDLE_LEN]
        )
        ranges = []
        roff = base + _R_RANGES
        for i in range(min(int(n_ranges), MAX_RANGES_PER_RECORD)):
            layer_idx, size, offset = struct.unpack_from(
                "<IIQ",
                self._buf,
                roff + i * _RANGE_STRIDE,
            )
            ranges.append((int(layer_idx), int(size), int(offset)))
        # Mark slot consumed.
        struct.pack_into("<Q", self._buf, base + _R_SEQ, 0)
        struct.pack_into("<Q", self._buf, _OFF_TAIL_SEQ, tail + 1)
        return {
            "op": int(op),
            "flags": int(flags),
            "block_id": int(block_id),
            "ipc_event": ipc_event,
            "ranges": ranges,
        }

    def drain_all(self, callback, max_records: int = 1024) -> int:
        """Drain up to `max_records` ready records, calling
        `callback(record_dict)` per record. Returns the number drained."""
        n = 0
        while n < max_records:
            rec = self.drain_one()
            if rec is None:
                break
            try:
                callback(rec)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "evict_ring: consumer callback failed",
                    exc_info=True,
                )
            n += 1
        return n


# ---------------------------------------------------------------------------
# Eventfd helpers — optional wakeup signal so the daemon can block-wait
# instead of polling. SCM_RIGHTS-shared at attach time.
# ---------------------------------------------------------------------------


def make_eventfd() -> int:
    """Create a new eventfd. Returns the fd. Use `eventfd_write(1)` to
    signal, `eventfd_read()` to consume.

    Returns -1 if eventfd isn't available (unlikely on modern Linux)."""
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    # EFD_CLOEXEC=0x80000, EFD_NONBLOCK=0x800, EFD_SEMAPHORE=1
    EFD_CLOEXEC = 0o2000000
    EFD_NONBLOCK = 0o0004000
    fd = libc.eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)
    if fd < 0:
        return -1
    return int(fd)


def eventfd_write(fd: int, n: int = 1) -> None:
    if fd < 0:
        return
    try:
        os.write(fd, int(n).to_bytes(8, "little"))
    except BlockingIOError:
        # Counter saturated — already signalled. Fine.
        pass
    except OSError:
        pass


def eventfd_read_nonblock(fd: int) -> int:
    """Returns the counter value (and resets it) or 0 if nothing
    pending. Non-blocking."""
    if fd < 0:
        return 0
    try:
        data = os.read(fd, 8)
        return int.from_bytes(data, "little")
    except BlockingIOError:
        return 0
    except OSError:
        return 0
