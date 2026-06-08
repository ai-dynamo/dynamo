# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS metrics surface — counters in shared memory + Prometheus scrape.

Production observability for the restore-ring + spill-ring pipeline.
Counters live in a single /dev/shm tmpfs file (`gms-metrics.bin`) so
ANY process (engine, daemon, monitoring sidecar) can attach and read
without RPC overhead.

Counters are 64-bit unsigned. Engine and daemon increment via simple
struct.pack_into (atomic on x86 for aligned 8-byte stores). Read-only
consumers (Prometheus exporter, dashboards) attach the same file
read-only.

Slot layout (`MetricsArray`, fixed schema for now — bump VERSION to
extend):

  0   ring_spill_pushed       # of spill records pushed into evict ring
  1   ring_spill_drops        # of spill records dropped (ring full)
  2   ring_restore_pushed     # of chunk records pushed into restore ring
  3   ring_restore_drops      # of restore records dropped (ring full)
  4   spill_rpc_calls         # of legacy sync spill RPCs (non-ring)
  5   spill_rpc_failures      # of legacy spill RPC failures
  6   restore_rpc_calls       # of legacy sync restore RPCs (non-ring)
  7   restore_rpc_failures    # of legacy restore RPC failures
  8   daemon_h2d_batch_ops    # # of cuMemcpyBatchAsync calls
  9   daemon_h2d_per_op       # # of (block × layer) ops processed
 10   daemon_h2d_fallback     # # of times batched H2D fell back to loop
 11   counter_reservations    # # of EngineCounterArray.reserve_slot calls
 12   counter_writes          # # of cuStreamWriteValue32 by the daemon
 13   chunk_table_hits        # cache hits (chunk_hash → spill record)
 14   chunk_table_misses      # cache misses (full prefix not in table)
 15   reserved
 ...
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
from typing import Optional

logger = logging.getLogger(__name__)


MAGIC = 0x47_4D_53_4D  # "GMSM"
VERSION = 1
HEADER_SIZE = 32
SLOT_SIZE = 8
N_SLOTS = 64  # 512 bytes of counters
_FILE_SIZE = HEADER_SIZE + N_SLOTS * SLOT_SIZE

# Counter slot indices.
RING_SPILL_PUSHED = 0
RING_SPILL_DROPS = 1
RING_RESTORE_PUSHED = 2
RING_RESTORE_DROPS = 3
SPILL_RPC_CALLS = 4
SPILL_RPC_FAILURES = 5
RESTORE_RPC_CALLS = 6
RESTORE_RPC_FAILURES = 7
DAEMON_H2D_BATCH_OPS = 8
DAEMON_H2D_PER_OP = 9
DAEMON_H2D_FALLBACK = 10
COUNTER_RESERVATIONS = 11
COUNTER_WRITES = 12
CHUNK_TABLE_HITS = 13
CHUNK_TABLE_MISSES = 14

_SLOT_NAMES = {
    RING_SPILL_PUSHED: "ring_spill_pushed",
    RING_SPILL_DROPS: "ring_spill_drops",
    RING_RESTORE_PUSHED: "ring_restore_pushed",
    RING_RESTORE_DROPS: "ring_restore_drops",
    SPILL_RPC_CALLS: "spill_rpc_calls",
    SPILL_RPC_FAILURES: "spill_rpc_failures",
    RESTORE_RPC_CALLS: "restore_rpc_calls",
    RESTORE_RPC_FAILURES: "restore_rpc_failures",
    DAEMON_H2D_BATCH_OPS: "daemon_h2d_batch_ops",
    DAEMON_H2D_PER_OP: "daemon_h2d_per_op",
    DAEMON_H2D_FALLBACK: "daemon_h2d_fallback",
    COUNTER_RESERVATIONS: "counter_reservations",
    COUNTER_WRITES: "counter_writes",
    CHUNK_TABLE_HITS: "chunk_table_hits",
    CHUNK_TABLE_MISSES: "chunk_table_misses",
}


def _default_path() -> str:
    return os.environ.get(
        "GMS_METRICS_FILE",
        "/dev/shm/gms-metrics.bin",
    )


class MetricsArray:
    """Shared-memory counter array. Open-or-create pattern."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd = -1
        self._buf: Optional[mmap.mmap] = None
        self._lock = threading.Lock()

    @classmethod
    def open(cls, path: Optional[str] = None) -> "MetricsArray":
        inst = cls(path or _default_path())
        inst._open_or_create()
        return inst

    def _open_or_create(self) -> None:
        first = not os.path.exists(self.path)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        if first:
            os.ftruncate(self._fd, _FILE_SIZE)
            # Initialize header.
            buf = mmap.mmap(
                self._fd,
                _FILE_SIZE,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
            struct.pack_into("<II", buf, 0, MAGIC, VERSION)
            # Zero counters.
            for i in range(N_SLOTS):
                struct.pack_into(
                    "<Q",
                    buf,
                    HEADER_SIZE + i * SLOT_SIZE,
                    0,
                )
            buf.close()

        self._buf = mmap.mmap(
            self._fd,
            _FILE_SIZE,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        magic, version = struct.unpack_from("<II", self._buf, 0)
        if magic != MAGIC:
            self.close()
            raise ValueError(
                f"GMS metrics file has wrong magic 0x{magic:x} at {self.path}",
            )
        if version != VERSION:
            self.close()
            raise ValueError(
                f"GMS metrics file version {version} != {VERSION}",
            )

    def inc(self, slot: int, by: int = 1) -> None:
        """Atomic-ish increment. Under CPython's GIL + x86 TSO this is
        safe for single-process atomicity; multi-process concurrent
        increments may race (slight undercount) but never corrupt."""
        if self._buf is None or not (0 <= slot < N_SLOTS):
            return
        # Read-modify-write — race-prone but bounded. For production
        # correctness use OS-level atomics if needed.
        off = HEADER_SIZE + slot * SLOT_SIZE
        cur = struct.unpack_from("<Q", self._buf, off)[0]
        struct.pack_into("<Q", self._buf, off, cur + int(by))

    def get(self, slot: int) -> int:
        if self._buf is None or not (0 <= slot < N_SLOTS):
            return 0
        off = HEADER_SIZE + slot * SLOT_SIZE
        return struct.unpack_from("<Q", self._buf, off)[0]

    def snapshot(self) -> dict[str, int]:
        return {name: self.get(slot) for slot, name in _SLOT_NAMES.items()}

    def prometheus_text(self) -> str:
        """Render counters in Prometheus text/plain format. Suitable
        for serving on a /metrics endpoint."""
        lines = []
        for slot, name in _SLOT_NAMES.items():
            full = f"gms_{name}"
            lines.append(f"# TYPE {full} counter")
            lines.append(f"{full} {self.get(slot)}")
        return "\n".join(lines) + "\n"

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                pass
            self._buf = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = -1


_LAZY: Optional[MetricsArray] = None
_LAZY_LOCK = threading.Lock()


def get_global() -> MetricsArray:
    """Cached process-wide handle to the metrics file. Lazy-creates on
    first access. Safe to call from any thread."""
    global _LAZY
    if _LAZY is not None:
        return _LAZY
    with _LAZY_LOCK:
        if _LAZY is None:
            _LAZY = MetricsArray.open()
    return _LAZY


def inc(slot: int, by: int = 1) -> None:
    """Convenience: increment via the global handle. No-op on failure
    (metrics must never break the hot path)."""
    try:
        get_global().inc(slot, by)
    except Exception:
        pass
