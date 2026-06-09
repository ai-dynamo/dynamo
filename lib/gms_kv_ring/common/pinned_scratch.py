# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny pool of pinned-host scratch buffers.

Used by the overlapped HBM→storage demote path: we want a pinned
host buffer for the async D2H + CPU CRC compute. Pinned because:
  - Async D2H from device to PAGEABLE host falls back to a
    synchronous path on the hot stream (silently makes the
    overlap useless).
  - Pinned D2H runs at PCIe bandwidth (~25 GB/s Gen4) and is
    properly stream-ordered.

The pool is process-wide and keyed by size. Each scratch buffer
is `cudaHostAlloc`'d once and reused. We never free — the buffers
are reclaimed at process exit. For typical KV-block sizes (tens
of KB to a few MB) this caps RAM use at "1 buffer per distinct
size we've seen," which is small.

`acquire(size)` returns a context manager that yields the buffer
address. The buffer is parked back on the pool on exit."""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Generator

logger = logging.getLogger(__name__)


_lock = threading.Lock()
# size_bucket -> list of (host_ptr, allocated_size) free buffers
_pool: dict[int, list[int]] = {}


def _size_bucket(size: int) -> int:
    """Round up to a power-of-two bucket to limit the number of
    distinct sizes we cache. A 1.8 MB ask gets a 2 MB buffer."""
    if size <= 1:
        return 1
    return 1 << (int(size - 1).bit_length())


def _alloc_pinned(size: int) -> int:
    """One-shot cudaHostAlloc of `size` bytes. Returns the host VA."""
    from cuda.bindings import runtime as rt

    err, ptr = rt.cudaHostAlloc(int(size), 0)
    if err != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaHostAlloc({size}) failed: {err}")
    return int(ptr)


@contextlib.contextmanager
def acquire(size: int) -> Generator[int, None, None]:
    """Yield a pinned host pointer with at least `size` bytes. The
    pointer is returned to the pool on exit, NOT freed."""
    bucket = _size_bucket(size)
    with _lock:
        free_list = _pool.setdefault(bucket, [])
        if free_list:
            ptr = free_list.pop()
        else:
            ptr = None
    if ptr is None:
        ptr = _alloc_pinned(bucket)
    try:
        yield ptr
    finally:
        with _lock:
            _pool[bucket].append(ptr)
