# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host memory for the prefix cache's host tier: pinned rows, and batched
copies of KV block rows between the GPU cache and them.

A KV block is one row of ``cache_engine.gpu_blocks`` (all layers packed), so a
block moves with one copy of ``row_bytes``. Copies are ordered with the work
already queued on the current stream and before any work queued after them,
like the snapshot copies: a block freed on the CPU may still be written by a
step in flight, and the next prefill reads what a load wrote.

Imported lazily by rank workers: no ``dynamo._core``; torch only for torch
tensors.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import mmap
import sys
import weakref
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# One pinned allocation per chunk: the clamp re-checks the host between them.
CHUNK_BYTES = 1 << 32
# Leave at least this share of the host's memory available for everything else
# (Engram tables, page cache, the other ranks' pools allocating concurrently).
HOST_RESERVE_FRACTION = 0.1


def mem_available() -> tuple[int, int]:
    """(MemAvailable, MemTotal) in bytes from ``/proc/meminfo``."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemAvailable", "MemTotal"):
                    values[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    return values.get("MemAvailable", 1 << 62), values.get("MemTotal", 1 << 62)


def _torch() -> Any:
    return importlib.import_module("torch")


def _unregister(ptrs: list[int]) -> None:
    # Before the mappings go: a later mapping at the same address would fail
    # to register.
    torch = sys.modules.get("torch")
    if torch is None:
        return
    for ptr in ptrs:
        try:
            torch.cuda.cudart().cudaHostUnregister(ptr)
        except Exception:  # interpreter exit: CUDA may be gone already
            return


def row_nbytes(row: Any) -> int:
    """Bytes of one row (a numpy array or a torch tensor)."""
    if isinstance(row, np.ndarray):
        return int(row.nbytes)
    return int(row.numel() * row.element_size())


class PinnedRows:
    """Up to ``count`` host rows shaped like the rows of ``like``.

    Allocated in chunks of at most :data:`CHUNK_BYTES`, pinned for a torch
    ``like``: each chunk is its own anonymous mapping (page aligned, so no two
    registrations overlap) pinned with ``cudaHostRegister`` (PyTorch's pinned
    allocator would round it up to a power of two). Allocation stops early when the host
    would be left with less than :data:`HOST_RESERVE_FRACTION` of its memory
    available; :attr:`count` is what was allocated.
    """

    def __init__(self, like: Any, count: int) -> None:
        self.numpy = isinstance(like, np.ndarray)
        self._row_shape = tuple(like.shape[1:])
        self._dtype = like.dtype
        self.row_bytes = row_nbytes(like[0])
        self.per_chunk = max(1, CHUNK_BYTES // self.row_bytes)
        self.chunks: list[Any] = []
        self._maps: list[mmap.mmap] = []
        self._registered: list[int] = []
        weakref.finalize(self, _unregister, self._registered)
        self.count = 0
        while self.count < count:
            n = min(count - self.count, self.per_chunk)
            available, total = mem_available()
            if available - n * self.row_bytes < HOST_RESERVE_FRACTION * total:
                logger.warning(
                    "Host tier clamped to %d of %d rows (%.1f GiB): %.1f GiB "
                    "available of %.1f GiB",
                    self.count,
                    count,
                    self.count * self.row_bytes / 2**30,
                    available / 2**30,
                    total / 2**30,
                )
                break
            self.chunks.append(self._alloc(n))
            self.count += n

    def _alloc(self, n: int) -> Any:
        if self.numpy:
            return np.zeros((n, *self._row_shape), dtype=self._dtype)
        torch = _torch()
        size = n * self.row_bytes
        self._maps.append(mmap.mmap(-1, size))
        chunk = torch.frombuffer(self._maps[-1], dtype=torch.uint8, count=size)
        chunk = chunk.view(self._dtype).view(n, *self._row_shape)
        err = torch.cuda.cudart().cudaHostRegister(chunk.data_ptr(), size, 0)
        if int(getattr(err, "value", err)) != 0:
            raise RuntimeError(f"cudaHostRegister of {n} host tier rows failed: {err}")
        self._registered.append(chunk.data_ptr())
        return chunk

    @property
    def nbytes(self) -> int:
        return self.count * self.row_bytes

    def row(self, i: int) -> Any:
        return self.chunks[i // self.per_chunk][i % self.per_chunk]

    def ptr(self, i: int) -> int:
        chunk = self.chunks[i // self.per_chunk]
        return chunk.data_ptr() + (i % self.per_chunk) * self.row_bytes


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


# CU_MEMCPY_SRC_ACCESS_ORDER_STREAM: read the source in stream order.
_SRC_ACCESS_ORDER_STREAM = 1


def _batch_memcpy() -> Optional[Any]:
    """``cuMemcpyBatchAsync`` (CUDA 12.8 ABI, as vLLM's simple_kv_offload uses
    it): every block of a store or load in one driver call."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        get = lib.cuGetProcAddress_v2
    except (OSError, AttributeError):
        return None
    get.restype = ctypes.c_int
    get.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int),
    ]
    fn, status = ctypes.c_void_p(), ctypes.c_int()
    if get(b"cuMemcpyBatchAsync", ctypes.byref(fn), 12080, 0, ctypes.byref(status)):
        return None
    if not fn.value:
        return None
    proto = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,  # dsts
        ctypes.c_void_p,  # srcs
        ctypes.c_void_p,  # sizes
        ctypes.c_size_t,  # count
        ctypes.c_void_p,  # attrs
        ctypes.c_void_p,  # attrsIdxs
        ctypes.c_size_t,  # numAttrs
        ctypes.c_void_p,  # failIdx
        ctypes.c_void_p,  # stream
    )
    return proto(fn.value)


class HostBlockPool:
    """Host rows for GPU KV blocks, a free list, and the copies between them."""

    def __init__(self, gpu_blocks: Any, count: int) -> None:
        self._gpu = gpu_blocks
        self._rows = PinnedRows(gpu_blocks, count)
        self.capacity = self._rows.count
        self._free = list(range(self.capacity - 1, -1, -1))
        self._batch: Optional[Any] = None
        self._stream: Any = None
        if not self._rows.numpy:
            if not gpu_blocks[0].is_contiguous():
                raise ValueError("host tier needs contiguous KV block rows")
            self._row_bytes = self._rows.row_bytes
            self._gpu_base = gpu_blocks.data_ptr()
            self._gpu_pitch = gpu_blocks.stride(0) * gpu_blocks.element_size()
            self._batch = _batch_memcpy()
            if self._batch is None:
                logger.warning(
                    "cuMemcpyBatchAsync unavailable: the host tier copies one "
                    "block at a time"
                )
            else:
                self._stream = _torch().cuda.Stream(device=gpu_blocks.device)

    @property
    def nbytes(self) -> int:
        return self._rows.nbytes

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> Optional[int]:
        return self._free.pop() if self._free else None

    def release(self, slot: int) -> None:
        self._free.append(slot)

    def store(self, gpu_ids: list[int], slots: list[int]) -> Optional[Any]:
        """GPU block rows -> host rows (before the blocks are freed).

        Returns a ``torch.cuda.Event`` that completes when the copy has: the
        host rows may be read (by an SSD write) once it is synchronized. None
        on the numpy path, where the copy is done on return.
        """
        return self._copy(slots, gpu_ids, to_host=True)

    def load(self, slots: list[int], gpu_ids: list[int]) -> None:
        """Host rows -> GPU block rows (before the prefill that reads them)."""
        self._copy(slots, gpu_ids, to_host=False)

    def _copy(
        self, slots: list[int], gpu_ids: list[int], *, to_host: bool
    ) -> Optional[Any]:
        if not slots:
            return None
        if self._rows.numpy:
            for slot, gid in zip(slots, gpu_ids):
                if to_host:
                    self._rows.row(slot)[...] = self._gpu[gid]
                else:
                    self._gpu[gid] = self._rows.row(slot)
            return None
        torch = _torch()
        current = torch.cuda.current_stream(self._gpu.device)
        if self._batch is None:
            with torch.cuda.stream(current):
                for slot, gid in zip(slots, gpu_ids):
                    host, gpu = self._rows.row(slot), self._gpu[gid]
                    dst, src = (host, gpu) if to_host else (gpu, host)
                    dst.copy_(src, non_blocking=True)
            return self._record(current) if to_host else None
        host = np.fromiter((self._rows.ptr(s) for s in slots), np.uint64, len(slots))
        gpu = np.asarray(gpu_ids, dtype=np.uint64) * np.uint64(self._gpu_pitch)
        gpu += np.uint64(self._gpu_base)
        dsts, srcs = (host, gpu) if to_host else (gpu, host)
        sizes = np.full(len(slots), self._row_bytes, dtype=np.uint64)
        attrs = _CUmemcpyAttributes(srcAccessOrder=_SRC_ACCESS_ORDER_STREAM)
        index, fail = ctypes.c_size_t(0), ctypes.c_size_t(0)
        # The driver call needs a stream other than the legacy default one:
        # fence it to the current stream on both sides.
        stream = self._stream
        stream.wait_stream(current)
        err = self._batch(
            dsts.ctypes.data,
            srcs.ctypes.data,
            sizes.ctypes.data,
            len(slots),
            ctypes.byref(attrs),
            ctypes.byref(index),
            1,
            ctypes.byref(fail),
            ctypes.c_void_p(stream.cuda_stream),
        )
        event = self._record(stream) if to_host else None
        current.wait_stream(stream)
        if err:
            raise RuntimeError(
                f"cuMemcpyBatchAsync failed with CUresult {err} at copy {fail.value}"
            )
        return event

    @staticmethod
    def _record(stream: Any) -> Any:
        event = _torch().cuda.Event()
        event.record(stream)
        return event
