# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import ctypes
import ctypes.util
import errno
import json
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

from gpu_memory_service.snapshot.model import AllocationEntry, SaveManifest


class ShardWriter:
    """Packs allocation bytes sequentially into large binary shard files.

    This is a single-threaded utility for streaming writes.  The parallel save
    path in GMSStorageClient._write_shards assigns allocations to shards via
    plan_shard_layout and writes each shard file concurrently, so it does not
    use ShardWriter directly.  ShardWriter is kept as a public utility for
    callers that want a simple sequential writer.
    """

    def __init__(self, shards_dir: str, shard_size_bytes: int = 4 * 1024**3) -> None:
        self._shards_dir = shards_dir
        self._shard_size = shard_size_bytes
        self._shard_idx = -1
        self._current_offset = 0
        self._current_file: Optional[Any] = None
        self._current_rel_path: str = ""
        os.makedirs(shards_dir, exist_ok=True)

    def _roll_shard(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
        self._shard_idx += 1
        filename = f"shard_{self._shard_idx:04d}.bin"
        abs_path = os.path.join(self._shards_dir, filename)
        self._current_file = open(abs_path, "wb")
        self._current_rel_path = os.path.join("shards", filename)
        self._current_offset = 0

    def write(self, tensor: torch.Tensor) -> Tuple[str, int]:
        cpu = tensor.cpu() if hasattr(tensor, "is_cuda") and tensor.is_cuda else tensor
        if hasattr(cpu, "is_contiguous") and not cpu.is_contiguous():
            cpu = cpu.contiguous()
        arr = cpu.numpy()
        size = arr.nbytes
        if self._current_file is None or (
            self._current_offset > 0 and self._current_offset + size > self._shard_size
        ):
            self._roll_shard()

        offset = self._current_offset
        arr.tofile(self._current_file)
        self._current_offset += size
        return self._current_rel_path, offset

    def close(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def read_shard_sequential(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    device: int,
    *,
    pin_memory: bool = False,
    os_module=os,
    np_module=None,
    torch_module=None,
    logger=None,
) -> Dict[str, torch.Tensor]:
    """Read one shard file front-to-back without seeking."""
    if np_module is None or torch_module is None:
        raise RuntimeError("numpy and torch modules are required to read shards")

    result: Dict[str, torch.Tensor] = {}
    device_str = f"cuda:{device}" if device >= 0 else "cpu"

    if abs_path.endswith(".pt"):
        if len(sorted_entries) != 1:
            raise RuntimeError(
                f"Expected exactly 1 entry for legacy .pt file, got "
                f"{len(sorted_entries)}: {abs_path}"
            )
        entry = sorted_entries[0]
        result[entry.allocation_id] = torch_module.load(
            abs_path,
            weights_only=True,
            map_location=device_str,
        )
        return result

    odirect_flag = getattr(os_module, "O_DIRECT", None)
    if odirect_flag is not None:
        fd: Optional[int] = None
        done = 0
        try:
            total_size = sum(entry.aligned_size for entry in sorted_entries)
            # Avoid torch.empty(pin_memory=True): cudaHostAlloc is ~1-3 s/GiB
            # and dominates wall time.  Plain numpy gives good throughput since
            # PCIe H2D bandwidth far exceeds network disk bandwidth.
            shard_t = None
            arr = np_module.empty(total_size, dtype=np_module.uint8)

            fd = os_module.open(abs_path, os_module.O_RDONLY | odirect_flag)
            try:
                mv = memoryview(arr)
                try:
                    while done < total_size:
                        read = os_module.readv(fd, [mv[done:]])
                        if read == 0:
                            raise RuntimeError(
                                f"Unexpected EOF in O_DIRECT read from {abs_path}: "
                                f"got {done} of {total_size} bytes"
                            )
                        done += read
                finally:
                    mv.release()
            finally:
                os_module.close(fd)

            offset = 0
            for entry in sorted_entries:
                size = entry.aligned_size
                if shard_t is not None:
                    tensor = shard_t[offset : offset + size]
                else:
                    tensor = torch_module.from_numpy(arr[offset : offset + size])
                if device >= 0:
                    tensor = tensor.to(device_str)
                result[entry.allocation_id] = tensor
                offset += size
            return result
        except OSError as exc:
            fallback_errnos = {errno.EINVAL, errno.EOPNOTSUPP}
            if fd is not None and exc.errno not in fallback_errnos:
                raise
            result.clear()
            if logger is not None:
                if fd is None:
                    logger.debug(
                        "O_DIRECT unsupported on %s (errno %s); using buffered reads",
                        abs_path,
                        exc.errno,
                    )
                else:
                    logger.debug(
                        "O_DIRECT read on %s hit EINVAL after %d/%d bytes; using buffered reads",
                        abs_path,
                        done,
                        total_size,
                    )

    if sorted_entries and sorted_entries[0].tensor_offset != 0:
        raise RuntimeError(
            f"Buffered shard read requires entries starting at offset 0, "
            f"got {sorted_entries[0].tensor_offset} in {abs_path}"
        )
    with open(abs_path, "rb") as handle:
        for entry in sorted_entries:
            raw = handle.read(entry.aligned_size)
            if len(raw) != entry.aligned_size:
                raise RuntimeError(
                    f"Short read from {abs_path} at offset {entry.tensor_offset}: "
                    f"expected {entry.aligned_size} bytes, got {len(raw)}"
                )
            arr = np_module.frombuffer(raw, dtype=np_module.uint8).copy()
            tensor = torch_module.from_numpy(arr)
            if device >= 0:
                tensor = tensor.to(device_str)
            result[entry.allocation_id] = tensor
    return result


def decode_metadata(raw_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "allocation_id": entry["allocation_id"],
            "offset_bytes": int(entry["offset_bytes"]),
            "value": base64.b64decode(entry["value"]),
        }
        for key, entry in raw_meta.items()
    }


def group_entries_by_shard(
    allocations: List[AllocationEntry],
) -> Dict[str, List[AllocationEntry]]:
    groups: Dict[str, List[AllocationEntry]] = defaultdict(list)
    for entry in allocations:
        groups[entry.tensor_file].append(entry)
    for entries in groups.values():
        entries.sort(key=lambda entry: entry.tensor_offset)
    return dict(groups)


def plan_shard_layout(
    allocations_info: List[Dict[str, Any]],
    shard_size_bytes: int,
) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    shard_idx = -1
    current_offset = 0
    started = False
    for alloc in allocations_info:
        size = int(alloc["aligned_size"])
        if not started or (
            current_offset > 0 and current_offset + size > shard_size_bytes
        ):
            shard_idx += 1
            current_offset = 0
            started = True
        result.append((shard_idx, current_offset))
        current_offset += size
    return result


def _put_entry(
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, "torch.Tensor"]]],
    entry: AllocationEntry,
    tensor: "torch.Tensor",
    cancel_event: Optional[threading.Event],
    abs_path: str,
    stats: Optional[Any] = None,
) -> None:
    """Put one entry into the work queue, respecting cancellation."""
    t0 = time.monotonic()
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError(f"shard read cancelled: {abs_path}")
        try:
            work_q.put((entry, tensor), timeout=0.1)
            if stats is not None:
                stats.record_queue_put(time.monotonic() - t0)
                stats.record_queued_entry(entry.aligned_size)
            return
        except queue.Full:
            pass


# 64 MiB chunks for parallel preadv — gives high effective iodepth on NFS
# while keeping each syscall large enough to amortize overhead.
_CHUNK_SIZE = 64 * 1024 * 1024
# How many preadv calls to keep in-flight per shard.  On Vast NFS each
# outstanding preadv becomes a separate NFS READ RPC, so higher iodepth
# means more network-level parallelism from a single file descriptor.
_IO_DEPTH = 16


class _IOCB(ctypes.Structure):
    _fields_ = [
        ("aio_data", ctypes.c_uint64),
        ("aio_key", ctypes.c_uint32),
        ("aio_rw_flags", ctypes.c_uint32),
        ("aio_lio_opcode", ctypes.c_uint16),
        ("aio_reqprio", ctypes.c_int16),
        ("aio_fildes", ctypes.c_uint32),
        ("aio_buf", ctypes.c_uint64),
        ("aio_nbytes", ctypes.c_uint64),
        ("aio_offset", ctypes.c_int64),
        ("aio_reserved2", ctypes.c_uint64),
        ("aio_flags", ctypes.c_uint32),
        ("aio_resfd", ctypes.c_uint32),
    ]


class _IOEvent(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_uint64),
        ("obj", ctypes.c_uint64),
        ("res", ctypes.c_int64),
        ("res2", ctypes.c_int64),
    ]


class _AIOJob:
    __slots__ = ("offset", "size", "done", "iocb", "submitted_at")

    def __init__(self, offset: int, size: int) -> None:
        self.offset = offset
        self.size = size
        self.done = 0
        self.iocb: Optional[_IOCB] = None
        self.submitted_at = 0.0


class _NativeAIO:
    IOCB_CMD_PREAD = 0
    _SYS_IO_SETUP_X86_64 = 206
    _SYS_IO_DESTROY_X86_64 = 207
    _SYS_IO_GETEVENTS_X86_64 = 208
    _SYS_IO_SUBMIT_X86_64 = 209

    def __init__(self, depth: int) -> None:
        self._lib = self._load_libaio()
        self._libc = None
        self._syscall = None
        if self._lib is None:
            if os.uname().machine != "x86_64":
                raise RuntimeError(
                    "libaio is unavailable and syscall AIO fallback is only "
                    "implemented for x86_64"
                )
            self._libc = ctypes.CDLL(None, use_errno=True)
            self._syscall = self._libc.syscall
            self._syscall.restype = ctypes.c_long
        else:
            self._lib.io_setup.argtypes = [
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_ulong),
            ]
            self._lib.io_setup.restype = ctypes.c_int
            self._lib.io_destroy.argtypes = [ctypes.c_ulong]
            self._lib.io_destroy.restype = ctypes.c_int
            self._lib.io_submit.argtypes = [
                ctypes.c_ulong,
                ctypes.c_long,
                ctypes.POINTER(ctypes.POINTER(_IOCB)),
            ]
            self._lib.io_submit.restype = ctypes.c_int
            self._lib.io_getevents.argtypes = [
                ctypes.c_ulong,
                ctypes.c_long,
                ctypes.c_long,
                ctypes.POINTER(_IOEvent),
                ctypes.c_void_p,
            ]
            self._lib.io_getevents.restype = ctypes.c_int

        self._ctx = ctypes.c_ulong(0)
        ret = self._io_setup(depth, ctypes.byref(self._ctx))
        if ret < 0:
            raise OSError(-ret, os.strerror(-ret))

    @staticmethod
    def _load_libaio() -> Optional[Any]:
        candidates = [
            ctypes.util.find_library("aio"),
            "libaio.so.1",
            "libaio.so.1t64",
            "/lib/x86_64-linux-gnu/libaio.so.1",
            "/lib/x86_64-linux-gnu/libaio.so.1t64",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                pass
        return None

    def _syscall_result(self, *args: Any) -> int:
        assert self._syscall is not None
        ret = int(self._syscall(*args))
        if ret == -1:
            return -ctypes.get_errno()
        return ret

    def _io_setup(self, depth: int, ctx: Any) -> int:
        if self._lib is not None:
            return int(self._lib.io_setup(depth, ctx))
        return self._syscall_result(
            self._SYS_IO_SETUP_X86_64,
            ctypes.c_uint(depth),
            ctx,
        )

    def _io_destroy(self, ctx: ctypes.c_ulong) -> int:
        if self._lib is not None:
            return int(self._lib.io_destroy(ctx))
        return self._syscall_result(self._SYS_IO_DESTROY_X86_64, ctx)

    def _io_submit(
        self,
        ctx: ctypes.c_ulong,
        count: int,
        iocbs: Any,
    ) -> int:
        if self._lib is not None:
            return int(self._lib.io_submit(ctx, count, iocbs))
        return self._syscall_result(
            self._SYS_IO_SUBMIT_X86_64,
            ctx,
            ctypes.c_long(count),
            iocbs,
        )

    def _io_getevents(
        self,
        ctx: ctypes.c_ulong,
        min_events: int,
        max_events: int,
        events: Any,
    ) -> int:
        if self._lib is not None:
            return int(
                self._lib.io_getevents(ctx, min_events, max_events, events, None)
            )
        return self._syscall_result(
            self._SYS_IO_GETEVENTS_X86_64,
            ctx,
            ctypes.c_long(min_events),
            ctypes.c_long(max_events),
            events,
            ctypes.c_void_p(),
        )

    @property
    def ctx(self) -> int:
        return int(self._ctx.value)

    def submit(self, iocb: _IOCB) -> None:
        ptr = ctypes.pointer(iocb)
        arr = (ctypes.POINTER(_IOCB) * 1)(ptr)
        ret = self._io_submit(self._ctx, 1, arr)
        if ret == 1:
            return
        if ret < 0:
            raise OSError(-ret, os.strerror(-ret))
        raise RuntimeError(f"io_submit submitted {ret} iocbs, expected 1")

    def get_events(self, max_events: int) -> List[_IOEvent]:
        events = (_IOEvent * max_events)()
        ret = self._io_getevents(self._ctx, 1, max_events, events)
        if ret < 0:
            raise OSError(-ret, os.strerror(-ret))
        return [events[i] for i in range(ret)]

    def close(self) -> None:
        if self._ctx.value == 0:
            return
        ret = self._io_destroy(self._ctx)
        self._ctx = ctypes.c_ulong(0)
        if ret < 0:
            raise OSError(-ret, os.strerror(-ret))


def _aligned_empty(size: int, np_module, alignment: int = 4096):
    raw = np_module.empty(size + alignment, dtype=np_module.uint8)
    base = int(raw.ctypes.data)
    offset = (-base) % alignment
    return raw[offset : offset + size], raw


def _preadv_chunk(
    fd: int,
    buf: memoryview,
    file_offset: int,
    size: int,
    os_module,
    stats: Optional[Any] = None,
) -> None:
    """Read exactly *size* bytes from *fd* at *file_offset* into *buf*."""
    t0 = time.monotonic()
    done = 0
    while done < size:
        n = os_module.preadv(fd, [buf[done:size]], file_offset + done)
        if n == 0:
            raise RuntimeError(
                f"Unexpected EOF in preadv at offset {file_offset + done}"
            )
        done += n
    if stats is not None:
        stats.record_read(time.monotonic() - t0, size)


def read_shard_streaming_to_queue(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, "torch.Tensor"]]],
    *,
    pin_memory: bool,
    cancel_event: Optional[threading.Event] = None,
    os_module=os,
    np_module=None,
    torch_module=None,
    logger=None,
    stats: Optional[Any] = None,
) -> int:
    """Read a shard via parallel O_DIRECT preadv calls, streaming entries
    to *work_q* as they become readable.

    Multiple chunks are read concurrently from different file offsets to
    achieve high effective I/O depth on network filesystems (e.g. Vast NFS)
    where single-threaded synchronous reads severely under-utilize bandwidth.
    """
    if not sorted_entries:
        return 0
    if np_module is None or torch_module is None:
        raise RuntimeError("numpy and torch modules are required")

    span_start = sorted_entries[0].tensor_offset
    span_end = max(e.tensor_offset + e.aligned_size for e in sorted_entries)
    total_size = span_end - span_start

    # Allocate a buffer for the whole shard.  We intentionally avoid
    # torch.empty(pin_memory=True) because cudaHostAlloc is extremely
    # slow (~1-3 s per GiB) and dominates wall time for large shards.
    # A plain numpy buffer still gives good H2D throughput (the copy is
    # synchronous but PCIe bandwidth ≫ disk bandwidth).
    shard_t = None
    shard_arr = np_module.empty(total_size, dtype=np_module.uint8)

    odirect_flag = getattr(os_module, "O_DIRECT", None)
    preadv_fn = getattr(os_module, "preadv", None)
    if odirect_flag is not None and preadv_fn is not None:
        fd: Optional[int] = None
        io_pool: Optional[ThreadPoolExecutor] = None
        try:
            fd = os_module.open(abs_path, os_module.O_RDONLY | odirect_flag)
            if stats is not None:
                stats.record_mode("odirect-preadv")
            mv = memoryview(shard_arr)

            # Build aligned chunk list covering the full shard.
            chunk_size = _CHUNK_SIZE
            chunks: List[Tuple[int, int]] = []  # (offset, size)
            off = 0
            while off < total_size:
                sz = min(chunk_size, total_size - off)
                chunks.append((off, sz))
                off += sz

            # chunks_done[i] is set when chunk i finishes (success or error).
            chunks_done = [threading.Event() for _ in chunks]
            chunk_errors: List[BaseException] = []

            def _read_chunk(idx: int) -> None:
                try:
                    c_off, c_sz = chunks[idx]
                    _preadv_chunk(
                        fd,
                        mv[c_off : c_off + c_sz],
                        span_start + c_off,
                        c_sz,
                        os_module,
                        stats,
                    )
                except BaseException as exc:
                    chunk_errors.append(exc)
                finally:
                    chunks_done[idx].set()

            # Submit chunk reads with bounded concurrency.
            io_pool = ThreadPoolExecutor(max_workers=min(_IO_DEPTH, len(chunks)))
            for i in range(len(chunks)):
                io_pool.submit(_read_chunk, i)

            # Stream entries to the work queue as their data arrives.
            def _chunk_for_byte(byte_off: int) -> int:
                return byte_off // chunk_size

            for entry_idx in range(len(sorted_entries)):
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError(f"shard read cancelled: {abs_path}")
                entry = sorted_entries[entry_idx]
                local_offset = entry.tensor_offset - span_start
                start_chunk = _chunk_for_byte(local_offset)
                end_chunk = _chunk_for_byte(local_offset + entry.aligned_size - 1)
                for ci in range(start_chunk, end_chunk + 1):
                    chunks_done[ci].wait()
                if chunk_errors:
                    raise chunk_errors[0]

                if shard_t is not None:
                    tensor = shard_t[local_offset : local_offset + entry.aligned_size]
                else:
                    tensor = torch_module.from_numpy(
                        shard_arr[local_offset : local_offset + entry.aligned_size]
                    )
                _put_entry(work_q, entry, tensor, cancel_event, abs_path, stats)

            if chunk_errors:
                raise chunk_errors[0]
            return len(sorted_entries)
        except OSError as exc:
            fallback_errnos = {errno.EINVAL, errno.EOPNOTSUPP}
            if exc.errno not in fallback_errnos:
                raise
            if logger is not None:
                logger.debug(
                    "O_DIRECT preadv failed on %s (errno %s); "
                    "falling back to buffered read",
                    abs_path,
                    exc.errno,
                )
            if stats is not None:
                stats.record_fallback()
        finally:
            if io_pool is not None:
                io_pool.shutdown(wait=False)
                io_pool = None
            if fd is not None:
                os_module.close(fd)
                fd = None

    # Fallback: buffered full-shard read, then queue all entries.
    if stats is not None:
        stats.record_mode("buffered")
    t0 = time.monotonic()
    with open(abs_path, "rb") as handle:
        handle.seek(span_start)
        raw = handle.read(total_size)
    if stats is not None:
        stats.record_read(time.monotonic() - t0, total_size)
    arr = np_module.frombuffer(raw, dtype=np_module.uint8).copy()
    for entry in sorted_entries:
        off = entry.tensor_offset - span_start
        tensor = torch_module.from_numpy(arr[off : off + entry.aligned_size])
        _put_entry(work_q, entry, tensor, cancel_event, abs_path, stats)
    return len(sorted_entries)


def read_shard_aio_to_queue(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, "torch.Tensor"]]],
    *,
    pin_memory: bool,
    cancel_event: Optional[threading.Event] = None,
    os_module=os,
    np_module=None,
    torch_module=None,
    logger=None,
    stats: Optional[Any] = None,
) -> int:
    """Read a shard with Linux native AIO and stream entries to *work_q*.

    This backend keeps the same CPU-staged GMS copy path as the default
    backend but uses libaio `io_submit`/`io_getevents` instead of a thread pool
    of blocking `preadv` calls for file reads.
    """
    if not sorted_entries:
        return 0
    if np_module is None or torch_module is None:
        raise RuntimeError("numpy and torch modules are required")

    span_start = sorted_entries[0].tensor_offset
    span_end = max(e.tensor_offset + e.aligned_size for e in sorted_entries)
    total_size = span_end - span_start

    shard_arr, _raw_arr = _aligned_empty(total_size, np_module)
    base_ptr = int(shard_arr.ctypes.data)
    chunk_size = _CHUNK_SIZE
    jobs: List[_AIOJob] = []
    off = 0
    while off < total_size:
        size = min(chunk_size, total_size - off)
        jobs.append(_AIOJob(off, size))
        off += size

    odirect_flag = getattr(os_module, "O_DIRECT", None)
    if odirect_flag is None:
        raise RuntimeError("O_DIRECT is required for the aio transfer backend")

    fd: Optional[int] = None
    aio: Optional[_NativeAIO] = None
    mv = memoryview(shard_arr)
    try:
        fd = os_module.open(abs_path, os_module.O_RDONLY | odirect_flag)
        depth = min(_IO_DEPTH, len(jobs))
        aio = _NativeAIO(depth)
        if stats is not None:
            stats.record_mode("odirect-libaio")

        next_submit = 0
        inflight = 0
        completed = 0
        next_entry = 0
        chunk_done = [False] * len(jobs)

        def _submit(idx: int) -> None:
            job = jobs[idx]
            iocb = _IOCB()
            iocb.aio_data = idx
            iocb.aio_lio_opcode = _NativeAIO.IOCB_CMD_PREAD
            iocb.aio_fildes = fd
            iocb.aio_buf = base_ptr + job.offset + job.done
            iocb.aio_nbytes = job.size - job.done
            iocb.aio_offset = span_start + job.offset + job.done
            job.iocb = iocb
            job.submitted_at = time.monotonic()
            aio.submit(iocb)

        def _submit_available() -> None:
            nonlocal next_submit, inflight
            while inflight < depth and next_submit < len(jobs):
                _submit(next_submit)
                next_submit += 1
                inflight += 1

        def _chunk_for_byte(byte_off: int) -> int:
            return byte_off // chunk_size

        def _queue_ready_entries() -> None:
            nonlocal next_entry
            while next_entry < len(sorted_entries):
                entry = sorted_entries[next_entry]
                local_offset = entry.tensor_offset - span_start
                start_chunk = _chunk_for_byte(local_offset)
                end_chunk = _chunk_for_byte(local_offset + entry.aligned_size - 1)
                if not all(chunk_done[i] for i in range(start_chunk, end_chunk + 1)):
                    return
                tensor = torch_module.from_numpy(
                    shard_arr[local_offset : local_offset + entry.aligned_size]
                )
                _put_entry(work_q, entry, tensor, cancel_event, abs_path, stats)
                next_entry += 1

        _submit_available()
        while completed < len(jobs):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError(f"shard read cancelled: {abs_path}")
            assert aio is not None
            events = aio.get_events(max(1, min(depth, inflight)))
            for event in events:
                inflight -= 1
                idx = int(event.data)
                job = jobs[idx]
                elapsed = time.monotonic() - job.submitted_at
                if event.res < 0:
                    raise OSError(-int(event.res), os.strerror(-int(event.res)))
                if event.res == 0:
                    raise RuntimeError(
                        f"Unexpected EOF in aio read at offset "
                        f"{span_start + job.offset + job.done}"
                    )
                got = int(event.res)
                job.done += got
                if stats is not None:
                    stats.record_read(elapsed, got)
                if job.done < job.size:
                    _submit(idx)
                    inflight += 1
                else:
                    chunk_done[idx] = True
                    completed += 1
            _submit_available()
            _queue_ready_entries()

        _queue_ready_entries()
        if next_entry != len(sorted_entries):
            raise RuntimeError(
                f"BUG: queued {next_entry} of {len(sorted_entries)} entries "
                f"after aio read of {abs_path}"
            )
        return len(sorted_entries)
    finally:
        mv.release()
        if aio is not None:
            aio.close()
        if fd is not None:
            os_module.close(fd)


def read_shard_to_queue(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, torch.Tensor]]],
    *,
    pin_memory: bool,
    read_shard,
    cancel_event: Optional[threading.Event] = None,
) -> int:
    shard_result = read_shard(
        abs_path,
        sorted_entries,
        -1,
        pin_memory=pin_memory,
    )
    for entry in sorted_entries:
        _put_entry(
            work_q, entry, shard_result[entry.allocation_id], cancel_event, abs_path
        )
    return len(sorted_entries)


def load_manifest_and_metadata(
    input_dir: str,
) -> Tuple[SaveManifest, Dict[str, Dict[str, Any]]]:
    manifest_path = os.path.join(input_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = SaveManifest.from_dict(json.load(handle))

    metadata_path = os.path.join(input_dir, "gms_metadata.json")
    raw_meta: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            raw_meta = json.load(handle)

    return manifest, decode_metadata(raw_meta)
