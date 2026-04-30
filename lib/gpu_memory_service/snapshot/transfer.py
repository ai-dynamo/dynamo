# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer backends for GMS snapshot restore."""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from gpu_memory_service.snapshot.disk import (
    read_shard_aio_to_queue,
    read_shard_streaming_to_queue,
)
from gpu_memory_service.snapshot.model import AllocationEntry
from gpu_memory_service.snapshot.restore import WORK_QUEUE_DEPTH_MULTIPLIER

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

DEFAULT_TRANSFER_BACKEND = "default"
AIO_TRANSFER_BACKEND = "aio"
NIXL_GDS_TRANSFER_BACKEND = "nixl-gds"
CUFILE_GDS_TRANSFER_BACKEND = "cufile-gds"
TRANSFER_BACKEND_CHOICES = (
    DEFAULT_TRANSFER_BACKEND,
    AIO_TRANSFER_BACKEND,
    NIXL_GDS_TRANSFER_BACKEND,
    CUFILE_GDS_TRANSFER_BACKEND,
)

try:
    from nixl._api import nixl_agent, nixl_agent_config

    _NIXL_AVAILABLE = True
except ImportError:
    _NIXL_AVAILABLE = False
    nixl_agent = None  # type: ignore[assignment]
    nixl_agent_config = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FileTransferSource:
    """One source extent in a snapshot file."""

    allocation_id: str
    file_path: str
    file_offset: int
    byte_count: int


@dataclass(frozen=True)
class GMSTransferTarget:
    """One destination extent in GMS-owned GPU virtual memory."""

    allocation_id: str
    va: int
    device: int
    byte_count: int


class TransferSession(Protocol):
    """Live restore operation for a set of transfer sources."""

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        """Move all source bytes into matching GMS targets."""

    def close(self) -> None:
        """Release resources and cancel any pending work."""


class TransferBackend(Protocol):
    """Backend capable of restoring bytes into GMS targets."""

    name: str

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        """Start or stage restore work for the given sources."""

    def close(self) -> None:
        """Release backend-global resources."""


def build_file_transfer_sources(
    input_dir: str,
    allocations: Sequence[AllocationEntry],
) -> List[FileTransferSource]:
    """Convert manifest allocation placement into backend-neutral extents."""
    return [
        FileTransferSource(
            allocation_id=entry.allocation_id,
            file_path=os.path.join(input_dir, entry.tensor_file),
            file_offset=int(entry.tensor_offset),
            byte_count=int(entry.aligned_size),
        )
        for entry in allocations
    ]


def create_transfer_backend(
    name: str,
    *,
    device: int,
    max_workers: int,
    torch_module: Any = None,
    tensor_from_pointer: Any = None,
) -> TransferBackend:
    if name == DEFAULT_TRANSFER_BACKEND:
        if torch_module is None or tensor_from_pointer is None:
            raise RuntimeError(
                "default GMS transfer backend requires PyTorch tensor imports"
            )
        return DefaultTransferBackend(
            device=device,
            max_workers=max_workers,
            torch_module=torch_module,
            tensor_from_pointer=tensor_from_pointer,
        )
    if name == AIO_TRANSFER_BACKEND:
        if torch_module is None or tensor_from_pointer is None:
            raise RuntimeError(
                "aio GMS transfer backend requires PyTorch tensor imports"
            )
        return AioTransferBackend(
            device=device,
            max_workers=max_workers,
            torch_module=torch_module,
            tensor_from_pointer=tensor_from_pointer,
        )
    if name == NIXL_GDS_TRANSFER_BACKEND:
        return NixlGDSTransferBackend(device=device)
    if name == CUFILE_GDS_TRANSFER_BACKEND:
        return CufileGDSTransferBackend(
            device=device,
            max_workers=max_workers,
        )
    raise ValueError(
        f"Unsupported GMS transfer backend {name!r}; "
        f"expected one of {', '.join(TRANSFER_BACKEND_CHOICES)}"
    )


def _get_numpy_module() -> Any:
    try:
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for the default GMS transfer backend"
        ) from exc
    return np_module


def _group_sources_by_path(
    sources: Sequence[FileTransferSource],
) -> Dict[str, List[FileTransferSource]]:
    groups: Dict[str, List[FileTransferSource]] = defaultdict(list)
    for source in sources:
        groups[source.file_path].append(source)
    for grouped in groups.values():
        grouped.sort(key=lambda source: source.file_offset)
    return dict(groups)


def _source_as_allocation_entry(source: FileTransferSource) -> AllocationEntry:
    return AllocationEntry(
        allocation_id=source.allocation_id,
        size=source.byte_count,
        aligned_size=source.byte_count,
        tag="",
        tensor_file=source.file_path,
        tensor_offset=source.file_offset,
    )


@dataclass
class _TransferStats:
    backend_name: str
    total_bytes: int
    worker_count: int
    shard_count: int
    queued_entries: int = 0
    queued_bytes: int = 0
    copied_entries: int = 0
    copied_bytes: int = 0
    transfer_wall_s: float = 0.0
    disk_task_wall_s: float = 0.0
    read_s: float = 0.0
    queue_put_s: float = 0.0
    target_wait_s: float = 0.0
    copy_s: float = 0.0
    finalize_s: float = 0.0
    fallbacks: int = 0
    modes: Dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_mode(self, mode: str) -> None:
        with self.lock:
            self.modes[mode] = self.modes.get(mode, 0) + 1

    def record_fallback(self) -> None:
        with self.lock:
            self.fallbacks += 1

    def record_read(self, seconds: float, byte_count: int) -> None:
        with self.lock:
            self.read_s += seconds

    def record_queue_put(self, seconds: float) -> None:
        with self.lock:
            self.queue_put_s += seconds

    def record_queued_entry(self, byte_count: int) -> None:
        with self.lock:
            self.queued_entries += 1
            self.queued_bytes += byte_count

    def record_disk_task(self, seconds: float) -> None:
        with self.lock:
            self.disk_task_wall_s += seconds

    def record_transfer_wall(self, seconds: float) -> None:
        with self.lock:
            self.transfer_wall_s += seconds

    def record_target_wait(self, seconds: float) -> None:
        with self.lock:
            self.target_wait_s += seconds

    def record_copy(self, seconds: float, byte_count: int) -> None:
        with self.lock:
            self.copy_s += seconds
            self.copied_entries += 1
            self.copied_bytes += byte_count

    def record_finalize(self, seconds: float) -> None:
        with self.lock:
            self.finalize_s += seconds

    def log(self) -> None:
        with self.lock:
            transfer_wall_s = self.transfer_wall_s
            disk_task_wall_s = self.disk_task_wall_s
            read_s = self.read_s
            queue_put_s = self.queue_put_s
            target_wait_s = self.target_wait_s
            copy_s = self.copy_s
            finalize_s = self.finalize_s
            queued_entries = self.queued_entries
            queued_bytes = self.queued_bytes
            copied_entries = self.copied_entries
            copied_bytes = self.copied_bytes
            fallbacks = self.fallbacks
            modes = dict(self.modes)
        gib = self.total_bytes / (1024**3)
        transfer_bw = gib / transfer_wall_s if transfer_wall_s > 0 else 0.0
        disk_task_bw = gib / disk_task_wall_s if disk_task_wall_s > 0 else 0.0
        copy_bw = copied_bytes / (1024**3) / copy_s if copy_s > 0 else 0.0
        logger.info(
            "Transfer metrics: backend=%s total=%.2f GiB workers=%d shards=%d "
            "transfer_wall=%.3fs transfer_bw=%.2f GiB/s "
            "disk_task_sum=%.3fs disk_task_sum_bw=%.2f GiB/s "
            "read_syscall_sum=%.3fs "
            "queue_put=%.3fs target_wait=%.3fs copy=%.3fs copy_bw=%.2f GiB/s "
            "finalize=%.3fs queued=%d/%d copied=%d/%d fallbacks=%d modes=%s",
            self.backend_name,
            gib,
            self.worker_count,
            self.shard_count,
            transfer_wall_s,
            transfer_bw,
            disk_task_wall_s,
            disk_task_bw,
            read_s,
            queue_put_s,
            target_wait_s,
            copy_s,
            copy_bw,
            finalize_s,
            queued_entries,
            queued_bytes,
            copied_entries,
            copied_bytes,
            fallbacks,
            ",".join(f"{key}:{value}" for key, value in sorted(modes.items())) or "-",
        )


@dataclass
class _DefaultRestoreContext:
    worker_count: int
    use_streams: bool
    device: int
    work_q: queue.Queue[Optional[Tuple[AllocationEntry, "torch.Tensor"]]]
    streams: List[torch.cuda.Stream]
    stats: _TransferStats
    cancel_event: threading.Event = field(default_factory=threading.Event)
    targets_ready: threading.Event = field(default_factory=threading.Event)
    targets: Dict[str, GMSTransferTarget] = field(default_factory=dict)
    staged_srcs: List[torch.Tensor] = field(default_factory=list)
    copy_errors: List[BaseException] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def build(
        cls,
        worker_count: int,
        *,
        device: int,
        use_streams: bool,
        torch_module: Any,
        stats: _TransferStats,
    ) -> "_DefaultRestoreContext":
        streams = (
            [torch_module.cuda.Stream(device=device) for _ in range(worker_count)]
            if use_streams
            else []
        )
        return cls(
            worker_count=worker_count,
            use_streams=use_streams,
            device=device,
            work_q=queue.Queue(maxsize=worker_count * WORK_QUEUE_DEPTH_MULTIPLIER),
            streams=streams,
            stats=stats,
        )


class DefaultTransferBackend:
    """Existing CPU-staged disk read plus PyTorch copy restore path."""

    name = DEFAULT_TRANSFER_BACKEND

    def __init__(
        self,
        *,
        device: int,
        max_workers: int,
        torch_module: Any,
        tensor_from_pointer: Any,
    ) -> None:
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._torch = torch_module
        self._tensor_from_pointer = tensor_from_pointer

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        groups = _group_sources_by_path(sources)
        worker_count = max(1, min(self._max_workers, len(groups) or 1))
        use_streams = bool(self._torch.cuda.is_available())
        stats = _TransferStats(
            backend_name=self.name,
            total_bytes=sum(source.byte_count for source in sources),
            worker_count=worker_count,
            shard_count=len(groups),
        )
        ctx = _DefaultRestoreContext.build(
            worker_count,
            device=self._device,
            use_streams=use_streams,
            torch_module=self._torch,
            stats=stats,
        )
        copy_threads = self._start_copy_threads(ctx)
        disk_pool = ThreadPoolExecutor(max_workers=worker_count)
        disk_futures = {
            disk_pool.submit(
                self._read_sources_to_queue,
                file_path,
                grouped_sources,
                ctx,
            ): file_path
            for file_path, grouped_sources in groups.items()
        }
        return _DefaultTransferSession(
            ctx=ctx,
            disk_pool=disk_pool,
            disk_futures=disk_futures,
            copy_threads=copy_threads,
            source_lengths={
                source.allocation_id: source.byte_count for source in sources
            },
            torch_module=self._torch,
            stats=stats,
        )

    def close(self) -> None:
        pass

    def _read_sources_to_queue(
        self,
        file_path: str,
        sources: List[FileTransferSource],
        ctx: _DefaultRestoreContext,
    ) -> int:
        entries = [_source_as_allocation_entry(source) for source in sources]
        t0 = time.monotonic()
        try:
            return self._read_entries_to_queue(file_path, entries, ctx)
        finally:
            ctx.stats.record_disk_task(time.monotonic() - t0)

    def _read_entries_to_queue(
        self,
        file_path: str,
        entries: List[AllocationEntry],
        ctx: _DefaultRestoreContext,
    ) -> int:
        return read_shard_streaming_to_queue(
            file_path,
            entries,
            ctx.work_q,
            pin_memory=ctx.use_streams,
            cancel_event=ctx.cancel_event,
            os_module=os,
            np_module=_get_numpy_module(),
            torch_module=self._torch,
            logger=logger,
            stats=ctx.stats,
        )

    def _start_copy_threads(
        self,
        ctx: _DefaultRestoreContext,
    ) -> List[threading.Thread]:
        threads = [
            threading.Thread(
                target=self._run_copy_worker,
                args=(ctx, index),
                daemon=True,
            )
            for index in range(ctx.worker_count)
        ]
        for thread in threads:
            thread.start()
        return threads

    def _run_copy_worker(
        self,
        ctx: _DefaultRestoreContext,
        stream_idx: int,
    ) -> None:
        while True:
            try:
                item = ctx.work_q.get(timeout=0.1)
            except queue.Empty:
                if ctx.cancel_event.is_set():
                    return
                continue
            if item is None:
                return

            entry, src = item
            try:
                wait_t0 = time.monotonic()
                while not ctx.targets_ready.wait(timeout=0.1):
                    if ctx.cancel_event.is_set():
                        return
                ctx.stats.record_target_wait(time.monotonic() - wait_t0)
                target = ctx.targets[entry.allocation_id]
                dst = self._tensor_from_pointer(
                    target.va,
                    [target.byte_count],
                    [1],
                    self._torch.uint8,
                    target.device,
                )
                copy_t0 = time.monotonic()
                if ctx.streams:
                    with self._torch.cuda.stream(ctx.streams[stream_idx]):
                        dst.copy_(src, non_blocking=src.is_pinned())
                else:
                    dst.copy_(src)
                ctx.stats.record_copy(time.monotonic() - copy_t0, target.byte_count)
                if ctx.use_streams and src.is_pinned():
                    with ctx.lock:
                        ctx.staged_srcs.append(src)
            except Exception as exc:  # noqa: BLE001
                with ctx.lock:
                    ctx.copy_errors.append(exc)


class AioTransferBackend(DefaultTransferBackend):
    """CPU-staged restore path with Linux native AIO shard reads."""

    name = AIO_TRANSFER_BACKEND

    def _read_entries_to_queue(
        self,
        file_path: str,
        entries: List[AllocationEntry],
        ctx: _DefaultRestoreContext,
    ) -> int:
        return read_shard_aio_to_queue(
            file_path,
            entries,
            ctx.work_q,
            pin_memory=ctx.use_streams,
            cancel_event=ctx.cancel_event,
            os_module=os,
            np_module=_get_numpy_module(),
            torch_module=self._torch,
            logger=logger,
            stats=ctx.stats,
        )


class _DefaultTransferSession:
    def __init__(
        self,
        *,
        ctx: _DefaultRestoreContext,
        disk_pool: ThreadPoolExecutor,
        disk_futures: Dict[Future[int], str],
        copy_threads: List[threading.Thread],
        source_lengths: Dict[str, int],
        torch_module: Any,
        stats: _TransferStats,
    ) -> None:
        self._ctx = ctx
        self._disk_pool = disk_pool
        self._disk_futures = disk_futures
        self._copy_threads = copy_threads
        self._source_lengths = source_lengths
        self._torch = torch_module
        self._stats = stats
        self._active = True

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        self._validate_targets(targets)
        transfer_t0 = time.monotonic()
        with self._ctx.lock:
            self._ctx.targets = dict(targets)
            self._ctx.targets_ready.set()
        disk_error: Optional[BaseException] = None
        finalize_error: Optional[BaseException] = None
        drain_queue = False
        try:
            self._await_disk_reads()
        except Exception as exc:
            disk_error = exc
            self._cancel()
            drain_queue = True
            self._disk_pool.shutdown(wait=True, cancel_futures=True)
        else:
            self._disk_pool.shutdown(wait=True)
        try:
            self._stop_copy_threads(drain_queue=drain_queue)
        finally:
            self._active = False
            try:
                self._finalize()
            except Exception as exc:  # noqa: BLE001
                finalize_error = exc
            self._stats.record_transfer_wall(time.monotonic() - transfer_t0)
        if disk_error is not None:
            raise disk_error
        if finalize_error is not None:
            raise finalize_error
        self._stats.log()

    def close(self) -> None:
        if not self._active:
            return
        self._cancel()
        self._disk_pool.shutdown(wait=True, cancel_futures=True)
        self._stop_copy_threads(drain_queue=True)
        self._active = False
        try:
            self._finalize()
        except Exception:  # noqa: BLE001
            logger.warning(
                "cleanup failed during restore transfer error handling",
                exc_info=True,
            )

    def _validate_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        missing = [
            allocation_id
            for allocation_id in self._source_lengths
            if allocation_id not in targets
        ]
        if missing:
            raise RuntimeError(
                f"Missing GMS transfer target for allocation {missing[0]}"
            )
        for allocation_id, source_length in self._source_lengths.items():
            target_length = targets[allocation_id].byte_count
            if source_length != target_length:
                raise RuntimeError(
                    f"GMS target size mismatch for allocation {allocation_id}: "
                    f"source={source_length} target={target_length}"
                )

    def _await_disk_reads(self) -> None:
        for future in as_completed(self._disk_futures):
            file_path = self._disk_futures[future]
            try:
                future.result()
            except CancelledError:
                pass
            except Exception as exc:
                raise RuntimeError(f"Failed to load shard {file_path}: {exc}") from exc

    def _stop_copy_threads(self, *, drain_queue: bool = False) -> None:
        if drain_queue:
            self._drain_queue()
        for _ in self._copy_threads:
            if drain_queue:
                while True:
                    try:
                        self._ctx.work_q.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        self._drain_queue()
            else:
                self._ctx.work_q.put(None)
        for thread in self._copy_threads:
            thread.join()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._ctx.work_q.get_nowait()
            except queue.Empty:
                return

    def _cancel(self) -> None:
        self._ctx.cancel_event.set()
        self._ctx.targets_ready.set()
        self._drain_queue()

    def _finalize(self) -> None:
        t0 = time.monotonic()
        try:
            if self._ctx.use_streams:
                self._torch.cuda.synchronize(device=self._ctx.device)
                self._ctx.staged_srcs.clear()
        finally:
            self._stats.record_finalize(time.monotonic() - t0)
        if self._ctx.copy_errors:
            raise RuntimeError(
                f"Failed to copy restored data to GMS: {self._ctx.copy_errors[0]}"
            )


def _load_cufile_bindings() -> Any:
    try:
        from cufile import bindings
    except ImportError as exc:
        raise RuntimeError(
            "cufile Python bindings are required for the cufile-gds transfer backend"
        ) from exc
    bindings.libcufile.cuFileRead.restype = ctypes.c_ssize_t
    return bindings


def _set_current_cuda_device(device: int) -> None:
    try:
        from cuda.bindings import runtime as cuda_runtime
    except ImportError as exc:
        raise RuntimeError(
            "cuda-python is required for the cufile-gds transfer backend"
        ) from exc
    result = cuda_runtime.cudaSetDevice(device)
    if isinstance(result, tuple):
        result = result[0]
    if result != cuda_runtime.cudaError_t.cudaSuccess:
        _, message = cuda_runtime.cudaGetErrorString(result)
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        raise RuntimeError(f"cudaSetDevice({device}) failed: {message}")


class CufileGDSTransferBackend:
    """Direct cuFile GDS backend for file-to-GMS GPU memory transfers."""

    name = CUFILE_GDS_TRANSFER_BACKEND

    def __init__(self, *, device: int, max_workers: int) -> None:
        self._device = device
        self._max_workers = max(1, int(max_workers))
        _set_current_cuda_device(device)
        self._bindings = _load_cufile_bindings()
        self._bindings.cuFileDriverOpen()
        logger.info(
            "cuFile GDS backend initialized for device %d with %d workers",
            device,
            self._max_workers,
        )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _CufileGDSTransferSession(
            bindings=self._bindings,
            device=self._device,
            max_workers=self._max_workers,
            sources=sources,
        )

    def close(self) -> None:
        if self._bindings is None:
            return
        try:
            self._bindings.cuFileDriverClose()
        finally:
            self._bindings = None


class _CufileGDSTransferSession:
    def __init__(
        self,
        *,
        bindings: Any,
        device: int,
        max_workers: int,
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._bindings = bindings
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._sources = list(sources)
        self._active = True

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        self._validate_targets(targets)
        if not self._sources:
            self._active = False
            return

        total_bytes = sum(source.byte_count for source in self._sources)
        worker_count = min(self._max_workers, len(self._sources))
        t0 = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        self._restore_source,
                        source,
                        targets[source.allocation_id],
                    ): source
                    for source in self._sources
                }
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"cuFile GDS transfer failed for {source.file_path} "
                            f"at offset {source.file_offset}: {exc}"
                        ) from exc
        finally:
            self._active = False

        elapsed = time.monotonic() - t0
        throughput = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0
        logger.info(
            "cuFile GDS transfers complete: %.2f GiB in %.3fs (%.2f GiB/s, workers=%d)",
            total_bytes / (1024**3),
            elapsed,
            throughput,
            worker_count,
        )

    def close(self) -> None:
        self._active = False

    def _validate_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        for source in self._sources:
            target = targets.get(source.allocation_id)
            if target is None:
                raise RuntimeError(
                    f"Missing GMS transfer target for allocation {source.allocation_id}"
                )
            if target.byte_count != source.byte_count:
                raise RuntimeError(
                    f"GMS target size mismatch for allocation {source.allocation_id}: "
                    f"source={source.byte_count} target={target.byte_count}"
                )
            if target.device != self._device:
                raise RuntimeError(
                    f"GMS target device mismatch for allocation {source.allocation_id}: "
                    f"backend={self._device} target={target.device}"
                )

    def _restore_source(
        self,
        source: FileTransferSource,
        target: GMSTransferTarget,
    ) -> None:
        fd: Optional[int] = None
        file_handle = None
        target_ptr = ctypes.c_void_p(target.va)
        buffer_registered = False
        try:
            fd = os.open(source.file_path, os.O_RDONLY)
            file_handle = self._bindings.cuFileHandleRegister(fd)
            self._bindings.cuFileBufRegister(target_ptr, target.byte_count, 0)
            buffer_registered = True
            self._read_exact(
                file_handle,
                target_ptr,
                source.byte_count,
                source.file_offset,
            )
        finally:
            if buffer_registered:
                try:
                    self._bindings.cuFileBufDeregister(target_ptr)
                except Exception:
                    logger.warning(
                        "cuFileBufDeregister failed during restore cleanup",
                        exc_info=True,
                    )
            if file_handle is not None:
                try:
                    self._bindings.cuFileHandleDeregister(file_handle)
                except Exception:
                    logger.warning(
                        "cuFileHandleDeregister failed during restore cleanup",
                        exc_info=True,
                    )
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _read_exact(
        self,
        file_handle: Any,
        target_ptr: ctypes.c_void_p,
        byte_count: int,
        file_offset: int,
    ) -> None:
        done = 0
        while done < byte_count:
            read = self._bindings.cuFileRead(
                file_handle,
                target_ptr,
                byte_count - done,
                file_offset + done,
                done,
            )
            if read < 0:
                raise RuntimeError(f"cuFileRead returned error {read}")
            if read == 0:
                raise RuntimeError(
                    f"short cuFileRead: got {done} of {byte_count} bytes"
                )
            done += int(read)


class NixlGDSTransferBackend:
    """NIXL GDS backend for direct file-to-GMS GPU memory transfers."""

    name = NIXL_GDS_TRANSFER_BACKEND

    def __init__(self, *, device: int) -> None:
        if not _NIXL_AVAILABLE:
            raise RuntimeError(
                "NIXL Python bindings are required for the nixl-gds transfer backend"
            )
        self._device = device
        self._agent_name = f"gms_gds_{device}_{os.getpid()}"
        self._agent = nixl_agent(
            self._agent_name,
            nixl_agent_config(backends=[]),
        )
        self._agent.create_backend("GDS_MT")
        logger.info("NIXL GDS_MT backend initialized for device %d", device)

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _NixlGDSTransferSession(
            agent=self._agent,
            agent_name=self._agent_name,
            device=self._device,
            sources=sources,
        )

    def close(self) -> None:
        self._agent = None


class _NixlGDSTransferSession:
    def __init__(
        self,
        *,
        agent: Any,
        agent_name: str,
        device: int,
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._agent = agent
        self._agent_name = agent_name
        self._device = device
        self._sources = list(sources)
        self._active = True

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        self._validate_targets(targets)
        pending: List[Tuple[Any, Any, Any, Optional[int], str]] = []
        total_bytes = sum(source.byte_count for source in self._sources)
        t0 = time.monotonic()
        try:
            for file_path, sources in _group_sources_by_path(self._sources).items():
                fd: Optional[int] = None
                file_reg = None
                vram_reg = None
                handle = None
                try:
                    fd = os.open(file_path, os.O_RDONLY)
                    file_descs = [
                        (source.file_offset, source.byte_count, fd, "")
                        for source in sources
                    ]
                    file_reg = self._agent.register_memory(file_descs, "FILE")

                    vram_descs = [
                        (
                            targets[source.allocation_id].va,
                            targets[source.allocation_id].byte_count,
                            targets[source.allocation_id].device,
                            "",
                        )
                        for source in sources
                    ]
                    vram_reg = self._agent.register_memory(vram_descs, "VRAM")

                    handle = self._agent.initialize_xfer(
                        "READ",
                        vram_reg.trim(),
                        file_reg.trim(),
                        self._agent_name,
                    )
                    state = self._agent.transfer(handle)
                    if state == "ERR":
                        raise RuntimeError(
                            f"NIXL GDS transfer failed to start: {file_path}"
                        )
                    if state not in {"PROC", "DONE"}:
                        raise RuntimeError(
                            f"NIXL GDS transfer returned unexpected state {state!r}"
                        )
                    pending.append((handle, file_reg, vram_reg, fd, file_path))
                except Exception:
                    self._release_transfer_resources(handle, file_reg, vram_reg, fd)
                    raise

            for handle, file_reg, vram_reg, fd, file_path in pending:
                state = self._agent.check_xfer_state(handle)
                while state == "PROC":
                    time.sleep(0.001)
                    state = self._agent.check_xfer_state(handle)
                if state == "ERR":
                    raise RuntimeError(f"NIXL GDS transfer failed: {file_path}")
                if state != "DONE":
                    raise RuntimeError(
                        f"NIXL GDS transfer ended in unexpected state {state!r}: {file_path}"
                    )
        finally:
            for handle, file_reg, vram_reg, fd, _ in pending:
                self._release_transfer_resources(handle, file_reg, vram_reg, fd)
            self._active = False

        elapsed = time.monotonic() - t0
        throughput = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0
        logger.info(
            "NIXL GDS transfers complete: %.2f GiB in %.3fs (%.2f GiB/s)",
            total_bytes / (1024**3),
            elapsed,
            throughput,
        )

    def close(self) -> None:
        self._active = False

    def _validate_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        for source in self._sources:
            target = targets.get(source.allocation_id)
            if target is None:
                raise RuntimeError(
                    f"Missing GMS transfer target for allocation {source.allocation_id}"
                )
            if target.byte_count != source.byte_count:
                raise RuntimeError(
                    f"GMS target size mismatch for allocation {source.allocation_id}: "
                    f"source={source.byte_count} target={target.byte_count}"
                )

    def _release_transfer_resources(
        self,
        handle: Any,
        file_reg: Any,
        vram_reg: Any,
        fd: Optional[int],
    ) -> None:
        if handle is not None:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:
                pass
        if vram_reg is not None:
            try:
                self._agent.deregister_memory(vram_reg)
            except Exception:
                pass
        if file_reg is not None:
            try:
                self._agent.deregister_memory(file_reg)
            except Exception:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
