# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer backends for GMS snapshot restore."""

from __future__ import annotations

import ctypes
import errno
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
LOCAL_SSD_STRIPED_TRANSFER_BACKEND = "local-ssd-striped"
LOCAL_SSD_PINNED_TRANSFER_BACKEND = "local-ssd-pinned"
NIXL_GDS_TRANSFER_BACKEND = "nixl-gds"
CUFILE_GDS_TRANSFER_BACKEND = "cufile-gds"
GMS_LOCAL_SSD_ROOTS_ENV = "GMS_LOCAL_SSD_ROOTS"
TRANSFER_BACKEND_CHOICES = (
    DEFAULT_TRANSFER_BACKEND,
    AIO_TRANSFER_BACKEND,
    LOCAL_SSD_STRIPED_TRANSFER_BACKEND,
    LOCAL_SSD_PINNED_TRANSFER_BACKEND,
    NIXL_GDS_TRANSFER_BACKEND,
    CUFILE_GDS_TRANSFER_BACKEND,
)

_PINNED_COPY_CHUNK_SIZE = 64 * 1024 * 1024
_PINNED_COPY_BUFFERS_PER_ROOT = 2
_LIBC = ctypes.CDLL(None)
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None


def _load_nixl_api() -> Tuple[Any, Any]:
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as exc:
        raise RuntimeError(
            "NIXL Python bindings are required for the nixl-gds transfer backend"
        ) from exc
    return nixl_agent, nixl_agent_config


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
    if name == LOCAL_SSD_STRIPED_TRANSFER_BACKEND:
        if torch_module is None or tensor_from_pointer is None:
            raise RuntimeError(
                "local-ssd-striped GMS transfer backend requires PyTorch tensor imports"
            )
        return LocalSSDStripedTransferBackend(
            device=device,
            max_workers=max_workers,
            torch_module=torch_module,
            tensor_from_pointer=tensor_from_pointer,
            local_roots=_parse_local_ssd_roots(
                os.environ.get(GMS_LOCAL_SSD_ROOTS_ENV, "")
            ),
        )
    if name == LOCAL_SSD_PINNED_TRANSFER_BACKEND:
        return LocalSSDPinnedTransferBackend(
            device=device,
            max_workers=max_workers,
            local_roots=_parse_local_ssd_roots(
                os.environ.get(GMS_LOCAL_SSD_ROOTS_ENV, "")
            ),
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


def _parse_local_ssd_roots(value: str) -> List[str]:
    return [os.path.abspath(part.strip()) for part in value.split(",") if part.strip()]


def _match_local_root(file_path: str, local_roots: Sequence[str]) -> Optional[str]:
    abs_path = os.path.abspath(file_path)
    for root in local_roots:
        try:
            if os.path.commonpath([root, abs_path]) == root:
                return root
        except ValueError:
            continue
    return None


def _group_sources_by_local_root(
    sources: Sequence[FileTransferSource],
    local_roots: Sequence[str],
    backend_name: str,
) -> Dict[str, List[Tuple[str, List[FileTransferSource]]]]:
    groups_by_path = _group_sources_by_path(sources)
    groups: Dict[str, List[Tuple[str, List[FileTransferSource]]]] = defaultdict(list)
    for file_path, grouped_sources in groups_by_path.items():
        root = _match_local_root(file_path, local_roots)
        if root is None:
            raise RuntimeError(
                f"{backend_name} source path {file_path!r} is not under any "
                f"{GMS_LOCAL_SSD_ROOTS_ENV} root: {list(local_roots)}"
            )
        groups[root].append((file_path, grouped_sources))
    for grouped_paths in groups.values():
        grouped_paths.sort(key=lambda item: item[0])
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


class LocalSSDStripedTransferBackend(DefaultTransferBackend):
    """CPU-staged restore path striped across host-local SSD roots.

    The checkpoint manifest must point shard entries at absolute paths under
    GMS_LOCAL_SSD_ROOTS. The backend runs one disk task per active SSD root so
    each local device is read sequentially while all roots run in parallel.
    """

    name = LOCAL_SSD_STRIPED_TRANSFER_BACKEND

    def __init__(
        self,
        *,
        device: int,
        max_workers: int,
        torch_module: Any,
        tensor_from_pointer: Any,
        local_roots: Sequence[str],
    ) -> None:
        super().__init__(
            device=device,
            max_workers=max_workers,
            torch_module=torch_module,
            tensor_from_pointer=tensor_from_pointer,
        )
        self._local_roots = [
            os.path.abspath(root) for root in local_roots if str(root).strip()
        ]
        if not self._local_roots:
            raise RuntimeError(
                f"{LOCAL_SSD_STRIPED_TRANSFER_BACKEND} requires "
                f"{GMS_LOCAL_SSD_ROOTS_ENV}=<root0>,<root1>,..."
            )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        root_groups = self._group_sources_by_local_root(sources)
        worker_count = max(1, min(self._max_workers, len(root_groups) or 1))
        if worker_count < len(root_groups):
            logger.warning(
                "%s has %d active SSD roots but only %d workers; "
                "increase GMS_LOAD_WORKERS for full stripe parallelism",
                self.name,
                len(root_groups),
                worker_count,
            )
        use_streams = bool(self._torch.cuda.is_available())
        stats = _TransferStats(
            backend_name=self.name,
            total_bytes=sum(source.byte_count for source in sources),
            worker_count=worker_count,
            shard_count=sum(len(grouped) for grouped in root_groups.values()),
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
                self._read_local_root_to_queue,
                root,
                grouped_paths,
                ctx,
            ): root
            for root, grouped_paths in root_groups.items()
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

    def _group_sources_by_local_root(
        self,
        sources: Sequence[FileTransferSource],
    ) -> Dict[str, List[Tuple[str, List[FileTransferSource]]]]:
        return _group_sources_by_local_root(
            sources,
            self._local_roots,
            self.name,
        )

    def _match_local_root(self, file_path: str) -> Optional[str]:
        return _match_local_root(file_path, self._local_roots)

    def _read_local_root_to_queue(
        self,
        root: str,
        grouped_paths: List[Tuple[str, List[FileTransferSource]]],
        ctx: _DefaultRestoreContext,
    ) -> int:
        t0 = time.monotonic()
        total_entries = 0
        try:
            for file_path, grouped_sources in grouped_paths:
                total_entries += self._read_sources_to_queue(
                    file_path,
                    grouped_sources,
                    ctx,
                )
            return total_entries
        finally:
            logger.info(
                "%s completed root=%s shards=%d entries=%d elapsed=%.3fs",
                self.name,
                root,
                len(grouped_paths),
                total_entries,
                time.monotonic() - t0,
            )


class _CudaRuntime:
    _CUDA_MEMCPY_HOST_TO_DEVICE = 1
    _CUDA_STREAM_NON_BLOCKING = 1

    def __init__(self) -> None:
        self._lib = ctypes.CDLL("libcudart.so")
        self._lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self._lib.cudaGetErrorString.restype = ctypes.c_char_p
        self._lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self._lib.cudaSetDevice.restype = ctypes.c_int
        self._lib.cudaHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self._lib.cudaHostRegister.restype = ctypes.c_int
        self._lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        self._lib.cudaHostUnregister.restype = ctypes.c_int
        self._lib.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self._lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int
        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int

    def _check(self, code: int, operation: str) -> None:
        if code == 0:
            return
        message = self._lib.cudaGetErrorString(code)
        if isinstance(message, bytes):
            text = message.decode("utf-8", errors="replace")
        else:
            text = str(message)
        raise RuntimeError(f"{operation} failed: {text} ({code})")

    def set_device(self, device: int) -> None:
        self._check(self._lib.cudaSetDevice(device), f"cudaSetDevice({device})")

    def host_register(self, ptr: int, size: int) -> None:
        self._check(
            self._lib.cudaHostRegister(
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(0),
            ),
            "cudaHostRegister",
        )

    def host_unregister(self, ptr: int) -> None:
        self._check(
            self._lib.cudaHostUnregister(ctypes.c_void_p(ptr)),
            "cudaHostUnregister",
        )

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self._check(
            self._lib.cudaStreamCreateWithFlags(
                ctypes.byref(stream),
                ctypes.c_uint(self._CUDA_STREAM_NON_BLOCKING),
            ),
            "cudaStreamCreateWithFlags",
        )
        return stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        self._check(self._lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def synchronize_stream(self, stream: ctypes.c_void_p) -> None:
        self._check(self._lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def memcpy_h2d_async(
        self,
        dst_ptr: int,
        src_ptr: int,
        size: int,
        stream: ctypes.c_void_p,
    ) -> None:
        self._check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(size),
                ctypes.c_int(self._CUDA_MEMCPY_HOST_TO_DEVICE),
                stream,
            ),
            "cudaMemcpyAsync",
        )


def _load_cuda_runtime() -> _CudaRuntime:
    return _CudaRuntime()


def _allocate_aligned_buffer(size: int) -> Tuple[memoryview, Any, int]:
    alignment = 4096
    ptr = ctypes.c_void_p()
    rc = _LIBC.posix_memalign(ctypes.byref(ptr), alignment, size)
    if rc != 0:
        raise OSError(rc, os.strerror(rc))
    array = (ctypes.c_ubyte * size).from_address(ptr.value)
    return memoryview(array), array, int(ptr.value)


def _free_aligned_buffer(view: memoryview, ptr: int) -> None:
    view.release()
    _LIBC.free(ctypes.c_void_p(ptr))


def _open_read_fd(path: str) -> int:
    odirect = getattr(os, "O_DIRECT", 0)
    flags = os.O_RDONLY | odirect
    try:
        return os.open(path, flags)
    except OSError as exc:
        if odirect and exc.errno in {errno.EINVAL, errno.EOPNOTSUPP}:
            logger.warning(
                "O_DIRECT unavailable for %s; falling back to buffered reads",
                path,
            )
            return os.open(path, os.O_RDONLY)
        raise


def _read_exact_into_buffer(
    fd: int,
    buf: memoryview,
    file_offset: int,
    size: int,
    stats: Optional[_TransferStats],
) -> None:
    t0 = time.monotonic()
    done = 0
    while done < size:
        read = os.preadv(fd, [buf[done:size]], file_offset + done)
        if read == 0:
            raise RuntimeError(f"short read at offset {file_offset + done}")
        done += read
    if stats is not None:
        stats.record_read(time.monotonic() - t0, size)


class _PinnedCopySlot:
    def __init__(self, cuda: Any, size: int) -> None:
        self._cuda = cuda
        self.view, self._raw, self.ptr = _allocate_aligned_buffer(size)
        self.stream = cuda.create_stream()
        self.busy = False
        self.byte_count = 0
        self._registered = False
        try:
            cuda.host_register(self.ptr, size)
            self._registered = True
        except Exception:
            try:
                cuda.destroy_stream(self.stream)
            finally:
                _free_aligned_buffer(self.view, self.ptr)
            raise

    def wait(self, stats: Optional[_TransferStats]) -> None:
        if not self.busy:
            return
        t0 = time.monotonic()
        self._cuda.synchronize_stream(self.stream)
        if stats is not None:
            stats.record_copy(time.monotonic() - t0, self.byte_count)
        self.busy = False
        self.byte_count = 0

    def close(self) -> None:
        self.wait(None)
        try:
            if self._registered:
                self._cuda.host_unregister(self.ptr)
                self._registered = False
        finally:
            try:
                self._cuda.destroy_stream(self.stream)
            finally:
                _free_aligned_buffer(self.view, self.ptr)


class LocalSSDPinnedTransferBackend:
    """Same-node local SSD restore with reusable pinned host buffers.

    This is still CPU-staged, but it avoids the default path's full-shard numpy
    allocations, PyTorch tensor wrappers, and pageable H2D copies.
    """

    name = LOCAL_SSD_PINNED_TRANSFER_BACKEND

    def __init__(
        self,
        *,
        device: int,
        max_workers: int,
        local_roots: Sequence[str],
    ) -> None:
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._local_roots = [
            os.path.abspath(root) for root in local_roots if str(root).strip()
        ]
        if not self._local_roots:
            raise RuntimeError(
                f"{LOCAL_SSD_PINNED_TRANSFER_BACKEND} requires "
                f"{GMS_LOCAL_SSD_ROOTS_ENV}=<root0>,<root1>,..."
            )
        self._cuda = _load_cuda_runtime()
        self._cuda.set_device(device)

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _LocalSSDPinnedTransferSession(
            cuda=self._cuda,
            device=self._device,
            max_workers=self._max_workers,
            local_roots=self._local_roots,
            sources=sources,
        )

    def close(self) -> None:
        pass


class _LocalSSDPinnedTransferSession:
    def __init__(
        self,
        *,
        cuda: Any,
        device: int,
        max_workers: int,
        local_roots: Sequence[str],
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._cuda = cuda
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._local_roots = list(local_roots)
        self._sources = list(sources)
        self._cancel_event = threading.Event()
        self._active = True

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        self._validate_targets(targets)
        root_groups = _group_sources_by_local_root(
            self._sources,
            self._local_roots,
            LOCAL_SSD_PINNED_TRANSFER_BACKEND,
        )
        if not root_groups:
            self._active = False
            return

        worker_count = min(self._max_workers, len(root_groups))
        if worker_count < len(root_groups):
            logger.warning(
                "%s has %d active SSD roots but only %d workers; "
                "increase GMS_LOAD_WORKERS for full stripe parallelism",
                LOCAL_SSD_PINNED_TRANSFER_BACKEND,
                len(root_groups),
                worker_count,
            )

        stats = _TransferStats(
            backend_name=LOCAL_SSD_PINNED_TRANSFER_BACKEND,
            total_bytes=sum(source.byte_count for source in self._sources),
            worker_count=worker_count,
            shard_count=sum(len(grouped) for grouped in root_groups.values()),
        )
        stats.record_mode("odirect-pinned-cudamemcpy")
        transfer_t0 = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        self._restore_root,
                        root,
                        grouped_paths,
                        targets,
                        stats,
                    ): root
                    for root, grouped_paths in root_groups.items()
                }
                for future in as_completed(futures):
                    root = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        self._cancel_event.set()
                        raise RuntimeError(
                            f"{LOCAL_SSD_PINNED_TRANSFER_BACKEND} failed for "
                            f"root {root}: {exc}"
                        ) from exc
        finally:
            self._active = False
            stats.record_transfer_wall(time.monotonic() - transfer_t0)
            stats.log()

    def close(self) -> None:
        self._cancel_event.set()
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

    def _restore_root(
        self,
        root: str,
        grouped_paths: List[Tuple[str, List[FileTransferSource]]],
        targets: Mapping[str, GMSTransferTarget],
        stats: _TransferStats,
    ) -> None:
        slots: List[_PinnedCopySlot] = []
        root_t0 = time.monotonic()
        root_bytes = 0
        next_slot = 0
        try:
            for _ in range(_PINNED_COPY_BUFFERS_PER_ROOT):
                slots.append(_PinnedCopySlot(self._cuda, _PINNED_COPY_CHUNK_SIZE))
            for file_path, sources in grouped_paths:
                fd = _open_read_fd(file_path)
                try:
                    for source in sources:
                        copied, next_slot = self._restore_source(
                            fd,
                            source,
                            targets[source.allocation_id],
                            slots,
                            next_slot,
                            stats,
                        )
                        root_bytes += copied
                finally:
                    os.close(fd)
            for slot in slots:
                slot.wait(stats)
        finally:
            for slot in slots:
                try:
                    slot.close()
                except Exception:
                    logger.warning(
                        "failed to release pinned copy slot for %s",
                        root,
                        exc_info=True,
                    )
            elapsed = time.monotonic() - root_t0
            stats.record_disk_task(elapsed)
            throughput = root_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
            logger.info(
                "%s completed root=%s shards=%d bytes=%.2f GiB "
                "elapsed=%.3fs bw=%.2f GiB/s",
                LOCAL_SSD_PINNED_TRANSFER_BACKEND,
                root,
                len(grouped_paths),
                root_bytes / (1024**3),
                elapsed,
                throughput,
            )

    def _restore_source(
        self,
        fd: int,
        source: FileTransferSource,
        target: GMSTransferTarget,
        slots: List[_PinnedCopySlot],
        next_slot: int,
        stats: _TransferStats,
    ) -> Tuple[int, int]:
        done = 0
        while done < source.byte_count:
            if self._cancel_event.is_set():
                raise CancelledError(f"{LOCAL_SSD_PINNED_TRANSFER_BACKEND} cancelled")
            slot = slots[next_slot]
            slot.wait(stats)
            chunk_size = min(_PINNED_COPY_CHUNK_SIZE, source.byte_count - done)
            _read_exact_into_buffer(
                fd,
                slot.view,
                source.file_offset + done,
                chunk_size,
                stats,
            )
            self._cuda.memcpy_h2d_async(
                target.va + done,
                slot.ptr,
                chunk_size,
                slot.stream,
            )
            slot.busy = True
            slot.byte_count = chunk_size
            done += chunk_size
            next_slot = (next_slot + 1) % len(slots)
        return done, next_slot


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
        nixl_agent, nixl_agent_config = _load_nixl_api()
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
