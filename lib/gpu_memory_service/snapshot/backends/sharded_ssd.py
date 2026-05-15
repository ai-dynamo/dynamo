# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharded SSD transfer backend for GMS snapshot restore."""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.snapshot.backends.pinned_host import (
    PINNED_COPY_CHUNK_SIZE,
    PinnedCopySlot,
    close_pinned_copy_slots,
    make_pinned_copy_slots,
)
from gpu_memory_service.snapshot.transfer import (
    SHARDED_SSD_TRANSFER_BACKEND,
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferSession,
    group_sources_by_path,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)

GMS_SHARDED_SSD_ROOTS_ENV = "GMS_SHARDED_SSD_ROOTS"
SHARDED_SSD_ROOTS_CONFIG_KEY = "sharded_ssd_roots"

_PINNED_COPY_BUFFERS_PER_ROOT = 2


def parse_sharded_ssd_roots(value: str | None) -> List[str]:
    if not value:
        return []
    return _normalize_roots(value.split(","))


def device_sharded_ssd_roots(
    checkpoint_dir: str,
    device: int,
    roots: Sequence[str],
) -> List[str]:
    suffix = _checkpoint_suffix(checkpoint_dir) / f"device-{device}"
    return [str(Path(root) / suffix) for root in _normalize_roots(roots)]


def _checkpoint_suffix(checkpoint_dir: str) -> Path:
    parts = Path(checkpoint_dir).parts
    if "versions" in parts:
        idx = parts.index("versions")
        if idx > 0 and idx + 1 < len(parts):
            return Path(parts[idx - 1]) / "versions" / parts[idx + 1]
    return Path(checkpoint_dir.strip(os.sep).replace(os.sep, "_"))


def _normalize_roots(values: Sequence[str]) -> List[str]:
    return [
        os.path.abspath(str(value).strip()) for value in values if str(value).strip()
    ]


def _roots_from_config(config: Mapping[str, Any]) -> List[str]:
    configured = config.get(SHARDED_SSD_ROOTS_CONFIG_KEY)
    if configured is None:
        return parse_sharded_ssd_roots(os.environ.get(GMS_SHARDED_SSD_ROOTS_ENV))
    if isinstance(configured, str):
        return parse_sharded_ssd_roots(configured)
    return _normalize_roots(configured)


def _match_root(file_path: str, roots: Sequence[str]) -> Optional[str]:
    abs_path = os.path.abspath(file_path)
    for root in roots:
        try:
            if os.path.commonpath([root, abs_path]) == root:
                return root
        except ValueError:
            continue
    return None


def _group_sources_by_root(
    sources: Sequence[FileTransferSource],
    roots: Sequence[str],
) -> Mapping[str, List[Tuple[str, List[FileTransferSource]]]]:
    groups_by_path = group_sources_by_path(sources)
    groups: dict[str, List[Tuple[str, List[FileTransferSource]]]] = {}
    for file_path, grouped_sources in groups_by_path.items():
        root = _match_root(file_path, roots)
        if root is None:
            raise RuntimeError(
                f"{SHARDED_SSD_TRANSFER_BACKEND} source path {file_path!r} is not "
                f"under any {GMS_SHARDED_SSD_ROOTS_ENV} root: {list(roots)}"
            )
        groups.setdefault(root, []).append((file_path, grouped_sources))
    for grouped_paths in groups.values():
        grouped_paths.sort(key=lambda item: item[0])
    return groups


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
) -> None:
    done = 0
    while done < size:
        read = os.preadv(fd, [buf[done:size]], file_offset + done)
        if read == 0:
            raise RuntimeError(f"short read at offset {file_offset + done}")
        done += read


class ShardedSSDTransferBackend:
    """Same-node sharded SSD restore with reusable pinned host buffers."""

    name = SHARDED_SSD_TRANSFER_BACKEND

    def __init__(
        self,
        *,
        config: GMSSnapshotConfig,
    ) -> None:
        self._device = config.device
        self._max_workers = config.max_workers
        self._roots = _roots_from_config(config.backend_config)
        if not self._roots:
            raise RuntimeError(
                f"{SHARDED_SSD_TRANSFER_BACKEND} requires "
                f"{GMS_SHARDED_SSD_ROOTS_ENV}=<root0>,<root1>,..."
            )
        cuda_utils.cuda_runtime_set_device(self._device)

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _ShardedSSDTransferSession(
            device=self._device,
            max_workers=self._max_workers,
            roots=self._roots,
            sources=sources,
        )

    def close(self) -> None:
        pass


class _ShardedSSDTransferSession:
    def __init__(
        self,
        *,
        device: int,
        max_workers: int,
        roots: Sequence[str],
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._roots = list(roots)
        self._sources = list(sources)
        self._cancel_event = threading.Event()
        self._active = True

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._device)
        root_groups = _group_sources_by_root(self._sources, self._roots)
        if not root_groups:
            self._active = False
            return

        worker_count = min(self._max_workers, len(root_groups))
        if worker_count < len(root_groups):
            logger.warning(
                "%s has %d active SSD roots but only %d workers; "
                "increase GMS_LOAD_WORKERS for full shard parallelism",
                SHARDED_SSD_TRANSFER_BACKEND,
                len(root_groups),
                worker_count,
            )

        total_bytes = sum(source.byte_count for source in self._sources)
        t0 = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        self._restore_root,
                        root,
                        grouped_paths,
                        targets,
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
                            f"{SHARDED_SSD_TRANSFER_BACKEND} failed for root {root}: {exc}"
                        ) from exc
        finally:
            self._active = False

        elapsed = time.monotonic() - t0
        throughput = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
        logger.info(
            "%s transfers complete: %.2f GiB in %.3fs (%.2f GiB/s, roots=%d)",
            SHARDED_SSD_TRANSFER_BACKEND,
            total_bytes / (1024**3),
            elapsed,
            throughput,
            len(root_groups),
        )

    def close(self) -> None:
        self._cancel_event.set()
        self._active = False

    def _restore_root(
        self,
        root: str,
        grouped_paths: List[Tuple[str, List[FileTransferSource]]],
        targets: Mapping[str, GMSTransferTarget],
    ) -> None:
        slots: List[PinnedCopySlot] = []
        root_t0 = time.monotonic()
        root_bytes = 0
        next_slot = 0
        try:
            slots = make_pinned_copy_slots(_PINNED_COPY_BUFFERS_PER_ROOT)
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
                        )
                        root_bytes += copied
                finally:
                    os.close(fd)
            for slot in slots:
                slot.wait()
        finally:
            close_pinned_copy_slots(
                slots,
                logger,
                "failed to release pinned copy slot for %s",
                root,
            )
            elapsed = time.monotonic() - root_t0
            throughput = root_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
            logger.info(
                "%s completed root=%s shards=%d bytes=%.2f GiB "
                "elapsed=%.3fs bw=%.2f GiB/s",
                SHARDED_SSD_TRANSFER_BACKEND,
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
        slots: List[PinnedCopySlot],
        next_slot: int,
    ) -> Tuple[int, int]:
        done = 0
        while done < source.byte_count:
            if self._cancel_event.is_set():
                raise CancelledError(f"{SHARDED_SSD_TRANSFER_BACKEND} cancelled")
            slot = slots[next_slot]
            slot.wait()
            chunk_size = min(PINNED_COPY_CHUNK_SIZE, source.byte_count - done)
            _read_exact_into_buffer(
                fd,
                slot.view,
                source.file_offset + done,
                chunk_size,
            )
            slot.copy_to_device_async(target.va + done, chunk_size)
            done += chunk_size
            next_slot = (next_slot + 1) % len(slots)
        return done, next_slot
