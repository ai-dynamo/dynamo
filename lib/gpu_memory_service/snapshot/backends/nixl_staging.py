# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL POSIX FILE -> pinned DRAM -> VRAM staging restore."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.snapshot.backends.nixl_common import (
    DRAM_MEM_TYPE,
    FILE_MEM_TYPE,
    NIXL_POSIX_BACKEND,
    NixlApi,
    create_nixl_agent,
    load_nixl_api,
    open_direct_read_fd,
    release_transfer_resources,
    wait_for_transfer,
)
from gpu_memory_service.snapshot.backends.pinned_host import (
    PINNED_COPY_CHUNK_SIZE,
    PinnedCopySlot,
    close_pinned_copy_slots,
    make_pinned_copy_slots,
)
from gpu_memory_service.snapshot.transfer import (
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferSession,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)

_PINNED_COPY_BUFFERS_PER_WORKER = 2

NixlFileGroup = Tuple[str, Sequence[FileTransferSource]]
NixlWorkGroup = Tuple[str, Sequence[NixlFileGroup]]
NixlGroupingFn = Callable[
    [Sequence[FileTransferSource]], Mapping[str, List[NixlFileGroup]]
]


class NixlPosixStagingTransferBackend:
    """Restore files through NIXL POSIX direct I/O and pinned host staging."""

    def __init__(
        self,
        *,
        config: GMSSnapshotConfig,
        backend_name: str,
        group_sources: NixlGroupingFn,
        group_kind: str,
        warn_under_parallelized: bool = False,
    ) -> None:
        self.name = backend_name
        self._api = load_nixl_api()
        self._device = config.device
        self._max_workers = config.max_workers
        self._group_sources = group_sources
        self._group_kind = group_kind
        self._warn_under_parallelized = warn_under_parallelized
        cuda_utils.cuda_runtime_set_device(self._device)
        logger.info(
            "%s initialized for device %d with %d workers using NIXL POSIX staging",
            backend_name,
            self._device,
            self._max_workers,
        )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _NixlPosixStagingTransferSession(
            api=self._api,
            backend_name=self.name,
            device=self._device,
            max_workers=self._max_workers,
            group_sources=self._group_sources,
            group_kind=self._group_kind,
            warn_under_parallelized=self._warn_under_parallelized,
            sources=sources,
        )

    def close(self) -> None:
        pass


class _NixlPosixStagingTransferSession:
    def __init__(
        self,
        *,
        api: NixlApi,
        backend_name: str,
        device: int,
        max_workers: int,
        group_sources: NixlGroupingFn,
        group_kind: str,
        warn_under_parallelized: bool,
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._api = api
        self._backend_name = backend_name
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._group_sources = group_sources
        self._group_kind = group_kind
        self._warn_under_parallelized = warn_under_parallelized
        self._sources = list(sources)
        self._agent_name_base = (
            f"gms_{backend_name.replace('-', '_')}_{device}_{os.getpid()}_{id(self):x}"
        )
        self._cancel_event = threading.Event()
        self._active = True

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._device)
        grouped = self._group_sources(self._sources)
        if not grouped:
            self._active = False
            return

        work_groups: List[NixlWorkGroup] = [
            (group_name, file_groups) for group_name, file_groups in grouped.items()
        ]
        worker_count = min(self._max_workers, len(work_groups))
        if self._warn_under_parallelized and worker_count < len(work_groups):
            logger.warning(
                "%s has %d active %s groups but only %d workers; "
                "increase GMS_LOAD_WORKERS for full parallelism",
                self._backend_name,
                len(work_groups),
                self._group_kind,
                worker_count,
            )

        total_bytes = sum(source.byte_count for source in self._sources)
        t0 = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        self._restore_group,
                        worker_index,
                        group_name,
                        file_groups,
                        targets,
                    ): group_name
                    for worker_index, (group_name, file_groups) in enumerate(
                        work_groups
                    )
                }
                for future in as_completed(futures):
                    group_name = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        self._cancel_event.set()
                        raise RuntimeError(
                            f"{self._backend_name} failed for "
                            f"{self._group_kind} group {group_name}: {exc}"
                        ) from exc
        finally:
            self._active = False

        elapsed = time.monotonic() - t0
        throughput = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
        logger.info(
            "%s transfers complete: %.2f GiB in %.3fs (%.2f GiB/s, %s_groups=%d)",
            self._backend_name,
            total_bytes / (1024**3),
            elapsed,
            throughput,
            self._group_kind,
            len(work_groups),
        )

    def close(self) -> None:
        self._cancel_event.set()
        self._active = False

    def _restore_group(
        self,
        worker_index: int,
        group_name: str,
        file_groups: Sequence[NixlFileGroup],
        targets: Mapping[str, GMSTransferTarget],
    ) -> None:
        cuda_utils.cuda_runtime_set_device(self._device)
        agent_name = f"{self._agent_name_base}_{worker_index}"
        agent = create_nixl_agent(
            self._api,
            agent_name=agent_name,
            backend_name=NIXL_POSIX_BACKEND,
        )
        group_t0 = time.monotonic()
        group_bytes = 0
        try:
            group_bytes = restore_file_groups_with_nixl_staging(
                backend_name=self._backend_name,
                agent=agent,
                agent_name=agent_name,
                file_groups=file_groups,
                targets=targets,
                cancel_event=self._cancel_event,
                buffers_per_worker=_PINNED_COPY_BUFFERS_PER_WORKER,
            )
        finally:
            elapsed = time.monotonic() - group_t0
            throughput = group_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
            logger.info(
                "%s completed %s=%s files=%d bytes=%.2f GiB elapsed=%.3fs "
                "bw=%.2f GiB/s",
                self._backend_name,
                self._group_kind,
                group_name,
                len(file_groups),
                group_bytes / (1024**3),
                elapsed,
                throughput,
            )


class NixlPosixFileReader:
    """NIXL POSIX FILE reader for pinned host staging slots."""

    def __init__(
        self,
        *,
        agent: object,
        agent_name: str,
        file_path: str,
        backend_name: str,
    ) -> None:
        self._agent = agent
        self._agent_name = agent_name
        self._file_path = file_path
        self._backend_name = backend_name
        self._fd = open_direct_read_fd(file_path, logger=logger, require_direct=True)

    def read_into_slot(
        self,
        slot: PinnedCopySlot,
        file_offset: int,
        size: int,
    ) -> None:
        _read_file_to_dram(
            self._agent,
            self._agent_name,
            self._fd,
            self._file_path,
            file_offset,
            slot.ptr,
            size,
            self._backend_name,
        )

    def close(self) -> None:
        os.close(self._fd)


def restore_file_groups_with_nixl_staging(
    *,
    backend_name: str,
    agent: object,
    agent_name: str,
    file_groups: Sequence[NixlFileGroup],
    targets: Mapping[str, GMSTransferTarget],
    cancel_event: Optional[threading.Event] = None,
    buffers_per_worker: int = _PINNED_COPY_BUFFERS_PER_WORKER,
) -> int:
    slots: List[PinnedCopySlot] = []
    total_bytes = 0
    next_slot = 0
    try:
        slots = make_pinned_copy_slots(buffers_per_worker)
        for file_path, sources in file_groups:
            reader = NixlPosixFileReader(
                agent=agent,
                agent_name=agent_name,
                file_path=file_path,
                backend_name=backend_name,
            )
            try:
                for source in sources:
                    copied, next_slot = _restore_source(
                        backend_name=backend_name,
                        reader=reader,
                        source=source,
                        target=targets[source.allocation_id],
                        slots=slots,
                        next_slot=next_slot,
                        cancel_event=cancel_event,
                    )
                    total_bytes += copied
            finally:
                reader.close()

        for slot in slots:
            slot.wait()
        return total_bytes
    finally:
        close_pinned_copy_slots(
            slots,
            logger,
            "failed to release NIXL pinned copy slot",
        )


def _restore_source(
    *,
    backend_name: str,
    reader: NixlPosixFileReader,
    source: FileTransferSource,
    target: GMSTransferTarget,
    slots: List[PinnedCopySlot],
    next_slot: int,
    cancel_event: Optional[threading.Event],
) -> Tuple[int, int]:
    done = 0
    while done < source.byte_count:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError(f"{backend_name} cancelled")

        slot = slots[next_slot]
        slot.wait()
        chunk_size = min(PINNED_COPY_CHUNK_SIZE, source.byte_count - done)
        reader.read_into_slot(
            slot,
            source.file_offset + done,
            chunk_size,
        )
        slot.copy_to_device_async(target.va + done, chunk_size)
        done += chunk_size
        next_slot = (next_slot + 1) % len(slots)

    return done, next_slot


def _read_file_to_dram(
    agent: object,
    agent_name: str,
    fd: int,
    file_path: str,
    file_offset: int,
    host_ptr: int,
    size: int,
    backend_name: str,
) -> None:
    file_reg = None
    host_reg = None
    handle = None
    try:
        file_reg = agent.register_memory(
            [(file_offset, size, fd, "")],
            FILE_MEM_TYPE,
        )
        host_reg = agent.register_memory(
            [(host_ptr, size, 0, "")],
            DRAM_MEM_TYPE,
        )
        handle = agent.initialize_xfer(
            "READ",
            host_reg.trim(),
            file_reg.trim(),
            agent_name,
        )
        wait_for_transfer(agent, handle, file_path, backend_name)
    finally:
        release_transfer_resources(agent, handle, host_reg, file_reg)
