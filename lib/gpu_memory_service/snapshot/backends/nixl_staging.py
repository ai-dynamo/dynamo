# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL POSIX FILE -> pinned DRAM -> VRAM staging restore."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping as MappingABC
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence

from gpu_memory_service.common.vmm import VMMDevice, get_vmm_device
from gpu_memory_service.snapshot.backends.nixl_common import (
    DRAM_MEM_TYPE,
    FILE_MEM_TYPE,
    NIXL_POSIX_BACKEND,
    NixlFileGroup,
    NixlWorkGroup,
    create_nixl_agent,
    load_nixl_api,
    open_direct_read_fd,
    release_transfer_resources,
    split_work_groups,
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

NixlGroupingFn = Callable[
    [Sequence[FileTransferSource]], Mapping[str, List[NixlFileGroup]]
]


@dataclass
class _PreparedNixlGroup:
    group_name: str
    file_groups: Sequence[NixlFileGroup]
    agent: object
    agent_name: str
    slots: List[PinnedCopySlot]
    prep_elapsed_s: float
    closed: bool = False


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
        self._backend_name = backend_name
        self._device = config.device
        self._vmm = get_vmm_device(config.device_kind)
        self._max_workers = config.max_workers
        # These are NIXL POSIX backend custom-param keys. NIXL's POSIX default
        # preallocates a large I/O pool, while GMS staging workers issue one
        # POSIX NIXL transfer at a time. Keep smaller GMS defaults, but let
        # callers override or add NIXL POSIX params through backend_config.
        self._posix_backend_params = {
            "ios_pool_size": "1024",
            "kernel_queue_size": "128",
        }
        posix_backend_params = config.backend_config.get("posix_backend_params")
        if posix_backend_params is not None:
            if not isinstance(posix_backend_params, MappingABC):
                raise TypeError("posix_backend_params must be a mapping")
            self._posix_backend_params.update(
                {str(key): str(value) for key, value in posix_backend_params.items()}
            )
        self._api_pool = ThreadPoolExecutor(max_workers=1)
        self._api_future = self._api_pool.submit(load_nixl_api)
        self._group_sources = group_sources
        self._group_kind = group_kind
        self._warn_under_parallelized = warn_under_parallelized
        logger.info(
            "%s configured for device %d with %d workers using NIXL POSIX "
            "staging backend_params=%s; NIXL import is running in the "
            "background and agent setup starts in start_restore() so it can "
            "overlap manifest planning, RW connect, and restore target allocation",
            backend_name,
            self._device,
            self._max_workers,
            self._posix_backend_params,
        )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _NixlPosixStagingTransferSession(
            backend_name=self._backend_name,
            vmm=self._vmm,
            device=self._device,
            max_workers=self._max_workers,
            group_sources=self._group_sources,
            group_kind=self._group_kind,
            warn_under_parallelized=self._warn_under_parallelized,
            posix_backend_params=self._posix_backend_params,
            sources=sources,
            api_future=self._api_future,
        )

    def close(self) -> None:
        self._api_pool.shutdown(wait=True, cancel_futures=True)


class _NixlPosixStagingTransferSession:
    def __init__(
        self,
        *,
        backend_name: str,
        device: int,
        max_workers: int,
        group_sources: NixlGroupingFn,
        group_kind: str,
        warn_under_parallelized: bool,
        posix_backend_params: Mapping[str, str],
        sources: Sequence[FileTransferSource],
        vmm: VMMDevice,
        api_future: Optional[Future[object]] = None,
    ) -> None:
        self._backend_name = backend_name
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._vmm = vmm
        self._group_kind = group_kind
        self._posix_backend_params = dict(posix_backend_params)
        self._sources = list(sources)
        self._api_future = api_future
        self._agent_name_base = (
            f"gms_{backend_name.replace('-', '_')}_{device}_{os.getpid()}_{id(self):x}"
        )
        self._cancel_event = threading.Event()
        self._prep_started_at = time.monotonic()
        grouped = group_sources(self._sources)
        self._logical_group_count = len(grouped)
        work_groups: List[NixlWorkGroup] = [
            (group_name, file_groups) for group_name, file_groups in grouped.items()
        ]
        self._worker_count = min(self._max_workers, self._logical_group_count)
        if warn_under_parallelized and self._worker_count < self._logical_group_count:
            logger.warning(
                "%s has %d active %s groups but only %d workers; "
                "increase --max-workers for full parallelism",
                self._backend_name,
                self._logical_group_count,
                self._group_kind,
                self._worker_count,
            )
        self._work_groups = split_work_groups(work_groups, self._worker_count)
        self._total_bytes = sum(source.byte_count for source in self._sources)
        self._prep_pool: Optional[ThreadPoolExecutor] = None
        self._prep_futures: dict[Future[_PreparedNixlGroup], str] = {}
        if self._work_groups:
            self._prep_pool = ThreadPoolExecutor(max_workers=self._worker_count)
            self._prep_futures = {
                self._prep_pool.submit(
                    self._prepare_group,
                    worker_index,
                    group_name,
                    file_groups,
                ): group_name
                for worker_index, (group_name, file_groups) in enumerate(
                    self._work_groups
                )
            }
            logger.info(
                "%s staging prep started: work_groups=%d logical_%s_groups=%d "
                "workers=%d bytes=%.2f GiB",
                self._backend_name,
                len(self._work_groups),
                self._group_kind,
                self._logical_group_count,
                self._worker_count,
                self._total_bytes / (1024**3),
            )

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._device)
        if not self._work_groups:
            return

        t0 = time.monotonic()
        prep_overlap_s = t0 - self._prep_started_at
        logger.info(
            "%s restore targets ready after %.3fs of background staging prep; "
            "starting transfers",
            self._backend_name,
            prep_overlap_s,
        )
        try:
            with ThreadPoolExecutor(max_workers=self._worker_count) as pool:
                transfer_futures: dict[Future[None], str] = {}
                for prep_future in as_completed(self._prep_futures):
                    group_name = self._prep_futures[prep_future]
                    try:
                        prepared = prep_future.result()
                    except Exception as exc:
                        self._cancel_event.set()
                        raise RuntimeError(
                            f"{self._backend_name} failed while preparing "
                            f"{self._group_kind} group {group_name}: {exc}"
                        ) from exc
                    try:
                        transfer_futures[
                            pool.submit(
                                self._restore_prepared_group,
                                prepared,
                                targets,
                            )
                        ] = group_name
                    except Exception:
                        self._close_prepared_group(prepared)
                        raise

                for transfer_future in as_completed(transfer_futures):
                    group_name = transfer_futures[transfer_future]
                    try:
                        transfer_future.result()
                    except Exception as exc:
                        self._cancel_event.set()
                        raise RuntimeError(
                            f"{self._backend_name} failed for "
                            f"{self._group_kind} group {group_name}: {exc}"
                        ) from exc
        finally:
            if self._prep_pool is not None:
                self._prep_pool.shutdown(wait=True, cancel_futures=True)

        elapsed = time.monotonic() - t0
        throughput = self._total_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
        logger.info(
            "%s transfers complete: %.2f GiB in %.3fs (%.2f GiB/s, %s_groups=%d)",
            self._backend_name,
            self._total_bytes / (1024**3),
            elapsed,
            throughput,
            self._group_kind,
            self._logical_group_count,
        )

    def close(self) -> None:
        self._cancel_event.set()
        if self._prep_pool is not None:
            self._prep_pool.shutdown(wait=True, cancel_futures=True)
        for future in self._prep_futures:
            if not future.done() or future.cancelled():
                continue
            try:
                prepared = future.result()
            except Exception:
                continue
            self._close_prepared_group(prepared)
        self._prep_futures.clear()

    def _prepare_group(
        self,
        worker_index: int,
        group_name: str,
        file_groups: Sequence[NixlFileGroup],
    ) -> _PreparedNixlGroup:
        prep_started_after_s = time.monotonic() - self._prep_started_at
        prep_t0 = time.monotonic()
        slots: List[PinnedCopySlot] = []
        agent: Optional[object] = None
        agent_name = f"{self._agent_name_base}_{worker_index}"
        try:
            if self._cancel_event.is_set():
                raise CancelledError(f"{self._backend_name} cancelled")
            api = (
                self._api_future.result()
                if self._api_future is not None
                else load_nixl_api()
            )
            if self._cancel_event.is_set():
                raise CancelledError(f"{self._backend_name} cancelled")
            self._vmm.runtime_set_device(self._device)
            agent = create_nixl_agent(
                api,
                agent_name=agent_name,
                backend_name=NIXL_POSIX_BACKEND,
                backend_params=self._posix_backend_params,
            )
            if self._cancel_event.is_set():
                raise CancelledError(f"{self._backend_name} cancelled")
            slots = make_pinned_copy_slots(self._vmm, _PINNED_COPY_BUFFERS_PER_WORKER)
            prep_elapsed_s = time.monotonic() - prep_t0
            logger.info(
                "%s prepared %s=%s files=%d prep_elapsed=%.3fs "
                "prep_started_after=%.3fs",
                self._backend_name,
                self._group_kind,
                group_name,
                len(file_groups),
                prep_elapsed_s,
                prep_started_after_s,
            )
            return _PreparedNixlGroup(
                group_name=group_name,
                file_groups=file_groups,
                agent=agent,
                agent_name=agent_name,
                slots=slots,
                prep_elapsed_s=prep_elapsed_s,
            )
        except Exception:
            close_pinned_copy_slots(
                slots,
                logger,
                "failed to release prepared NIXL pinned copy slot",
            )
            raise

    def _restore_prepared_group(
        self,
        prepared: _PreparedNixlGroup,
        targets: Mapping[str, GMSTransferTarget],
    ) -> None:
        self._vmm.runtime_set_device(self._device)
        group_t0 = time.monotonic()
        group_bytes = 0
        try:
            group_bytes = restore_file_groups_with_nixl_staging(
                backend_name=self._backend_name,
                vmm=self._vmm,
                agent=prepared.agent,
                agent_name=prepared.agent_name,
                file_groups=prepared.file_groups,
                targets=targets,
                cancel_event=self._cancel_event,
                slots=prepared.slots,
            )
        finally:
            elapsed = time.monotonic() - group_t0
            throughput = group_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
            logger.info(
                "%s completed %s=%s files=%d bytes=%.2f GiB elapsed=%.3fs "
                "bw=%.2f GiB/s prep_elapsed=%.3fs",
                self._backend_name,
                self._group_kind,
                prepared.group_name,
                len(prepared.file_groups),
                group_bytes / (1024**3),
                elapsed,
                throughput,
                prepared.prep_elapsed_s,
            )
            self._close_prepared_group(prepared)

    def _close_prepared_group(self, prepared: _PreparedNixlGroup) -> None:
        if prepared.closed:
            return
        prepared.closed = True
        close_pinned_copy_slots(
            prepared.slots,
            logger,
            "failed to release prepared NIXL pinned copy slot",
        )


def restore_file_groups_with_nixl_staging(
    *,
    backend_name: str,
    vmm: VMMDevice,
    agent: object,
    agent_name: str,
    file_groups: Sequence[NixlFileGroup],
    targets: Mapping[str, GMSTransferTarget],
    cancel_event: Optional[threading.Event] = None,
    buffers_per_worker: int = _PINNED_COPY_BUFFERS_PER_WORKER,
    slots: Optional[List[PinnedCopySlot]] = None,
) -> int:
    owned_slots = slots is None
    if slots is None:
        slots = []
    total_bytes = 0
    next_slot = 0
    try:
        if owned_slots:
            slots = make_pinned_copy_slots(vmm, buffers_per_worker)
        for file_path, sources in file_groups:
            fd = open_direct_read_fd(file_path, logger=logger, require_direct=True)
            try:
                for source in sources:
                    target = targets[source.allocation_id]
                    done = 0
                    while done < source.byte_count:
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError(f"{backend_name} cancelled")

                        slot = slots[next_slot]
                        slot.wait()
                        chunk_size = min(
                            PINNED_COPY_CHUNK_SIZE,
                            source.byte_count - done,
                        )
                        _read_file_to_dram(
                            agent,
                            agent_name,
                            fd,
                            file_path,
                            source.file_offset + done,
                            slot.ptr,
                            chunk_size,
                            backend_name,
                        )
                        slot.copy_to_device_async(target.va + done, chunk_size)
                        done += chunk_size
                        total_bytes += chunk_size
                        next_slot = (next_slot + 1) % len(slots)
            finally:
                os.close(fd)

        for slot in slots:
            slot.wait()
        return total_bytes
    finally:
        if owned_slots:
            close_pinned_copy_slots(
                slots,
                logger,
                "failed to release NIXL pinned copy slot",
            )


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
