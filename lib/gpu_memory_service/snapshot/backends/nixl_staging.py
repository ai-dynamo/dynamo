# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL POSIX FILE -> pinned DRAM -> VRAM staging restore."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.snapshot.backends.nixl_common import (
    DRAM_MEM_TYPE,
    FILE_MEM_TYPE,
    NIXL_POSIX_BACKEND,
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
POSIX_IOS_POOL_SIZE_CONFIG_KEY = "nixl_posix_ios_pool_size"
POSIX_KERNEL_QUEUE_SIZE_CONFIG_KEY = "nixl_posix_kernel_queue_size"
DEFAULT_POSIX_IOS_POOL_SIZE = 1024
DEFAULT_POSIX_KERNEL_QUEUE_SIZE = 128

NixlFileGroup = Tuple[str, Sequence[FileTransferSource]]
NixlWorkGroup = Tuple[str, Sequence[NixlFileGroup]]
NixlGroupingFn = Callable[
    [Sequence[FileTransferSource]], Mapping[str, List[NixlFileGroup]]
]


@dataclass
class _PreparedNixlGroup:
    worker_index: int
    group_name: str
    file_groups: Sequence[NixlFileGroup]
    agent: object
    agent_name: str
    slots: List[PinnedCopySlot]
    prep_elapsed_s: float
    prep_started_after_s: float
    closed: bool = False


def _positive_int_config(
    config: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    raw_value = config.get(key)
    if raw_value is None or raw_value == "":
        return default

    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{key} must be a positive integer, got {raw_value!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {raw_value!r}")
    return value


def _posix_backend_params_from_config(
    config: Mapping[str, object],
) -> Mapping[str, str]:
    """Return bounded NIXL POSIX backend params for GMS staging agents.

    NIXL's POSIX backend default preallocates a large I/O pool.  GMS staging
    workers issue one POSIX NIXL transfer at a time, so a smaller explicit
    pool is sufficient and avoids spending agent startup time building an
    oversized default pool.
    """
    return {
        "ios_pool_size": str(
            _positive_int_config(
                config,
                POSIX_IOS_POOL_SIZE_CONFIG_KEY,
                DEFAULT_POSIX_IOS_POOL_SIZE,
            )
        ),
        "kernel_queue_size": str(
            _positive_int_config(
                config,
                POSIX_KERNEL_QUEUE_SIZE_CONFIG_KEY,
                DEFAULT_POSIX_KERNEL_QUEUE_SIZE,
            )
        ),
    }


def _file_group_size(file_group: NixlFileGroup) -> int:
    return sum(source.byte_count for source in file_group[1])


def _split_work_groups(
    work_groups: Sequence[NixlWorkGroup],
    worker_count: int,
) -> List[NixlWorkGroup]:
    """Split logical work groups into at most worker_count balanced buckets.

    The grouping function chooses the storage-affinity unit for a backend: one
    checkpoint file for the default NIXL/POSIX backend, or one SSD root for the
    sharded-SSD backend.  The staging code creates one NIXL agent and one pair
    of pinned staging buffers per submitted work item, so submitting one item per
    checkpoint shard can create hundreds of agents for large models.  Instead,
    preserve each logical file/root group internally while coalescing them into
    a bounded number of balanced worker buckets.
    """
    if not work_groups:
        return []

    worker_count = max(1, min(int(worker_count), len(work_groups)))
    if len(work_groups) <= worker_count:
        return list(work_groups)

    bucket_file_groups: List[List[NixlFileGroup]] = [[] for _ in range(worker_count)]
    bucket_names: List[List[str]] = [[] for _ in range(worker_count)]
    bucket_bytes = [0] * worker_count

    # Largest-first greedy bin packing keeps workers reasonably balanced while
    # staying deterministic for equal-sized groups.
    sized_groups = [
        (
            sum(_file_group_size(file_group) for file_group in file_groups),
            index,
            group_name,
            file_groups,
        )
        for index, (group_name, file_groups) in enumerate(work_groups)
    ]
    sized_groups.sort(key=lambda item: (-item[0], item[1]))

    for size_bytes, _index, group_name, file_groups in sized_groups:
        bucket_index = min(range(worker_count), key=lambda idx: bucket_bytes[idx])
        bucket_file_groups[bucket_index].extend(file_groups)
        bucket_names[bucket_index].append(group_name)
        bucket_bytes[bucket_index] += size_bytes

    buckets: List[NixlWorkGroup] = []
    for index, file_groups in enumerate(bucket_file_groups):
        if not file_groups:
            continue
        group_name = ",".join(bucket_names[index])
        buckets.append((group_name, file_groups))
    return buckets


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
        self._device = config.device
        self._max_workers = config.max_workers
        self._api_pool = ThreadPoolExecutor(max_workers=1)
        self._api_future = self._api_pool.submit(load_nixl_api)
        self._posix_backend_params = _posix_backend_params_from_config(
            config.backend_config
        )
        self._group_sources = group_sources
        self._group_kind = group_kind
        self._warn_under_parallelized = warn_under_parallelized
        logger.info(
            "%s configured for device %d with %d workers using NIXL POSIX "
            "staging backend_params=%s; NIXL import is running in the "
            "background and agent setup starts in start_restore() so it can "
            "overlap manifest planning, RW connect, and Phase A allocation",
            backend_name,
            self._device,
            self._max_workers,
            self._posix_backend_params,
        )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _NixlPosixStagingTransferSession(
            backend_name=self.name,
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
        api_future: Optional[Future[object]] = None,
    ) -> None:
        self._backend_name = backend_name
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._group_kind = group_kind
        self._posix_backend_params = dict(posix_backend_params)
        self._sources = list(sources)
        self._api_future = api_future
        self._agent_name_base = (
            f"gms_{backend_name.replace('-', '_')}_{device}_{os.getpid()}_{id(self):x}"
        )
        self._cancel_event = threading.Event()
        self._active = True
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
        self._work_groups = _split_work_groups(work_groups, self._worker_count)
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
            self._active = False
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
            assert self._prep_pool is not None
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
            self._active = False
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
        self._active = False
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
            cuda_utils.cuda_runtime_set_device(self._device)
            agent = create_nixl_agent(
                api,
                agent_name=agent_name,
                backend_name=NIXL_POSIX_BACKEND,
                backend_params=self._posix_backend_params,
            )
            if self._cancel_event.is_set():
                raise CancelledError(f"{self._backend_name} cancelled")
            slots = make_pinned_copy_slots(_PINNED_COPY_BUFFERS_PER_WORKER)
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
                worker_index=worker_index,
                group_name=group_name,
                file_groups=file_groups,
                agent=agent,
                agent_name=agent_name,
                slots=slots,
                prep_elapsed_s=prep_elapsed_s,
                prep_started_after_s=prep_started_after_s,
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
        cuda_utils.cuda_runtime_set_device(self._device)
        group_t0 = time.monotonic()
        group_bytes = 0
        try:
            group_bytes = restore_file_groups_with_nixl_staging(
                backend_name=self._backend_name,
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
    slots: Optional[List[PinnedCopySlot]] = None,
) -> int:
    owned_slots = slots is None
    if slots is None:
        slots = []
    total_bytes = 0
    next_slot = 0
    try:
        if owned_slots:
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
        if owned_slots:
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
