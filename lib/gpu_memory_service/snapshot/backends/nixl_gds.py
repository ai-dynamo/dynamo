# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL GDS restore backend for direct FILE -> VRAM transfers."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.snapshot.backends.nixl_common import (
    FILE_MEM_TYPE,
    NIXL_GDS_BACKEND,
    VRAM_MEM_TYPE,
    NixlTransferResources,
    create_nixl_agent,
    load_nixl_api,
    open_direct_read_fd,
    release_nixl_transfer_resources,
    release_transfer_resources,
    run_bounded_nixl_transfers,
    start_transfer,
    wait_for_transfer_done,
)
from gpu_memory_service.snapshot.transfer import (
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    StreamingRestoreCoordinator,
    TransferBackendKind,
    TransferSession,
    group_sources_by_path,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)


class NixlGDSTransferBackend:
    """NIXL GDS_MT backend for direct file-to-GMS GPU memory transfers."""

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        api = load_nixl_api()
        self._device = config.device
        self._max_workers = config.max_workers
        self._agent_name = f"gms_gds_{self._device}_{os.getpid()}"
        cuda_utils.cuda_runtime_set_device(self._device)
        self._agent = create_nixl_agent(
            api,
            agent_name=self._agent_name,
            backend_name=NIXL_GDS_BACKEND,
        )
        logger.info(
            "NIXL GDS_MT backend initialized for device %d with %d max in-flight transfers",
            self._device,
            self._max_workers,
        )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _NixlGDSTransferSession(
            agent=self._agent,
            agent_name=self._agent_name,
            device=self._device,
            max_workers=self._max_workers,
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
        max_workers: int,
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._agent = agent
        self._agent_name = agent_name
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._sources = list(sources)
        self._file_groups = list(group_sources_by_path(self._sources).items())
        self._total_bytes = sum(source.byte_count for source in self._sources)
        self._coordinator = StreamingRestoreCoordinator(
            backend_name="NIXL GDS",
            device=self._device,
            sources=self._sources,
        )
        self._pending_group_indices = set(range(len(self._file_groups)))
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stream_started_at: Optional[float] = None
        self._first_transfer_at: Optional[float] = None

    def submit_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        if not targets:
            return
        self._coordinator.submit_targets(targets)
        self._ensure_streaming_started()

    def finish_restore(self) -> None:
        if not self._file_groups:
            self._coordinator.finish_submission()
            return
        self._ensure_streaming_started()
        self._coordinator.finish_submission()

        assert self._scheduler_thread is not None
        self._scheduler_thread.join()
        self._coordinator.raise_if_failed()

        now = time.monotonic()
        first_transfer_at = self._first_transfer_at or now
        transfer_elapsed = now - first_transfer_at
        total_elapsed = (
            now - self._stream_started_at
            if self._stream_started_at is not None
            else transfer_elapsed
        )
        throughput = (
            self._total_bytes / transfer_elapsed / (1024**3)
            if transfer_elapsed > 0
            else 0.0
        )
        logger.info(
            "NIXL GDS transfers complete: %.2f GiB in %.3fs "
            "(%.2f GiB/s, files=%d, max_inflight=%d, streaming=True, "
            "total_stream_elapsed=%.3fs)",
            self._total_bytes / (1024**3),
            transfer_elapsed,
            throughput,
            len(self._file_groups),
            self._max_workers,
            total_elapsed,
        )

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._device)
        cuda_utils.cuda_runtime_set_device(self._device)
        t0 = time.monotonic()

        def prepare_transfer(
            file_group: Tuple[str, Sequence[FileTransferSource]],
        ) -> NixlTransferResources:
            return self._prepare_file_transfer(file_group, targets)

        run_bounded_nixl_transfers(
            agent=self._agent,
            backend_name=TransferBackendKind.NIXL_GDS.value,
            items=self._file_groups,
            max_inflight=self._max_workers,
            prepare_transfer=prepare_transfer,
            logger=logger,
        )

        elapsed = time.monotonic() - t0
        throughput = self._total_bytes / elapsed / (1024**3) if elapsed > 0 else 0
        logger.info(
            "NIXL GDS transfers complete: %.2f GiB in %.3fs "
            "(%.2f GiB/s, files=%d, max_inflight=%d)",
            self._total_bytes / (1024**3),
            elapsed,
            throughput,
            len(self._file_groups),
            self._max_workers,
        )

    def close(self) -> None:
        self._coordinator.cancel()
        if (
            self._scheduler_thread is not None
            and self._scheduler_thread.is_alive()
            and threading.current_thread() is not self._scheduler_thread
        ):
            self._scheduler_thread.join()

    def _ensure_streaming_started(self) -> None:
        if not self._file_groups:
            return
        if self._scheduler_thread is not None:
            return
        self._stream_started_at = time.monotonic()
        logger.info(
            "NIXL GDS streaming restore targets started: files=%d bytes=%.2f GiB "
            "max_inflight=%d",
            len(self._file_groups),
            self._total_bytes / (1024**3),
            self._max_workers,
        )
        self._scheduler_thread = threading.Thread(
            target=self._run_streaming_scheduler,
            name="nixl-gds-streaming-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _file_group_allocation_ids(
        self,
        file_group: Tuple[str, Sequence[FileTransferSource]],
    ) -> list[str]:
        _file_path, sources = file_group
        return [source.allocation_id for source in sources]

    def _pop_ready_group_locked(
        self,
    ) -> Optional[
        Tuple[Tuple[str, Sequence[FileTransferSource]], dict[str, GMSTransferTarget]]
    ]:
        ready_indices = [
            group_index
            for group_index in sorted(self._pending_group_indices)
            if self._coordinator.has_targets_locked(
                self._file_group_allocation_ids(self._file_groups[group_index])
            )
        ]
        if not ready_indices:
            return None
        group_index = ready_indices[0]
        file_group = self._file_groups[group_index]
        allocation_ids = self._file_group_allocation_ids(file_group)
        group_targets = self._coordinator.targets_for_locked(allocation_ids)
        self._pending_group_indices.remove(group_index)
        return file_group, group_targets

    def _run_streaming_scheduler(self) -> None:
        inflight: list[NixlTransferResources] = []
        try:
            cuda_utils.cuda_runtime_set_device(self._device)
            while True:
                self._coordinator.raise_if_failed()
                while len(inflight) < self._max_workers:
                    with self._coordinator.condition:
                        ready = self._pop_ready_group_locked()
                    if ready is None:
                        break
                    file_group, targets = ready
                    inflight.append(
                        self._start_file_group_transfer(file_group, targets)
                    )

                self._poll_completed_transfers(inflight)

                with self._coordinator.condition:
                    if not self._pending_group_indices and not inflight:
                        return
                    ready_exists = any(
                        self._coordinator.has_targets_locked(
                            self._file_group_allocation_ids(
                                self._file_groups[group_index]
                            )
                        )
                        for group_index in self._pending_group_indices
                    )
                    if ready_exists and len(inflight) < self._max_workers:
                        continue
                    if (
                        self._coordinator.submission_finished_locked
                        and self._pending_group_indices
                    ):
                        if not ready_exists:
                            missing = sorted(
                                {
                                    allocation_id
                                    for group_index in self._pending_group_indices
                                    for allocation_id in (
                                        self._coordinator.missing_targets_locked(
                                            self._file_group_allocation_ids(
                                                self._file_groups[group_index]
                                            )
                                        )
                                    )
                                }
                            )
                            raise RuntimeError(
                                f"NIXL GDS missing {len(missing)} restore "
                                "target(s) before finish_restore"
                            )
                    if self._coordinator.cancelled_locked:
                        raise RuntimeError("NIXL GDS restore session was cancelled")
                    self._coordinator.condition.wait(
                        timeout=0.001 if inflight else None
                    )
        except BaseException as exc:
            self._coordinator.set_error(exc)
        finally:
            self._drain_inflight_transfers(inflight)

    def _start_file_group_transfer(
        self,
        file_group: Tuple[str, Sequence[FileTransferSource]],
        targets: Mapping[str, GMSTransferTarget],
    ) -> NixlTransferResources:
        transfer = self._prepare_file_transfer(file_group, targets)
        try:
            start_transfer(
                self._agent,
                transfer.handle,
                transfer.label,
                TransferBackendKind.NIXL_GDS.value,
            )
            if self._first_transfer_at is None:
                self._first_transfer_at = time.monotonic()
                assert self._stream_started_at is not None
                logger.info(
                    "NIXL GDS first streaming transfer started after %.3fs "
                    "from first target submission",
                    self._first_transfer_at - self._stream_started_at,
                )
            return transfer
        except Exception:
            release_nixl_transfer_resources(self._agent, transfer)
            raise

    def _poll_completed_transfers(
        self,
        inflight: list[NixlTransferResources],
    ) -> None:
        still_running: list[NixlTransferResources] = []
        first_error: Optional[BaseException] = None
        for transfer in inflight:
            state = self._agent.check_xfer_state(transfer.handle)
            if state == "PROC":
                still_running.append(transfer)
                continue
            try:
                if state == "ERR":
                    raise RuntimeError(f"NIXL GDS transfer failed: {transfer.label}")
                if state != "DONE":
                    raise RuntimeError(
                        f"NIXL GDS transfer ended in unexpected state {state!r}: "
                        f"{transfer.label}"
                    )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            finally:
                release_nixl_transfer_resources(self._agent, transfer)
        inflight[:] = still_running
        if first_error is not None:
            raise first_error

    def _drain_inflight_transfers(
        self,
        inflight: list[NixlTransferResources],
    ) -> None:
        while inflight:
            transfer = inflight.pop(0)
            try:
                wait_for_transfer_done(
                    self._agent,
                    transfer.handle,
                    transfer.label,
                    TransferBackendKind.NIXL_GDS.value,
                )
            except Exception:
                logger.warning(
                    "NIXL GDS failed while draining in-flight transfer %s",
                    transfer.label,
                    exc_info=True,
                )
            finally:
                release_nixl_transfer_resources(self._agent, transfer)

    def _prepare_file_transfer(
        self,
        file_group: Tuple[str, Sequence[FileTransferSource]],
        targets: Mapping[str, GMSTransferTarget],
    ) -> NixlTransferResources:
        file_path, sources = file_group
        fd: Optional[int] = None
        file_reg = None
        vram_reg = None
        handle = None
        try:
            fd = open_direct_read_fd(
                file_path,
                logger=logger,
                require_direct=True,
            )
            file_descs = [
                (source.file_offset, source.byte_count, fd, "") for source in sources
            ]
            file_reg = self._agent.register_memory(file_descs, FILE_MEM_TYPE)

            vram_descs = [
                (
                    targets[source.allocation_id].va,
                    targets[source.allocation_id].byte_count,
                    targets[source.allocation_id].device,
                    "",
                )
                for source in sources
            ]
            vram_reg = self._agent.register_memory(vram_descs, VRAM_MEM_TYPE)

            handle = self._agent.initialize_xfer(
                "READ",
                vram_reg.trim(),
                file_reg.trim(),
                self._agent_name,
            )
            return NixlTransferResources(
                handle=handle,
                label=file_path,
                registrations=(file_reg, vram_reg),
                fds=(fd,),
            )
        except Exception:
            release_transfer_resources(self._agent, handle, file_reg, vram_reg, fd)
            raise
