# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL GDS restore backend for direct FILE -> VRAM transfers."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

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
    NIXL_GDS_TRANSFER_BACKEND,
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferSession,
    group_sources_by_path,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)


class NixlGDSTransferBackend:
    """NIXL GDS_MT backend for direct file-to-GMS GPU memory transfers."""

    name = NIXL_GDS_TRANSFER_BACKEND

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        api = load_nixl_api()
        self._device = config.device
        self._max_workers = config.max_workers
        self._agent_name = f"gms_gds_{self._device}_{os.getpid()}"
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
        self._sources_by_id = {source.allocation_id: source for source in self._sources}
        self._file_groups = list(group_sources_by_path(self._sources).items())
        self._total_bytes = sum(source.byte_count for source in self._sources)
        self._active = True
        self._condition = threading.Condition()
        self._targets: dict[str, GMSTransferTarget] = {}
        self._submitted_done = False
        self._pending_group_indices = set(range(len(self._file_groups)))
        self._scheduler_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._error: Optional[BaseException] = None
        self._stream_started_at: Optional[float] = None
        self._first_transfer_at: Optional[float] = None

    def submit_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        if not self._active:
            raise RuntimeError("NIXL GDS restore session is closed")
        if not targets:
            return
        self._validate_submitted_targets(targets)
        self._ensure_streaming_started()

        with self._condition:
            self._raise_error_locked()
            for allocation_id, target in targets.items():
                previous = self._targets.get(allocation_id)
                if previous is not None and previous != target:
                    raise RuntimeError(
                        f"NIXL GDS got duplicate target for allocation {allocation_id}"
                    )
                self._targets[allocation_id] = target
            self._condition.notify_all()

    def finish_restore(self) -> None:
        if not self._file_groups:
            self._active = False
            return
        self._ensure_streaming_started()
        with self._condition:
            self._submitted_done = True
            self._condition.notify_all()

        try:
            assert self._scheduler_thread is not None
            self._scheduler_thread.join()
            self._raise_error()
        finally:
            self._active = False

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
        t0 = time.monotonic()

        def prepare_transfer(
            file_group: Tuple[str, Sequence[FileTransferSource]],
        ) -> NixlTransferResources:
            return self._prepare_file_transfer(file_group, targets)

        try:
            run_bounded_nixl_transfers(
                agent=self._agent,
                backend_name=NIXL_GDS_TRANSFER_BACKEND,
                items=self._file_groups,
                max_inflight=self._max_workers,
                prepare_transfer=prepare_transfer,
                logger=logger,
            )
        finally:
            self._active = False

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
        self._cancel_event.set()
        with self._condition:
            self._submitted_done = True
            self._condition.notify_all()
        if (
            self._scheduler_thread is not None
            and self._scheduler_thread.is_alive()
            and threading.current_thread() is not self._scheduler_thread
        ):
            self._scheduler_thread.join()
        self._active = False

    def _validate_submitted_targets(
        self,
        targets: Mapping[str, GMSTransferTarget],
    ) -> None:
        for allocation_id, target in targets.items():
            source = self._sources_by_id.get(allocation_id)
            if source is None:
                raise RuntimeError(
                    f"NIXL GDS got target for unknown allocation {allocation_id}"
                )
            if target.byte_count != source.byte_count:
                raise RuntimeError(
                    f"NIXL GDS target size mismatch for allocation {allocation_id}: "
                    f"source={source.byte_count} target={target.byte_count}"
                )
            if target.device != self._device:
                raise RuntimeError(
                    f"NIXL GDS target device mismatch for allocation {allocation_id}: "
                    f"backend={self._device} target={target.device}"
                )

    def _ensure_streaming_started(self) -> None:
        if not self._file_groups:
            self._active = False
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

    def _missing_group_targets(
        self,
        file_group: Tuple[str, Sequence[FileTransferSource]],
    ) -> list[str]:
        _file_path, sources = file_group
        return [
            source.allocation_id
            for source in sources
            if source.allocation_id not in self._targets
        ]

    def _pop_ready_group_locked(
        self,
    ) -> Optional[
        Tuple[Tuple[str, Sequence[FileTransferSource]], dict[str, GMSTransferTarget]]
    ]:
        ready_indices = [
            group_index
            for group_index in sorted(self._pending_group_indices)
            if not self._missing_group_targets(self._file_groups[group_index])
        ]
        if not ready_indices:
            return None
        group_index = ready_indices[0]
        file_group = self._file_groups[group_index]
        _file_path, sources = file_group
        group_targets = {
            source.allocation_id: self._targets[source.allocation_id]
            for source in sources
        }
        self._pending_group_indices.remove(group_index)
        return file_group, group_targets

    def _run_streaming_scheduler(self) -> None:
        inflight: list[NixlTransferResources] = []
        try:
            while True:
                self._raise_error()
                while len(inflight) < self._max_workers:
                    with self._condition:
                        ready = self._pop_ready_group_locked()
                    if ready is None:
                        break
                    file_group, targets = ready
                    inflight.append(
                        self._start_file_group_transfer(file_group, targets)
                    )

                self._poll_completed_transfers(inflight)

                with self._condition:
                    if not self._pending_group_indices and not inflight:
                        return
                    ready_exists = any(
                        not self._missing_group_targets(self._file_groups[group_index])
                        for group_index in self._pending_group_indices
                    )
                    if ready_exists and len(inflight) < self._max_workers:
                        continue
                    if self._submitted_done and self._pending_group_indices:
                        if not ready_exists:
                            missing = sorted(
                                {
                                    allocation_id
                                    for group_index in self._pending_group_indices
                                    for allocation_id in self._missing_group_targets(
                                        self._file_groups[group_index]
                                    )
                                }
                            )
                            raise RuntimeError(
                                f"NIXL GDS missing {len(missing)} restore "
                                "target(s) before finish_restore"
                            )
                    if self._cancel_event.is_set():
                        raise RuntimeError("NIXL GDS restore session was cancelled")
                    self._condition.wait(timeout=0.001 if inflight else None)
        except BaseException as exc:
            with self._condition:
                if self._error is None:
                    self._error = exc
                self._condition.notify_all()
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
                NIXL_GDS_TRANSFER_BACKEND,
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
                    NIXL_GDS_TRANSFER_BACKEND,
                )
            except Exception:
                logger.warning(
                    "NIXL GDS failed while draining in-flight transfer %s",
                    transfer.label,
                    exc_info=True,
                )
            finally:
                release_nixl_transfer_resources(self._agent, transfer)

    def _raise_error(self) -> None:
        with self._condition:
            self._raise_error_locked()

    def _raise_error_locked(self) -> None:
        if self._error is not None:
            raise self._error

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
