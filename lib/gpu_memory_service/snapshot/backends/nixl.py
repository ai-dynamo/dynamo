# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL transfer backends for GMS snapshot restore."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from gpu_memory_service.common import cuda_utils
from gpu_memory_service.snapshot.backends.pinned_host import (
    PINNED_COPY_CHUNK_SIZE,
    PinnedCopySlot,
    close_pinned_copy_slots,
    make_pinned_copy_slots,
)
from gpu_memory_service.snapshot.transfer import (
    NIXL_GDS_TRANSFER_BACKEND,
    NIXL_TRANSFER_BACKEND,
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferSession,
    group_sources_by_path,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)

_NIXL_POSIX_BACKEND = "POSIX"
_NIXL_GDS_BACKEND = "GDS_MT"
_DRAM_MEM_TYPE = "DRAM"
_PINNED_COPY_BUFFERS_PER_WORKER = 2


def _load_nixl_api() -> Tuple[Any, Any]:
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as exc:
        raise RuntimeError(
            "NIXL Python bindings are required for the nixl GMS transfer backends"
        ) from exc
    return nixl_agent, nixl_agent_config


def _start_transfer(agent: Any, handle: Any, path: str, backend_name: str) -> None:
    state = agent.transfer(handle)
    if state == "ERR":
        raise RuntimeError(f"{backend_name} transfer failed to start: {path}")
    if state not in {"PROC", "DONE"}:
        raise RuntimeError(
            f"{backend_name} transfer returned unexpected state {state!r}"
        )


def _wait_for_transfer(agent: Any, handle: Any, path: str, backend_name: str) -> None:
    _start_transfer(agent, handle, path, backend_name)
    _wait_for_transfer_done(agent, handle, path, backend_name)


def _wait_for_transfer_done(
    agent: Any,
    handle: Any,
    path: str,
    backend_name: str,
) -> None:
    state = agent.check_xfer_state(handle)
    while state == "PROC":
        time.sleep(0.001)
        state = agent.check_xfer_state(handle)
    if state == "ERR":
        raise RuntimeError(f"{backend_name} transfer failed: {path}")
    if state != "DONE":
        raise RuntimeError(
            f"{backend_name} transfer ended in unexpected state {state!r}: {path}"
        )


def _release_transfer_resources(
    agent: Any,
    handle: Any,
    first_reg: Any,
    second_reg: Any,
    fd: Optional[int] = None,
) -> None:
    if handle is not None:
        try:
            agent.release_xfer_handle(handle)
        except Exception:
            pass
    if first_reg is not None:
        try:
            agent.deregister_memory(first_reg)
        except Exception:
            pass
    if second_reg is not None:
        try:
            agent.deregister_memory(second_reg)
        except Exception:
            pass
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


class NixlTransferBackend:
    """NIXL POSIX backend for checkpoint shard restore without GDS."""

    name = NIXL_TRANSFER_BACKEND

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        nixl_agent, nixl_agent_config = _load_nixl_api()
        self._device = config.device
        self._max_workers = config.max_workers
        self._nixl_agent = nixl_agent
        self._nixl_agent_config = nixl_agent_config
        cuda_utils.cuda_runtime_set_device(self._device)
        logger.info(
            "NIXL POSIX backend initialized for device %d with %d workers",
            self._device,
            self._max_workers,
        )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return _NixlTransferSession(
            nixl_agent=self._nixl_agent,
            nixl_agent_config=self._nixl_agent_config,
            device=self._device,
            max_workers=self._max_workers,
            sources=sources,
        )

    def close(self) -> None:
        pass


class _NixlTransferSession:
    def __init__(
        self,
        *,
        nixl_agent: Any,
        nixl_agent_config: Any,
        device: int,
        max_workers: int,
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._nixl_agent = nixl_agent
        self._nixl_agent_config = nixl_agent_config
        self._device = device
        self._max_workers = max_workers
        self._sources = list(sources)
        self._agent_name_base = f"gms_nixl_{self._device}_{os.getpid()}_{id(self):x}"
        self._active = True

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._device)
        grouped_sources = group_sources_by_path(self._sources)
        if not grouped_sources:
            self._active = False
            return

        worker_count = min(self._max_workers, len(grouped_sources))
        total_bytes = sum(source.byte_count for source in self._sources)
        t0 = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        self._restore_file,
                        worker_index,
                        file_path,
                        sources,
                        targets,
                    ): file_path
                    for worker_index, (file_path, sources) in enumerate(
                        grouped_sources.items()
                    )
                }
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"{NIXL_TRANSFER_BACKEND} failed for {file_path}: {exc}"
                        ) from exc
        finally:
            self._active = False

        elapsed = time.monotonic() - t0
        throughput = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
        logger.info(
            "NIXL POSIX transfers complete: %.2f GiB in %.3fs (%.2f GiB/s)",
            total_bytes / (1024**3),
            elapsed,
            throughput,
        )

    def close(self) -> None:
        self._active = False

    def _restore_file(
        self,
        worker_index: int,
        file_path: str,
        sources: Sequence[FileTransferSource],
        targets: Mapping[str, GMSTransferTarget],
    ) -> None:
        cuda_utils.cuda_runtime_set_device(self._device)
        agent_name = f"{self._agent_name_base}_{worker_index}"
        agent = self._nixl_agent(
            agent_name,
            self._nixl_agent_config(backends=[]),
        )
        agent.create_backend(_NIXL_POSIX_BACKEND)
        slots: List[PinnedCopySlot] = []
        next_slot = 0
        fd = os.open(file_path, os.O_RDONLY)
        try:
            slots = make_pinned_copy_slots(_PINNED_COPY_BUFFERS_PER_WORKER)
            for source in sources:
                copied, next_slot = self._restore_source(
                    agent,
                    agent_name,
                    fd,
                    file_path,
                    source,
                    targets[source.allocation_id],
                    slots,
                    next_slot,
                )
                if copied != source.byte_count:
                    raise RuntimeError(
                        f"short {NIXL_TRANSFER_BACKEND} restore for "
                        f"{source.allocation_id}: {copied}/{source.byte_count}"
                    )
            for slot in slots:
                slot.wait()
        finally:
            os.close(fd)
            close_pinned_copy_slots(
                slots,
                logger,
                "failed to release NIXL pinned copy slot",
            )

    def _restore_source(
        self,
        agent: Any,
        agent_name: str,
        fd: int,
        file_path: str,
        source: FileTransferSource,
        target: GMSTransferTarget,
        slots: List[PinnedCopySlot],
        next_slot: int,
    ) -> Tuple[int, int]:
        done = 0
        while done < source.byte_count:
            slot = slots[next_slot]
            slot.wait()
            chunk_size = min(PINNED_COPY_CHUNK_SIZE, source.byte_count - done)
            self._read_file_to_host(
                agent,
                agent_name,
                fd,
                file_path,
                source.file_offset + done,
                slot.ptr,
                chunk_size,
            )
            slot.copy_to_device_async(target.va + done, chunk_size)
            done += chunk_size
            next_slot = (next_slot + 1) % len(slots)
        return done, next_slot

    def _read_file_to_host(
        self,
        agent: Any,
        agent_name: str,
        fd: int,
        file_path: str,
        file_offset: int,
        host_ptr: int,
        size: int,
    ) -> None:
        file_reg = None
        host_reg = None
        handle = None
        try:
            file_reg = agent.register_memory(
                [(file_offset, size, fd, "")],
                "FILE",
            )
            host_reg = agent.register_memory(
                [(host_ptr, size, 0, "")],
                _DRAM_MEM_TYPE,
            )
            handle = agent.initialize_xfer(
                "READ",
                host_reg.trim(),
                file_reg.trim(),
                agent_name,
            )
            _wait_for_transfer(agent, handle, file_path, NIXL_TRANSFER_BACKEND)
        finally:
            _release_transfer_resources(agent, handle, host_reg, file_reg)


class NixlGDSTransferBackend:
    """NIXL GDS_MT backend for direct file-to-GMS GPU memory transfers."""

    name = NIXL_GDS_TRANSFER_BACKEND

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        nixl_agent, nixl_agent_config = _load_nixl_api()
        self._device = config.device
        self._agent_name = f"gms_gds_{self._device}_{os.getpid()}"
        self._agent = nixl_agent(
            self._agent_name,
            nixl_agent_config(backends=[]),
        )
        self._agent.create_backend(_NIXL_GDS_BACKEND)
        logger.info("NIXL GDS_MT backend initialized for device %d", self._device)

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
        validate_transfer_targets(self._sources, targets, device=self._device)
        pending: List[Tuple[Any, Any, Any, Optional[int], str]] = []
        total_bytes = sum(source.byte_count for source in self._sources)
        t0 = time.monotonic()
        try:
            for file_path, sources in group_sources_by_path(self._sources).items():
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
                    _start_transfer(
                        self._agent,
                        handle,
                        file_path,
                        NIXL_GDS_TRANSFER_BACKEND,
                    )
                    pending.append((handle, file_reg, vram_reg, fd, file_path))
                except Exception:
                    _release_transfer_resources(
                        self._agent, handle, file_reg, vram_reg, fd
                    )
                    raise

            for handle, file_reg, vram_reg, fd, file_path in pending:
                _wait_for_transfer_done(
                    self._agent,
                    handle,
                    file_path,
                    NIXL_GDS_TRANSFER_BACKEND,
                )
        finally:
            for handle, file_reg, vram_reg, fd, _ in pending:
                _release_transfer_resources(
                    self._agent,
                    handle,
                    file_reg,
                    vram_reg,
                    fd,
                )
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
