# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL GDS restore backend for direct FILE -> VRAM transfers."""

from __future__ import annotations

import logging
import os
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
    release_transfer_resources,
    run_bounded_nixl_transfers,
)
from gpu_memory_service.snapshot.transfer import (
    FileIntegrity,
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferBackendKind,
    TransferSession,
    group_sources_by_path,
    validate_transfer_targets,
    verify_file_source_digests,
)

logger = logging.getLogger(__name__)


class NixlGDSTransferBackend:
    """NIXL GDS_MT backend for direct file-to-GMS GPU memory transfers."""

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

    def start_restore(
        self,
        sources: Sequence[FileTransferSource],
        integrity_by_path: Mapping[str, FileIntegrity] | None = None,
    ) -> TransferSession:
        return _NixlGDSTransferSession(
            agent=self._agent,
            agent_name=self._agent_name,
            device=self._device,
            max_workers=self._max_workers,
            sources=sources,
            integrity_by_path=integrity_by_path,
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
        integrity_by_path: Mapping[str, FileIntegrity] | None = None,
    ) -> None:
        self._agent = agent
        self._agent_name = agent_name
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._sources = list(sources)
        self._integrity_by_path = dict(integrity_by_path or {})

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._device)
        file_groups = list(group_sources_by_path(self._sources).items())
        if file_groups:
            # cuFile rejects an unsupported file system only when a file is
            # registered. Check one shard before the integrity pass reads every
            # byte of the artifact.
            self._check_file_registration(*file_groups[0])
        verification_t0 = time.monotonic()
        verified_files, verified_bytes = verify_file_source_digests(
            self._sources,
            self._integrity_by_path,
            max_workers=self._max_workers,
        )
        if verified_files:
            verification_elapsed = time.monotonic() - verification_t0
            throughput = (
                verified_bytes / verification_elapsed / (1024**3)
                if verification_elapsed > 0
                else 0.0
            )
            logger.info(
                "NIXL GDS pre-transfer integrity verification succeeded: "
                "%.2f GiB in %.3fs (%.2f GiB/s, files=%d, workers=%d)",
                verified_bytes / (1024**3),
                verification_elapsed,
                throughput,
                verified_files,
                min(self._max_workers, verified_files),
            )
        total_bytes = sum(source.byte_count for source in self._sources)
        t0 = time.monotonic()

        def prepare_transfer(
            file_group: Tuple[str, Sequence[FileTransferSource]],
        ) -> NixlTransferResources:
            return self._prepare_file_transfer(file_group, targets)

        run_bounded_nixl_transfers(
            agent=self._agent,
            backend_name=TransferBackendKind.NIXL_GDS.value,
            items=file_groups,
            max_inflight=self._max_workers,
            prepare_transfer=prepare_transfer,
            logger=logger,
        )
        elapsed = time.monotonic() - t0
        throughput = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0
        logger.info(
            "NIXL GDS transfers complete: %.2f GiB in %.3fs "
            "(%.2f GiB/s, files=%d, max_inflight=%d)",
            total_bytes / (1024**3),
            elapsed,
            throughput,
            len(file_groups),
            self._max_workers,
        )

    def close(self) -> None:
        pass

    def _check_file_registration(
        self,
        file_path: str,
        sources: Sequence[FileTransferSource],
    ) -> None:
        fd = open_direct_read_fd(file_path, logger=logger, require_direct=True)
        try:
            self._agent.deregister_memory(self._register_file(file_path, fd, sources))
        finally:
            os.close(fd)

    def _register_file(
        self,
        file_path: str,
        fd: int,
        sources: Sequence[FileTransferSource],
    ) -> Any:
        file_descs = [
            (source.file_offset, source.byte_count, fd, "") for source in sources
        ]
        try:
            return self._agent.register_memory(file_descs, FILE_MEM_TYPE)
        except Exception as exc:
            raise RuntimeError(
                f"NIXL GDS_MT could not register {file_path} on its "
                f"{_file_system_type(file_path)!r} file system. cuFile cannot "
                "open files on some file systems, such as SMB/CIFS, or on a "
                "local ext4 or xfs disk when the container cannot read the "
                "host's /run/udev. Store the checkpoint on a GDS-supported "
                "file system such as NFS, or restore with the 'nixl' transfer "
                "backend."
            ) from exc

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
            file_reg = self._register_file(file_path, fd, sources)

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


def _file_system_type(path: str) -> str:
    """Return the mount type of the file system holding path, or 'unknown'."""
    dev = os.stat(path).st_dev
    device = f"{os.major(dev)}:{os.minor(dev)}"
    try:
        with open("/proc/self/mountinfo") as mountinfo:
            for line in mountinfo:
                fields = line.split()
                if fields[2] == device:
                    # Optional fields end at "-", which precedes the type.
                    return fields[fields.index("-") + 1]
    except OSError:
        pass
    return "unknown"
