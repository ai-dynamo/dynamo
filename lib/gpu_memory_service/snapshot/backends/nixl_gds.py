# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL GDS restore backend for direct FILE -> VRAM transfers."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Mapping, Optional, Sequence, Tuple

from gpu_memory_service.common.profiling import profile_log
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
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferBackendKind,
    TransferSession,
    group_sources_by_path,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)


class NixlGDSTransferBackend:
    """NIXL GDS_MT backend for direct file-to-GMS GPU memory transfers."""

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        total_t0 = time.monotonic()
        api_t0 = time.monotonic()
        api = load_nixl_api()
        api_elapsed = time.monotonic() - api_t0
        self._device = config.device
        self._max_workers = config.max_workers
        self._agent_name = f"gms_gds_{self._device}_{os.getpid()}"
        agent_t0 = time.monotonic()
        self._agent = create_nixl_agent(
            api,
            agent_name=self._agent_name,
            backend_name=NIXL_GDS_BACKEND,
        )
        agent_elapsed = time.monotonic() - agent_t0
        logger.info(
            "NIXL GDS_MT backend initialized for device %d with %d max in-flight transfers",
            self._device,
            self._max_workers,
        )
        profile_log(
            logger,
            "NIXL GDS backend setup: device=%d max_workers=%d "
            "load_api=%.6fs create_agent_backend=%.6fs total=%.6fs",
            self._device,
            self._max_workers,
            api_elapsed,
            agent_elapsed,
            time.monotonic() - total_t0,
        )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        t0 = time.monotonic()
        session = _NixlGDSTransferSession(
            agent=self._agent,
            agent_name=self._agent_name,
            device=self._device,
            max_workers=self._max_workers,
            sources=sources,
        )
        profile_log(
            logger,
            "NIXL GDS start_restore: sources=%d bytes=%.2f GiB elapsed=%.6fs",
            len(sources),
            sum(source.byte_count for source in sources) / (1024**3),
            time.monotonic() - t0,
        )
        return session

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
        init_t0 = time.monotonic()
        self._agent = agent
        self._agent_name = agent_name
        self._device = device
        self._max_workers = max(1, int(max_workers))
        self._sources = list(sources)
        profile_log(
            logger,
            "NIXL GDS session created: sources=%d bytes=%.2f GiB "
            "max_workers=%d elapsed=%.6fs",
            len(self._sources),
            sum(source.byte_count for source in self._sources) / (1024**3),
            self._max_workers,
            time.monotonic() - init_t0,
        )

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_t0 = time.monotonic()
        validate_transfer_targets(self._sources, targets, device=self._device)
        validate_elapsed = time.monotonic() - validate_t0

        group_t0 = time.monotonic()
        file_groups = list(group_sources_by_path(self._sources).items())
        group_elapsed = time.monotonic() - group_t0
        total_bytes = sum(source.byte_count for source in self._sources)
        t0 = time.monotonic()

        profile_log(
            logger,
            "NIXL GDS restore setup: targets=%d file_groups=%d bytes=%.2f GiB "
            "validate_targets=%.6fs group_sources=%.6fs",
            len(targets),
            len(file_groups),
            total_bytes / (1024**3),
            validate_elapsed,
            group_elapsed,
        )

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

    def _prepare_file_transfer(
        self,
        file_group: Tuple[str, Sequence[FileTransferSource]],
        targets: Mapping[str, GMSTransferTarget],
    ) -> NixlTransferResources:
        prep_t0 = time.monotonic()
        file_path, sources = file_group
        fd: Optional[int] = None
        file_reg = None
        vram_reg = None
        handle = None
        try:
            open_t0 = time.monotonic()
            fd = open_direct_read_fd(
                file_path,
                logger=logger,
                require_direct=True,
            )
            open_elapsed = time.monotonic() - open_t0
            file_descs = [
                (source.file_offset, source.byte_count, fd, "") for source in sources
            ]
            file_reg_t0 = time.monotonic()
            file_reg = self._agent.register_memory(file_descs, FILE_MEM_TYPE)
            file_reg_elapsed = time.monotonic() - file_reg_t0

            vram_descs = [
                (
                    targets[source.allocation_id].va,
                    targets[source.allocation_id].byte_count,
                    targets[source.allocation_id].device,
                    "",
                )
                for source in sources
            ]
            vram_reg_t0 = time.monotonic()
            vram_reg = self._agent.register_memory(vram_descs, VRAM_MEM_TYPE)
            vram_reg_elapsed = time.monotonic() - vram_reg_t0

            init_t0 = time.monotonic()
            handle = self._agent.initialize_xfer(
                "READ",
                vram_reg.trim(),
                file_reg.trim(),
                self._agent_name,
            )
            init_elapsed = time.monotonic() - init_t0
            profile_log(
                logger,
                "NIXL GDS prepared file transfer: file=%s sources=%d bytes=%.2f GiB "
                "open_file=%.6fs file_register=%.6fs vram_register=%.6fs "
                "initialize_xfer=%.6fs total=%.6fs",
                file_path,
                len(sources),
                sum(source.byte_count for source in sources) / (1024**3),
                open_elapsed,
                file_reg_elapsed,
                vram_reg_elapsed,
                init_elapsed,
                time.monotonic() - prep_t0,
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
