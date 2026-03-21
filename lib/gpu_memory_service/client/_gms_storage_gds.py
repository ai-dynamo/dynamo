# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL GDS storage backend for direct file-to-GPU transfers."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from gpu_memory_service.client._gms_storage_model import AllocationEntry

logger = logging.getLogger(__name__)

try:
    from nixl._api import nixl_agent, nixl_agent_config

    _NIXL_AVAILABLE = True
except ImportError:
    _NIXL_AVAILABLE = False
    nixl_agent = None  # type: ignore[assignment]
    nixl_agent_config = None  # type: ignore[assignment]


class NixlGDSStorageBackend:
    """NIXL-based GDS backend: reads shard files directly into GPU VAs.

    Uses NVIDIA NIXL library with the GDS backend to perform
    GPUDirect Storage transfers, bypassing CPU memory entirely.
    """

    def __init__(self, device: int) -> None:
        if not _NIXL_AVAILABLE:
            raise RuntimeError(
                "NIXL Python bindings are required for GDS support. "
                "Install with: pip install nixl"
            )
        self._device = device
        self._agent_name = f"gms_gds_{device}_{os.getpid()}"
        self._agent = nixl_agent(
            self._agent_name,
            nixl_agent_config(backends=[]),
        )
        self._agent.create_backend("GDS_MT")
        logger.info("NIXL GDS_MT backend initialized for device %d", device)

    def restore_shards(
        self,
        input_dir: str,
        groups: Dict[str, List[AllocationEntry]],
        va_map: Dict[str, int],
    ) -> None:
        """Read shard files directly into pre-allocated GPU VAs using GDS.

        Each GMS VA (from create_mapping) is registered individually with cuFile
        for proper GDS without compat mode.  All shard transfers are posted
        concurrently and then awaited — GDS_MT handles internal parallelism.

        Note: ensure CUFILE_ENV_PATH_JSON points to a cufile.json with
        max_device_pinned_mem_size_kb large enough for the total model size.

        Args:
            input_dir: Base directory containing shard files.
            groups: Mapping of relative shard path to sorted allocation entries.
            va_map: Mapping of allocation_id to GPU virtual address.
        """
        pending: List[Tuple] = []  # (handle, file_reg, vram_reg, fd)
        total_bytes = sum(
            entry.aligned_size for entries in groups.values() for entry in entries
        )

        try:
            t0 = time.monotonic()

            # Register all files and VAs, then post transfers concurrently
            for rel_path, entries in groups.items():
                abs_path = os.path.join(input_dir, rel_path)
                fd = None
                file_reg = None
                vram_reg = None
                handle = None
                try:
                    fd = os.open(abs_path, os.O_RDONLY)

                    # FILE descriptors: (offset, length, fd, meta)
                    file_descs = [
                        (entry.tensor_offset, entry.aligned_size, fd, "")
                        for entry in entries
                    ]
                    file_reg = self._agent.register_memory(file_descs, "FILE")

                    # VRAM descriptors: (addr, length, device_id, meta)
                    vram_descs = [
                        (
                            va_map[entry.allocation_id],
                            entry.aligned_size,
                            self._device,
                            "",
                        )
                        for entry in entries
                    ]
                    vram_reg = self._agent.register_memory(vram_descs, "VRAM")

                    # Post transfer: FILE -> VRAM
                    vram_xfer = vram_reg.trim()
                    file_xfer = file_reg.trim()
                    handle = self._agent.initialize_xfer(
                        "READ", vram_xfer, file_xfer, self._agent_name
                    )
                    state = self._agent.transfer(handle)
                    if state == "ERR":
                        raise RuntimeError(
                            f"GDS transfer failed to start for {rel_path}"
                        )

                    pending.append((handle, file_reg, vram_reg, fd))
                except Exception:
                    self._release_transfer_resources(handle, file_reg, vram_reg, fd)
                    raise

            # Wait for all transfers to complete
            for handle, file_reg, vram_reg, fd in pending:
                state = self._agent.check_xfer_state(handle)
                while state == "PROC":
                    state = self._agent.check_xfer_state(handle)
                if state == "ERR":
                    raise RuntimeError("GDS transfer failed")

            elapsed = time.monotonic() - t0
            throughput = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0
            logger.info(
                "GDS transfers complete: %.2f GiB in %.3fs (%.2f GiB/s)",
                total_bytes / (1024**3),
                elapsed,
                throughput,
            )

        finally:
            for handle, file_reg, vram_reg, fd in pending:
                self._release_transfer_resources(handle, file_reg, vram_reg, fd)

    def close(self) -> None:
        """Release NIXL agent resources."""
        self._agent = None

    def _release_transfer_resources(
        self,
        handle: object,
        file_reg: object,
        vram_reg: object,
        fd: int | None,
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
