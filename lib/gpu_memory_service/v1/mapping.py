# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V1-local CUDA VMM mapping records and operations."""

from __future__ import annotations

import os
from dataclasses import dataclass

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.vmm import VMMDevice


@dataclass(frozen=True)
class LocalMapping:
    """One local VA reservation associated with opaque server backing."""

    allocation_id: str
    requested_size: int
    aligned_size: int
    base: int
    reservation_size: int


def install_mapping(
    vmm: VMMDevice,
    mapping: LocalMapping,
    fd: int,
    device: int,
    access: GrantedLockType,
) -> int:
    """Consume an export FD and import, map, and protect it."""
    handle = int(vmm.import_shareable_handle_close_fd(fd))
    mapped = False
    try:
        vmm.map(mapping.base, mapping.aligned_size, handle)
        mapped = True
        vmm.set_access(mapping.base, mapping.aligned_size, device, access)
    except Exception:
        try:
            if mapped:
                vmm.unmap(mapping.base, mapping.aligned_size)
        finally:
            vmm.release(handle)
        raise
    return handle


def unmap_mapping(vmm: VMMDevice, mapping: LocalMapping, handle: int) -> None:
    """Unmap one local handle while preserving its VA reservation."""
    vmm.unmap(mapping.base, mapping.aligned_size)
    vmm.release(handle)


def reserve_and_install_mapping(
    vmm: VMMDevice,
    fd: int,
    allocation_id: str,
    requested_size: int,
    aligned_size: int,
    reservation_size: int,
    reservation_alignment: int,
    device: int,
    access: GrantedLockType,
) -> tuple[LocalMapping, int]:
    """Reserve a VA and install an exported handle."""
    try:
        base = int(vmm.address_reserve(reservation_size, reservation_alignment))
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise

    mapping = LocalMapping(
        allocation_id,
        requested_size,
        aligned_size,
        base,
        reservation_size,
    )
    try:
        handle = install_mapping(vmm, mapping, fd, device, access)
    except Exception:
        vmm.address_free(mapping.base, mapping.reservation_size)
        raise
    return mapping, handle
