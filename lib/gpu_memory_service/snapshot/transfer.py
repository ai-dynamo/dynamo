# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer backend contract and factory for GMS snapshot restore."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from gpu_memory_service.snapshot.model import AllocationEntry


class TransferBackendKind(str, Enum):
    NIXL = "nixl"
    NIXL_GDS = "nixl-gds"
    SHARDED_SSD = "sharded-ssd"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class FileTransferSource:
    """One source extent in a snapshot file."""

    allocation_id: str
    file_path: str
    file_offset: int
    byte_count: int


@dataclass(frozen=True)
class GMSTransferTarget:
    """One destination extent in GMS-owned GPU virtual memory."""

    allocation_id: str
    va: int
    device: int
    byte_count: int


@dataclass(frozen=True)
class GMSSnapshotConfig:
    """Restore settings split into common and backend-specific fields."""

    device: int
    max_workers: int
    backend_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", int(self.device))
        object.__setattr__(self, "max_workers", max(1, int(self.max_workers)))
        object.__setattr__(self, "backend_config", dict(self.backend_config or {}))


class TransferSession(Protocol):
    """Live restore operation for a set of transfer sources."""

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        """Move all source bytes into matching GMS targets."""

    def close(self) -> None:
        """Release resources and cancel any pending work."""


class TransferBackend(Protocol):
    """Backend capable of restoring bytes into GMS targets."""

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        """Start or stage restore work for the given sources."""

    def close(self) -> None:
        """Release backend-global resources."""


def build_file_transfer_sources(
    input_dir: str,
    allocations: Sequence[AllocationEntry],
) -> List[FileTransferSource]:
    """Convert manifest allocation placement into backend-neutral extents."""
    return [
        FileTransferSource(
            allocation_id=entry.allocation_id,
            file_path=os.path.join(input_dir, entry.tensor_file),
            file_offset=int(entry.tensor_offset),
            byte_count=int(entry.aligned_size),
        )
        for entry in allocations
    ]


def create_transfer_backend(
    name: str,
    config: GMSSnapshotConfig,
) -> TransferBackend:
    """Create the configured restore transfer backend."""
    if name == TransferBackendKind.NIXL.value:
        from gpu_memory_service.snapshot.backends.nixl import NixlTransferBackend

        return NixlTransferBackend(config=config)

    if name == TransferBackendKind.NIXL_GDS.value:
        from gpu_memory_service.snapshot.backends.nixl_gds import NixlGDSTransferBackend

        return NixlGDSTransferBackend(config=config)

    if name == TransferBackendKind.SHARDED_SSD.value:
        from gpu_memory_service.snapshot.backends.sharded_ssd import (
            ShardedSSDTransferBackend,
        )

        return ShardedSSDTransferBackend(config=config)

    choices = ", ".join(backend.value for backend in TransferBackendKind)
    raise ValueError(
        f"Unsupported GMS transfer backend {name!r}; expected one of {choices}"
    )


def validate_transfer_targets(
    sources: Sequence[FileTransferSource],
    targets: Mapping[str, GMSTransferTarget],
    *,
    device: Optional[int] = None,
) -> None:
    for source in sources:
        target = targets.get(source.allocation_id)
        if target is None:
            raise RuntimeError(
                f"Missing GMS transfer target for allocation {source.allocation_id}"
            )
        if target.byte_count != source.byte_count:
            raise RuntimeError(
                f"GMS target size mismatch for allocation {source.allocation_id}: "
                f"source={source.byte_count} target={target.byte_count}"
            )
        if device is not None and target.device != device:
            raise RuntimeError(
                f"GMS target device mismatch for allocation {source.allocation_id}: "
                f"backend={device} target={target.device}"
            )


def group_sources_by_path(
    sources: Sequence[FileTransferSource],
) -> Dict[str, List[FileTransferSource]]:
    groups: Dict[str, List[FileTransferSource]] = defaultdict(list)
    for source in sources:
        groups[source.file_path].append(source)
    for grouped in groups.values():
        grouped.sort(key=lambda source: source.file_offset)
    return dict(groups)
