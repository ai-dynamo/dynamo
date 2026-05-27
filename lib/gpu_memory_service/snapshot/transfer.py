# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer backend contract and factory for GMS snapshot restore."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from gpu_memory_service.snapshot.model import AllocationEntry

DEFAULT_TRANSFER_BACKEND = "nixl"
NIXL_TRANSFER_BACKEND = "nixl"
NIXL_GDS_TRANSFER_BACKEND = "nixl-gds"
NIXL_UCX_TRANSFER_BACKEND = "nixl-ucx"
SHARDED_SSD_TRANSFER_BACKEND = "sharded-ssd"

TRANSFER_BACKEND_CHOICES = (
    NIXL_TRANSFER_BACKEND,
    NIXL_GDS_TRANSFER_BACKEND,
    NIXL_UCX_TRANSFER_BACKEND,
    SHARDED_SSD_TRANSFER_BACKEND,
)

CHECKPOINT_DIR_TRANSFER_BACKENDS = (
    NIXL_TRANSFER_BACKEND,
    NIXL_GDS_TRANSFER_BACKEND,
    NIXL_UCX_TRANSFER_BACKEND,
    SHARDED_SSD_TRANSFER_BACKEND,
)


@dataclass(frozen=True)
class FileTransferSource:
    """One source extent in a snapshot file."""

    allocation_id: str
    file_path: str
    file_offset: int
    byte_count: int


@dataclass(frozen=True)
class RemoteTransferSource:
    """One source extent in a remote NIXL agent's memory."""

    allocation_id: str
    remote_agent: str
    va: int
    device: int
    byte_count: int


TransferSource = FileTransferSource | RemoteTransferSource


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


class StreamingTransferSession(TransferSession, Protocol):
    """Transfer session that accepts GMS targets as soon as they are allocated.

    Restore allocation order is still controlled by the caller.  Backends that
    implement this protocol may start transferring any internally complete work
    group before every allocation in the checkpoint has a destination VA.
    """

    def submit_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        """Publish newly allocated restore targets to the backend."""

    def finish_restore(self) -> None:
        """Block until all submitted targets have been restored."""


class TransferBackend(Protocol):
    """Backend capable of restoring bytes into GMS targets."""

    name: str

    def start_restore(self, sources: Sequence[TransferSource]) -> TransferSession:
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


def build_remote_transfer_sources(
    allocations: Sequence[AllocationEntry],
    peer_sources: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    remote_agent: str = "",
) -> List[RemoteTransferSource]:
    """Convert manifest allocation records into remote-memory transfer extents."""
    if not isinstance(peer_sources, Mapping):
        peer_sources = {str(source["allocation_id"]): source for source in peer_sources}

    sources: List[RemoteTransferSource] = []
    for entry in allocations:
        source = peer_sources.get(entry.allocation_id)
        if source is None:
            raise RuntimeError(
                f"Missing UCX source metadata for allocation {entry.allocation_id}"
            )

        byte_count = int(source.get("byte_count", entry.aligned_size))
        if byte_count != int(entry.aligned_size):
            raise RuntimeError(
                f"UCX source size mismatch for allocation {entry.allocation_id}: "
                f"manifest={entry.aligned_size} source={byte_count}"
            )

        va = source.get("va", source.get("addr"))
        if va is None:
            raise RuntimeError(
                f"UCX source metadata for allocation {entry.allocation_id} "
                "must include va or addr"
            )

        sources.append(
            RemoteTransferSource(
                allocation_id=entry.allocation_id,
                remote_agent=str(source.get("remote_agent") or remote_agent or ""),
                va=int(va),
                device=int(source.get("device", 0)),
                byte_count=byte_count,
            )
        )
    return sources


def create_transfer_backend(
    name: str,
    config: GMSSnapshotConfig,
) -> TransferBackend:
    """Create the configured restore transfer backend."""
    if name == NIXL_TRANSFER_BACKEND:
        from gpu_memory_service.snapshot.backends.nixl import NixlTransferBackend

        return NixlTransferBackend(config=config)

    if name == NIXL_GDS_TRANSFER_BACKEND:
        from gpu_memory_service.snapshot.backends.nixl_gds import NixlGDSTransferBackend

        return NixlGDSTransferBackend(config=config)

    if name == NIXL_UCX_TRANSFER_BACKEND:
        from gpu_memory_service.snapshot.backends.nixl_ucx import NixlUCXTransferBackend

        return NixlUCXTransferBackend(config=config)

    if name == SHARDED_SSD_TRANSFER_BACKEND:
        from gpu_memory_service.snapshot.backends.sharded_ssd import (
            ShardedSSDTransferBackend,
        )

        return ShardedSSDTransferBackend(config=config)

    raise ValueError(
        f"Unsupported GMS transfer backend {name!r}; "
        f"expected one of {', '.join(TRANSFER_BACKEND_CHOICES)}"
    )


def validate_transfer_targets(
    sources: Sequence[TransferSource],
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
