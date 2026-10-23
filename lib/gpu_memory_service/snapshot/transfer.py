# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer backend contract and factory for GMS snapshot restore."""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from blake3 import blake3
from gpu_memory_service.snapshot.model import AllocationEntry

# Every GPU rank restores independently. Keep the aggregate native hashing
# concurrency bounded per rank, while still letting small restores use BLAKE3's
# internal parallelism. This avoids the `AUTO` setting multiplying all host
# cores by every rank and every in-flight shard.
_BLAKE3_THREADS_PER_RANK = 16
_BLAKE3_MAX_THREADS_PER_SHARD = 4


def blake3_threads_per_shard(worker_count: int) -> int:
    """Return a bounded native BLAKE3 thread count for each active shard."""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    return max(
        1,
        min(_BLAKE3_MAX_THREADS_PER_SHARD, _BLAKE3_THREADS_PER_RANK // worker_count),
    )


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
class FileIntegrity:
    """Expected size and BLAKE3 digest for one complete shard file."""

    byte_count: int
    blake3: str

    def __post_init__(self) -> None:
        if self.byte_count <= 0:
            raise ValueError("file integrity byte count must be positive")
        if len(self.blake3) != 64 or any(
            character not in "0123456789abcdef" for character in self.blake3
        ):
            raise ValueError("file integrity BLAKE3 digest must be lowercase hex")


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
        """Verify and move all source bytes into matching GMS targets."""

    def close(self) -> None:
        """Release resources and cancel any pending work."""


class TransferBackend(Protocol):
    """Backend capable of restoring bytes into GMS targets."""

    def start_restore(
        self,
        sources: Sequence[FileTransferSource],
        integrity_by_path: Mapping[str, FileIntegrity] | None = None,
    ) -> TransferSession:
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


def validate_file_integrity_extents(
    sources: Sequence[FileTransferSource],
    integrity: FileIntegrity,
) -> None:
    """Ensure source extents cover exactly the file described by its record."""
    if not sources:
        raise ValueError("file source group must not be empty")

    file_path = sources[0].file_path
    expected_offset = 0
    for source in sorted(sources, key=lambda item: item.file_offset):
        if source.file_path != file_path:
            raise RuntimeError("File source group contains multiple file paths")
        if source.file_offset != expected_offset:
            raise RuntimeError(
                f"File integrity extents are not contiguous for transfer source: "
                f"{file_path}"
            )
        expected_offset += source.byte_count
    if expected_offset != integrity.byte_count:
        raise RuntimeError(
            f"File integrity extent coverage does not match byte count: {file_path}"
        )


def verify_file_source_digests(
    sources: Sequence[FileTransferSource],
    integrity_by_path: Mapping[str, FileIntegrity] | None = None,
    *,
    max_workers: int,
) -> Tuple[int, int]:
    """Verify source files that cannot be checked inside their transfer path."""
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    records = integrity_by_path or {}
    pending: List[Tuple[str, FileIntegrity]] = []
    if set(records) - {source.file_path for source in sources}:
        raise RuntimeError("File integrity record has no matching transfer source")
    for file_path, grouped_sources in group_sources_by_path(sources).items():
        integrity = records.get(file_path)
        if integrity is not None:
            validate_file_integrity_extents(grouped_sources, integrity)
            pending.append((file_path, integrity))
    if not pending:
        return 0, 0

    worker_count = min(max_workers, len(pending))
    digest_threads = blake3_threads_per_shard(worker_count)

    def verify(item: Tuple[str, FileIntegrity]) -> int:
        file_path, integrity = item
        digest = blake3(max_threads=digest_threads)
        byte_count = 0
        with open(file_path, "rb") as source_file:
            while chunk := source_file.read(8 * 1024**2):
                digest.update(chunk)
                byte_count += len(chunk)
        if byte_count != integrity.byte_count:
            raise RuntimeError(
                f"GMS V1 shard length does not match record: {file_path}"
            )
        if digest.hexdigest() != integrity.blake3:
            raise RuntimeError(f"GMS V1 shard BLAKE3 mismatch: {file_path}")
        return byte_count

    verified_bytes = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(verify, item) for item in pending]
        try:
            for future in as_completed(futures):
                verified_bytes += future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return len(pending), verified_bytes
