# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer backend contract and factory for GMS snapshot restore."""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

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


@runtime_checkable
class StreamingTransferSession(TransferSession, Protocol):
    """Transfer session that accepts GMS targets as soon as they are allocated.

    Restore allocation order is still controlled by the caller.  Backends that
    implement this protocol may start transferring any internally complete work
    group before every allocation in the checkpoint has a mapped destination.
    """

    def submit_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        """Publish newly allocated restore targets to the backend."""

    def finish_restore(self) -> None:
        """Block until all submitted targets have been restored."""


class StreamingRestoreCoordinator:
    """Thread-safe target publication state for streaming restore sessions.

    Backends own their transfer scheduling policy, but they all need the same
    bookkeeping around target validation, duplicate detection, completion, and
    error/cancel propagation.  This helper centralizes that shared state while
    still letting each backend decide which ready group to run next.
    """

    def __init__(
        self,
        *,
        backend_name: str,
        device: int,
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._backend_name = backend_name
        self._device = device
        self._sources_by_id = {source.allocation_id: source for source in sources}
        self._condition = threading.Condition()
        self._targets: dict[str, GMSTransferTarget] = {}
        self._submission_finished = False
        self._cancelled = False
        self._error: Optional[BaseException] = None

    @property
    def condition(self) -> threading.Condition:
        return self._condition

    def submit_targets(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        if not targets:
            return
        self._validate_targets(targets)

        with self._condition:
            self.raise_if_failed_locked()
            if self._submission_finished or self._cancelled:
                raise RuntimeError(f"{self._backend_name} restore session is closed")
            for allocation_id, target in targets.items():
                previous = self._targets.get(allocation_id)
                if previous is not None and previous != target:
                    raise RuntimeError(
                        f"{self._backend_name} got duplicate target for "
                        f"allocation {allocation_id}"
                    )
                self._targets[allocation_id] = target
            self._condition.notify_all()

    def finish_submission(self) -> None:
        with self._condition:
            self._submission_finished = True
            self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._submission_finished = True
            self._condition.notify_all()

    def set_error(self, exc: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = exc
            self._condition.notify_all()

    def raise_if_failed(self) -> None:
        with self._condition:
            self.raise_if_failed_locked()

    def raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise self._error

    def wait_for_targets(
        self,
        allocation_ids: Sequence[str],
        *,
        missing_context: str,
        timeout: float = 0.01,
    ) -> dict[str, GMSTransferTarget]:
        with self._condition:
            while True:
                self.raise_if_failed_locked()
                if self._cancelled:
                    raise RuntimeError(
                        f"{self._backend_name} restore session was cancelled"
                    )
                missing = self.missing_targets_locked(allocation_ids)
                if not missing:
                    return self.targets_for_locked(allocation_ids)
                if self._submission_finished:
                    raise RuntimeError(
                        f"{self._backend_name} missing {len(missing)} restore "
                        f"target(s) for {missing_context}"
                    )
                self._condition.wait(timeout=timeout)

    def has_targets_locked(self, allocation_ids: Sequence[str]) -> bool:
        return all(allocation_id in self._targets for allocation_id in allocation_ids)

    def missing_targets_locked(self, allocation_ids: Sequence[str]) -> list[str]:
        return [
            allocation_id
            for allocation_id in allocation_ids
            if allocation_id not in self._targets
        ]

    def targets_for_locked(
        self,
        allocation_ids: Sequence[str],
    ) -> dict[str, GMSTransferTarget]:
        return {
            allocation_id: self._targets[allocation_id]
            for allocation_id in allocation_ids
        }

    @property
    def submission_finished_locked(self) -> bool:
        return self._submission_finished

    @property
    def cancelled_locked(self) -> bool:
        return self._cancelled

    def _validate_targets(
        self,
        targets: Mapping[str, GMSTransferTarget],
    ) -> None:
        for allocation_id, target in targets.items():
            source = self._sources_by_id.get(allocation_id)
            if source is None:
                raise RuntimeError(
                    f"{self._backend_name} got target for unknown allocation "
                    f"{allocation_id}"
                )
            if target.byte_count != source.byte_count:
                raise RuntimeError(
                    f"{self._backend_name} target size mismatch for allocation "
                    f"{allocation_id}: source={source.byte_count} "
                    f"target={target.byte_count}"
                )
            if target.device != self._device:
                raise RuntimeError(
                    f"{self._backend_name} target device mismatch for allocation "
                    f"{allocation_id}: backend={self._device} target={target.device}"
                )


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
