# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Epoch, metadata, and committed layout state for GMS."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from gpu_memory_service.common.types import GrantedLockType

from .allocations import AllocationInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetadataEntry:
    allocation_id: str
    offset_bytes: int
    value: bytes
    epoch_id: int


@dataclass
class Epoch:
    id: int
    metadata: dict[str, MetadataEntry] = field(default_factory=dict)


@dataclass
class EpochState:
    next_id: int = 1
    active_rw: Optional[Epoch] = None
    committed: Optional[Epoch] = None
    memory_layout_hash: str = ""


class GMSEpochManager:
    def __init__(self):
        self._epochs = EpochState()
        logger.info("GMSEpochManager initialized")

    @property
    def committed_epoch_id(self) -> Optional[int]:
        if self._epochs.committed is None:
            return None
        return self._epochs.committed.id

    @property
    def active_rw_epoch_id(self) -> Optional[int]:
        if self._epochs.active_rw is None:
            return None
        return self._epochs.active_rw.id

    @property
    def memory_layout_hash(self) -> str:
        return self._epochs.memory_layout_hash

    def _require_epoch(self, mode: GrantedLockType) -> Epoch:
        if mode == GrantedLockType.RW:
            if self._epochs.active_rw is None:
                raise AssertionError("RW epoch is not active")
            return self._epochs.active_rw
        if self._epochs.committed is None:
            raise AssertionError("Committed epoch is not available")
        return self._epochs.committed

    def require_epoch_id(self, mode: GrantedLockType) -> int:
        return self._require_epoch(mode).id

    def _validate_metadata_target(
        self,
        epoch: Epoch,
        allocation: AllocationInfo,
        offset_bytes: int,
    ) -> None:
        if allocation.epoch_id != epoch.id:
            raise ValueError(
                f"Allocation {allocation.allocation_id} is not in active epoch {epoch.id}"
            )

        if offset_bytes < 0:
            raise ValueError(f"offset_bytes must be >= 0, got {offset_bytes}")
        if offset_bytes >= allocation.aligned_size:
            raise ValueError(
                f"offset_bytes {offset_bytes} out of range for allocation {allocation.allocation_id} "
                f"(aligned_size={allocation.aligned_size})"
            )

    def drop_metadata_for_allocation(self, allocation_id: str) -> int:
        epoch = self._require_epoch(GrantedLockType.RW)
        keys_to_remove = [
            key
            for key, entry in epoch.metadata.items()
            if entry.allocation_id == allocation_id
        ]
        for key in keys_to_remove:
            epoch.metadata.pop(key, None)
        return len(keys_to_remove)

    def _validate_epoch_integrity(
        self,
        epoch: Epoch,
        allocations_by_id: dict[str, AllocationInfo],
    ) -> None:
        for key, entry in epoch.metadata.items():
            info = allocations_by_id.get(entry.allocation_id)
            if info is None:
                raise AssertionError(
                    f"Metadata key {key!r} references missing allocation "
                    f"{entry.allocation_id!r} in epoch {epoch.id}"
                )

            if entry.offset_bytes < 0 or entry.offset_bytes >= info.aligned_size:
                raise AssertionError(
                    f"Metadata key {key!r} has invalid offset {entry.offset_bytes} "
                    f"for allocation {entry.allocation_id!r} "
                    f"(aligned_size={info.aligned_size})"
                )

    def _compute_memory_layout_hash(
        self,
        epoch: Epoch,
        allocations: list[AllocationInfo],
    ) -> str:
        h = hashlib.sha256()
        for info in sorted(allocations, key=lambda info: info.allocation_id):
            h.update(
                f"{info.allocation_id}:{info.size}:{info.aligned_size}:{info.tag}:{info.epoch_id}".encode()
            )

        for key in sorted(epoch.metadata):
            entry = epoch.metadata[key]
            h.update(
                f"{key}:{entry.allocation_id}:{entry.offset_bytes}:{entry.epoch_id}:".encode()
            )
            h.update(entry.value)
        return h.hexdigest()

    def on_rw_connect(self) -> int | None:
        if self._epochs.active_rw is not None:
            raise AssertionError("RW epoch is already active")

        old_epoch_id = None
        if self._epochs.committed is not None:
            old_epoch = self._epochs.committed
            self._epochs.committed = None
            self._epochs.memory_layout_hash = ""
            old_epoch_id = old_epoch.id
            logger.info("RW connected; invalidated committed epoch %d", old_epoch.id)

        epoch = Epoch(id=self._epochs.next_id)
        self._epochs.next_id += 1
        self._epochs.active_rw = epoch
        logger.info("RW connected; opened active epoch %d", epoch.id)
        return old_epoch_id

    def on_rw_abort(self) -> int | None:
        epoch = self._epochs.active_rw
        if epoch is None:
            return None

        logger.warning("RW aborted; clearing active epoch %d", epoch.id)
        self._epochs.active_rw = None
        if self._epochs.committed is None:
            self._epochs.memory_layout_hash = ""
        return epoch.id

    def on_commit(self, allocations: list[AllocationInfo]) -> int | None:
        epoch = self._require_epoch(GrantedLockType.RW)
        allocations_by_id = {
            info.allocation_id: info
            for info in allocations
            if info.epoch_id == epoch.id
        }
        self._validate_epoch_integrity(epoch, allocations_by_id)
        self._epochs.memory_layout_hash = self._compute_memory_layout_hash(
            epoch, allocations
        )

        old_committed = self._epochs.committed
        self._epochs.committed = epoch
        self._epochs.active_rw = None

        logger.info(
            "Committed epoch %d with state hash: %s...",
            epoch.id,
            self._epochs.memory_layout_hash[:16],
        )
        if old_committed is None or old_committed.id == epoch.id:
            return None
        return old_committed.id

    def put_metadata(
        self,
        allocation: AllocationInfo,
        key: str,
        offset_bytes: int,
        value: bytes,
    ) -> None:
        epoch = self._require_epoch(GrantedLockType.RW)
        self._validate_metadata_target(epoch, allocation, offset_bytes)
        epoch.metadata[key] = MetadataEntry(
            allocation_id=allocation.allocation_id,
            offset_bytes=offset_bytes,
            value=value,
            epoch_id=epoch.id,
        )

    def get_metadata(
        self,
        mode: GrantedLockType,
        key: str,
    ) -> Optional[MetadataEntry]:
        return self._require_epoch(mode).metadata.get(key)

    def delete_metadata(self, key: str) -> bool:
        return (
            self._require_epoch(GrantedLockType.RW).metadata.pop(key, None) is not None
        )

    def list_metadata(
        self,
        mode: GrantedLockType,
        prefix: str,
    ) -> list[str]:
        metadata = self._require_epoch(mode).metadata
        if not prefix:
            return sorted(metadata)
        return sorted(key for key in metadata if key.startswith(prefix))
