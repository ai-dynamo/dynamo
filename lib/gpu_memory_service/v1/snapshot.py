# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-allocation cold storage for experimental GMS V1 weights."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import monotonic

import msgspec
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDevice, get_vmm
from gpu_memory_service.core.client.memory_manager import (
    LocalMapping,
    release_mapping,
    reserve_and_install_mapping,
    unmap_mapping,
)
from gpu_memory_service.core.client.session import _GMSClientSession
from gpu_memory_service.core.protocol import AllocationRecord
from gpu_memory_service.snapshot.disk import DeviceToFileWriter
from gpu_memory_service.snapshot.transfer import (
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferBackendKind,
    create_transfer_backend,
)

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_SHARDS_DIR = "shards"


class SnapshotAllocation(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    allocation_id: str
    aligned_size: int
    shard: str
    offset: int


class SnapshotManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    version: int
    allocations: tuple[SnapshotAllocation, ...]


def save_weights(
    artifact_dir: str,
    socket_path: str,
    device: int,
    *,
    shard_size_bytes: int = 4 * 1024**3,
) -> SnapshotManifest:
    """Save committed V1 weight allocation bytes and exact server IDs."""
    started_at = monotonic()
    if shard_size_bytes <= 0:
        raise ValueError("shard_size_bytes must be positive")

    vmm = get_vmm()
    vmm.ensure_initialized()
    vmm.runtime_set_device(device)
    granularity = int(vmm.get_allocation_granularity(device))
    session = _GMSClientSession(socket_path, RequestedLockType.RO)
    records = session.list_allocations()
    if not records:
        raise RuntimeError("GMS V1 weights server has no committed allocations")

    mappings = [
        _map_export(session, record, vmm, device, granularity, GrantedLockType.RO)
        for record in records
    ]
    artifact_path = Path(artifact_dir)
    shards_path = artifact_path / _SHARDS_DIR
    shards_path.mkdir(parents=True, exist_ok=True)
    allocations = _write_shards(
        records,
        mappings,
        shards_path,
        device,
        shard_size_bytes,
    )
    vmm.synchronize()
    _release_mappings(vmm, mappings)
    session.close()

    manifest = SnapshotManifest(_MANIFEST_VERSION, tuple(allocations))
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / _MANIFEST_NAME).write_bytes(msgspec.json.encode(manifest))
    total_bytes = sum(record.aligned_size for record in records)
    logger.info(
        "GMS V1 saver total device=%d allocations=%d bytes=%d elapsed=%.3fs",
        device,
        len(records),
        total_bytes,
        monotonic() - started_at,
    )
    return manifest


def hydrate_weights(
    artifact_dir: str,
    socket_path: str,
    device: int,
    *,
    max_workers: int = 16,
) -> _GMSClientSession:
    """Hydrate exact V1 weight IDs into a fresh server and publish them."""
    started_at = monotonic()
    manifest = _load_manifest(artifact_dir)
    if not manifest.allocations:
        raise RuntimeError("GMS V1 snapshot manifest has no weight allocations")
    sources = [
        FileTransferSource(
            allocation.allocation_id,
            os.path.join(artifact_dir, allocation.shard),
            allocation.offset,
            allocation.aligned_size,
        )
        for allocation in manifest.allocations
    ]
    vmm = get_vmm()
    vmm.ensure_initialized()
    vmm.runtime_set_device(device)
    granularity = int(vmm.get_allocation_granularity(device))
    backend = create_transfer_backend(
        TransferBackendKind.NIXL.value,
        GMSSnapshotConfig(device=device, max_workers=max_workers),
    )
    transfer = backend.start_restore(sources)

    session = _GMSClientSession(socket_path, RequestedLockType.RW)

    target_t0 = monotonic()
    mappings: list[tuple[LocalMapping, int]] = []
    targets: dict[str, GMSTransferTarget] = {}
    for allocation in manifest.allocations:
        record = AllocationRecord(
            allocation.allocation_id,
            allocation.aligned_size,
        )
        session.allocate(record.allocation_id, record.aligned_size)
        mapping = _map_export(
            session,
            record,
            vmm,
            device,
            granularity,
            GrantedLockType.RW,
        )
        mappings.append(mapping)
        targets[record.allocation_id] = GMSTransferTarget(
            record.allocation_id,
            mapping[0].base,
            device,
            record.aligned_size,
        )
    total_bytes = sum(allocation.aligned_size for allocation in manifest.allocations)
    logger.info(
        "GMS V1 loader target allocation device=%d allocations=%d bytes=%d "
        "elapsed=%.3fs",
        device,
        len(mappings),
        total_bytes,
        monotonic() - target_t0,
    )

    transfer_t0 = monotonic()
    transfer.restore(targets)
    vmm.synchronize()
    logger.info(
        "GMS V1 loader NIXL transfer device=%d allocations=%d bytes=%d elapsed=%.3fs",
        device,
        len(mappings),
        total_bytes,
        monotonic() - transfer_t0,
    )

    publish_t0 = monotonic()
    _release_mappings(vmm, mappings)
    transfer.close()
    backend.close()
    session.commit()
    logger.info(
        "GMS V1 loader commit/publish device=%d allocations=%d bytes=%d elapsed=%.3fs",
        device,
        len(mappings),
        total_bytes,
        monotonic() - publish_t0,
    )
    logger.info(
        "GMS V1 loader total device=%d allocations=%d bytes=%d elapsed=%.3fs",
        device,
        len(mappings),
        total_bytes,
        monotonic() - started_at,
    )
    return session


def _load_manifest(artifact_dir: str) -> SnapshotManifest:
    manifest = msgspec.json.decode(
        (Path(artifact_dir) / _MANIFEST_NAME).read_bytes(),
        type=SnapshotManifest,
        strict=True,
    )
    if manifest.version != _MANIFEST_VERSION:
        raise RuntimeError(
            f"unsupported GMS V1 snapshot manifest version {manifest.version}"
        )
    return manifest


def _map_export(
    session: _GMSClientSession,
    record: AllocationRecord,
    vmm: VMMDevice,
    device: int,
    granularity: int,
    access: GrantedLockType,
) -> tuple[LocalMapping, int]:
    return reserve_and_install_mapping(
        vmm,
        session.export(record.allocation_id),
        record.allocation_id,
        record.aligned_size,
        record.aligned_size,
        record.aligned_size,
        granularity,
        device,
        access,
    )


def _release_mappings(
    vmm: VMMDevice,
    mappings: list[tuple[LocalMapping, int]],
) -> None:
    for mapping, handle in reversed(mappings):
        unmap_mapping(vmm, mapping, handle)
        release_mapping(vmm, mapping)


def _write_shards(
    records: tuple[AllocationRecord, ...],
    mappings: list[tuple[LocalMapping, int]],
    shards_path: Path,
    device: int,
    shard_size_bytes: int,
) -> list[SnapshotAllocation]:
    groups: list[list[tuple[AllocationRecord, LocalMapping, int]]] = []
    current: list[tuple[AllocationRecord, LocalMapping, int]] = []
    current_size = 0
    for record, (mapping, _handle) in zip(records, mappings):
        if current and current_size + record.aligned_size > shard_size_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append((record, mapping, current_size))
        current_size += record.aligned_size
    groups.append(current)

    allocations: list[SnapshotAllocation] = []
    for shard_index, group in enumerate(groups):
        shard_name = f"shard_{shard_index:04d}.bin"
        with DeviceToFileWriter(str(shards_path / shard_name), device=device) as writer:
            for record, mapping, offset in group:
                writer.write_device(mapping.base, record.aligned_size)
                allocations.append(
                    SnapshotAllocation(
                        record.allocation_id,
                        record.aligned_size,
                        os.path.join(_SHARDS_DIR, shard_name),
                        offset,
                    )
                )
    return allocations
