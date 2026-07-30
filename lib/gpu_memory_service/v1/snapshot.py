# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-allocation cold storage for experimental GMS V1 weights."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import Callable

import msgspec
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDevice, get_vmm
from gpu_memory_service.core.client.memory_manager import (
    LocalMapping,
    release_mapping,
    reserve_and_install_mapping,
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
_COMMIT_VERIFICATION_TIMEOUT_SECONDS = 5.0


class _Cleanup:
    """Run all callbacks without replacing an active operation failure."""

    def __init__(self) -> None:
        self._callbacks: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    def __enter__(self) -> _Cleanup:
        return self

    def callback(self, callback: Callable[..., object], *args: object) -> None:
        self._callbacks.append((callback, args))

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        operation_error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        cleanup_error: BaseException | None = None
        cleanup_traceback: TracebackType | None = None
        for callback, args in reversed(self._callbacks):
            try:
                callback(*args)
            except BaseException as error:
                if operation_error is not None or cleanup_error is not None:
                    logger.exception(
                        "GMS V1 resource cleanup failed while preserving an "
                        "earlier error"
                    )
                else:
                    cleanup_error = error
                    cleanup_traceback = error.__traceback__
        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_traceback)
        return False


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

    setup_t0 = monotonic()
    vmm = get_vmm()
    vmm.ensure_initialized()
    vmm.runtime_set_device(device)
    granularity = int(vmm.get_allocation_granularity(device))
    with _Cleanup() as resources:
        session = _GMSClientSession(socket_path, RequestedLockType.RO)
        resources.callback(session.close)
        records = session.list_allocations()
        if not records:
            raise RuntimeError("GMS V1 weights server has no committed allocations")

        mappings: list[tuple[LocalMapping, int]] = []
        for record in records:
            mapping = _map_export(
                session,
                record,
                vmm,
                device,
                granularity,
                GrantedLockType.RO,
            )
            mappings.append(mapping)
            resources.callback(_release_mapping, vmm, mapping)
        total_bytes = sum(record.aligned_size for record in records)
        logger.info(
            "GMS V1 saver enumerate/map/import setup device=%d allocations=%d bytes=%d "
            "elapsed=%.3fs",
            device,
            len(records),
            total_bytes,
            monotonic() - setup_t0,
        )

        write_t0 = monotonic()
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
        logger.info(
            "GMS V1 saver device-to-file shard write device=%d allocations=%d bytes=%d "
            "elapsed=%.3fs",
            device,
            len(records),
            total_bytes,
            monotonic() - write_t0,
        )

        release_t0 = monotonic()
    logger.info(
        "GMS V1 saver release device=%d allocations=%d bytes=%d elapsed=%.3fs",
        device,
        len(records),
        total_bytes,
        monotonic() - release_t0,
    )

    manifest = SnapshotManifest(_MANIFEST_VERSION, tuple(allocations))
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / _MANIFEST_NAME).write_bytes(msgspec.json.encode(manifest))
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
) -> None:
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
    with _Cleanup() as resources:
        resources.callback(backend.close)
        session = _GMSClientSession(socket_path, RequestedLockType.RW)
        resources.callback(session.close)

        target_t0 = monotonic()
        mappings: list[tuple[LocalMapping, int]] = []
        targets: dict[str, GMSTransferTarget] = {}
        expected_records = tuple(
            AllocationRecord(allocation.allocation_id, allocation.aligned_size)
            for allocation in manifest.allocations
        )
        with _Cleanup() as mapping_resources:
            for record in expected_records:
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
                mapping_resources.callback(_release_mapping, vmm, mapping)
                targets[record.allocation_id] = GMSTransferTarget(
                    record.allocation_id,
                    mapping[0].base,
                    device,
                    record.aligned_size,
                )
            total_bytes = sum(record.aligned_size for record in expected_records)
            logger.info(
                "GMS V1 loader target allocation device=%d allocations=%d "
                "bytes=%d elapsed=%.3fs",
                device,
                len(mappings),
                total_bytes,
                monotonic() - target_t0,
            )

            transfer = backend.start_restore(sources)
            with _Cleanup() as transfer_resources:
                transfer_resources.callback(transfer.close)
                transfer_t0 = monotonic()
                transfer.restore(targets)
                vmm.synchronize()
                logger.info(
                    "GMS V1 loader NIXL transfer device=%d allocations=%d "
                    "bytes=%d elapsed=%.3fs",
                    device,
                    len(mappings),
                    total_bytes,
                    monotonic() - transfer_t0,
                )

        publish_t0 = monotonic()
        _commit_or_verify(session, socket_path, expected_records)
        logger.info(
            "GMS V1 loader commit/publish device=%d allocations=%d bytes=%d "
            "elapsed=%.3fs",
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


def _release_mapping(
    vmm: VMMDevice,
    mapping: tuple[LocalMapping, int],
) -> None:
    local_mapping, handle = mapping
    with _Cleanup() as resources:
        resources.callback(release_mapping, vmm, local_mapping)
        resources.callback(vmm.release, handle)
        resources.callback(vmm.unmap, local_mapping.base, local_mapping.aligned_size)


def _commit_or_verify(
    session: _GMSClientSession,
    socket_path: str,
    expected_records: tuple[AllocationRecord, ...],
) -> None:
    identity = session.identity
    try:
        session.commit()
        return
    except ConnectionError as commit_error:
        try:
            verifier = _GMSClientSession(
                socket_path,
                RequestedLockType.RO,
                expected_identity=identity,
                handshake_timeout=_COMMIT_VERIFICATION_TIMEOUT_SECONDS,
            )
            with _Cleanup() as resources:
                resources.callback(verifier.close)
                actual_records = verifier.list_allocations()
        except Exception as verification_error:
            raise RuntimeError(
                "GMS commit outcome is unknown and RO verification failed"
            ) from verification_error

        if actual_records != expected_records:
            raise RuntimeError(
                "GMS commit outcome is unknown: published allocations differ "
                f"(expected {expected_records!r}, found {actual_records!r})"
            ) from commit_error
        logger.warning(
            "GMS commit response was lost; verified %d published allocations",
            len(expected_records),
        )


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
