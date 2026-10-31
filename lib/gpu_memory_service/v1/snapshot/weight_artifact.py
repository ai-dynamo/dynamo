# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-bearing save and load implementation for GMS V1 weight artifacts.

This module owns artifact validation, exact allocation-ID reconstruction,
device mappings, data transfer, cleanup, and publication to readers.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import monotonic
from types import TracebackType
from uuid import uuid4

import msgspec
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDevice, get_vmm
from gpu_memory_service.snapshot.disk import write_device_shards
from gpu_memory_service.snapshot.transfer import (
    FileIntegrity,
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferBackendKind,
    create_transfer_backend,
)
from gpu_memory_service.v1 import device as device_identity
from gpu_memory_service.v1.client.mapping import (
    LocalMapping,
    reserve_and_install_mapping,
)
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.protocol import (
    AllocationRecord,
    ListAllocationsRequest,
    ListAllocationsResponse,
)

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 2
_MANIFEST_NAME = "manifest.json"
_SHARDS_DIR = "shards"


class _Cleanup:
    """Run all callbacks without replacing an active operation failure."""

    def __init__(self) -> None:
        self._callbacks: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    def __enter__(self) -> _Cleanup:  # noqa: PYI034
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


class WeightArtifactAllocation(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    allocation_id: str
    aligned_size: int
    shard: str
    offset: int


class WeightArtifactShard(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    path: str
    byte_count: int
    blake3: str


class WeightArtifactManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    version: int
    allocations: tuple[WeightArtifactAllocation, ...]
    shards: tuple[WeightArtifactShard, ...]


class _ManifestHeader(msgspec.Struct):
    version: int


class _LegacyWeightArtifactManifest(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    version: int
    allocations: tuple[WeightArtifactAllocation, ...]


def save_weights(
    artifact_dir: str,
    socket_path: str,
    device: int,
    *,
    shard_size_bytes: int = 4 * 1024**3,
    max_workers: int = 8,
    connect_timeout: float | None = 30 * 60,
    admission_timeout: float | None = None,
    sharded_ssd_roots: Sequence[str] | None = None,
) -> WeightArtifactManifest:
    """Save committed V1 weight bytes with exact allocation IDs and sizes."""
    started_at = monotonic()
    if shard_size_bytes <= 0:
        raise ValueError("shard_size_bytes must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    roots = _normalize_roots(sharded_ssd_roots)
    artifact_path = Path(artifact_dir).expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        raise FileExistsError(artifact_path)
    attempt_id = uuid4().hex
    staging_path = artifact_path.parent / f".{artifact_path.name}.{attempt_id}.attempt"
    staging_path.mkdir()
    external_attempts: list[Path] = []

    try:
        for root in roots:
            external_attempts.append(
                _create_external_attempt(root, artifact_path.name, attempt_id)
            )
        vmm = get_vmm()
        vmm.ensure_initialized()
        vmm.runtime_set_device(device)
        granularity = int(vmm.get_allocation_granularity(device))
        with _Cleanup() as resources:
            session = _GMSClientSession(
                socket_path,
                RequestedLockType.RO,
                connect_timeout=connect_timeout,
                admission_timeout=admission_timeout,
            )
            resources.callback(session.close)
            _verify_session_device(session, device)
            records = _list_allocations(session)
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

            shard_roots = tuple(external_attempts) or (staging_path,)
            placements, shard_digests = write_device_shards(
                [
                    (mapping.base, record.aligned_size)
                    for record, (mapping, _handle) in zip(
                        records,
                        mappings,
                        strict=True,
                    )
                ],
                [str(root / _SHARDS_DIR) for root in shard_roots],
                device=device,
                shard_size_bytes=shard_size_bytes,
                max_workers=max_workers,
                relative_to=None if external_attempts else str(staging_path),
                blake3=True,
            )
            allocations = [
                WeightArtifactAllocation(
                    record.allocation_id,
                    record.aligned_size,
                    path,
                    offset,
                )
                for record, (path, offset) in zip(
                    records,
                    placements,
                    strict=True,
                )
            ]
            shard_sizes: dict[str, int] = defaultdict(int)
            for allocation in allocations:
                shard_sizes[allocation.shard] += allocation.aligned_size
            shards = tuple(
                WeightArtifactShard(path, shard_sizes[path], digest)
                for path, digest in sorted(shard_digests.items())
            )
            vmm.synchronize()

        manifest = WeightArtifactManifest(
            _MANIFEST_VERSION,
            tuple(allocations),
            shards,
        )
        _write_manifest(staging_path, manifest)
        staging_path.rename(artifact_path)
    except BaseException:
        _remove_attempt_paths((staging_path, *external_attempts))
        raise

    total_bytes = sum(record.aligned_size for record in records)
    logger.info(
        "GMS V1 saver device=%d allocations=%d bytes=%d elapsed=%.3fs",
        device,
        len(records),
        total_bytes,
        monotonic() - started_at,
    )
    return manifest


def load_weights(
    artifact_dir: str,
    socket_path: str,
    device: int,
    *,
    max_workers: int = 16,
    transfer_backend: str = TransferBackendKind.NIXL.value,
    backend_config: Mapping[str, object] | None = None,
) -> None:
    """Load exact V1 weight IDs into a fresh server and publish them.

    Returns without loading if the server already has committed weights.
    ``backend_config`` holds the transfer backend's settings, keyed as each
    backend reads them: ``posix_backend_params`` for the NIXL POSIX backends,
    ``sharded_ssd_queues_per_root`` for ``sharded-ssd``.
    ``sharded_ssd_roots`` also lists where the artifact may keep its shards.
    """
    started_at = monotonic()
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    backend_config = dict(backend_config or {})
    roots = _normalize_roots(backend_config.get("sharded_ssd_roots"))
    vmm = get_vmm()
    vmm.ensure_initialized()
    vmm.runtime_set_device(device)
    granularity = int(vmm.get_allocation_granularity(device))
    manifest, sources, integrity_by_path = _load_manifest(
        artifact_dir,
        granularity,
        roots,
    )
    backend = create_transfer_backend(
        transfer_backend,
        GMSSnapshotConfig(
            device=device,
            max_workers=max_workers,
            backend_config={**backend_config, "sharded_ssd_roots": list(roots)},
        ),
    )
    try:
        session = _GMSClientSession(socket_path, RequestedLockType.RW_OR_RO)
    except BaseException:
        try:
            backend.close()
        except BaseException:
            logger.exception(
                "GMS V1 backend cleanup failed while preserving a session error"
            )
        raise

    with _Cleanup() as resources:
        resources.callback(session.close)
        backend_holder = [backend]
        resources.callback(_close_backend, backend_holder)
        _verify_session_device(session, device)
        if session.lock_type is GrantedLockType.RO:
            logger.info("GMS V1 weights already committed on device=%d", device)
            return
        logger.info(
            "GMS V1 loader RW granted device=%d elapsed=%.3fs",
            device,
            monotonic() - started_at,
        )

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
            logger.info(
                "GMS V1 loader allocated device=%d allocations=%d elapsed=%.3fs",
                device,
                len(targets),
                monotonic() - started_at,
            )

            if set(targets) != {source.allocation_id for source in sources}:
                raise RuntimeError(
                    "GMS V1 artifact source and target allocation IDs do not match"
                )
            transfer = backend.start_restore(sources, integrity_by_path)
            with _Cleanup() as transfer_resources:
                transfer_resources.callback(transfer.close)
                transfer.restore(targets)
                vmm.synchronize()
            logger.info(
                "GMS V1 loader transferred device=%d elapsed=%.3fs",
                device,
                monotonic() - started_at,
            )

        _close_backend(backend_holder)
        session.commit()

    total_bytes = sum(record.aligned_size for record in expected_records)
    logger.info(
        "GMS V1 loader device=%d backend=%s allocations=%d bytes=%d elapsed=%.3fs",
        device,
        transfer_backend,
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
        resources.callback(
            vmm.address_free,
            local_mapping.base,
            local_mapping.reservation_size,
        )
        resources.callback(vmm.release, handle)
        resources.callback(vmm.unmap, local_mapping.base, local_mapping.aligned_size)


def _list_allocations(
    session: _GMSClientSession,
) -> tuple[AllocationRecord, ...]:
    response = session._call(ListAllocationsRequest(), ListAllocationsResponse)
    return response.allocations


def _load_manifest(
    artifact_dir: str,
    granularity: int,
    sharded_ssd_roots: Sequence[str] | None = None,
) -> tuple[
    WeightArtifactManifest | _LegacyWeightArtifactManifest,
    list[FileTransferSource],
    dict[str, FileIntegrity],
]:
    if granularity <= 0:
        raise ValueError("allocation granularity must be positive")
    root = Path(artifact_dir).expanduser().resolve(strict=True)
    allowed_roots = tuple(Path(path) for path in _normalize_roots(sharded_ssd_roots))
    manifest_bytes = (root / _MANIFEST_NAME).read_bytes()
    header = msgspec.json.decode(manifest_bytes, type=_ManifestHeader, strict=True)
    if header.version not in (1, _MANIFEST_VERSION):
        raise RuntimeError(
            f"unsupported GMS V1 weight artifact version {header.version}; "
            f"expected 1 or {_MANIFEST_VERSION}"
        )
    manifest_type = (
        _LegacyWeightArtifactManifest if header.version == 1 else WeightArtifactManifest
    )
    manifest = msgspec.json.decode(
        manifest_bytes,
        type=manifest_type,
        strict=True,
    )
    if not manifest.allocations:
        raise RuntimeError("GMS V1 weight artifact has no allocations")
    shards = manifest.shards if header.version != 1 else ()
    if header.version == _MANIFEST_VERSION and not shards:
        raise RuntimeError("GMS V1 weight artifact has no shard integrity records")

    shard_records: dict[str, WeightArtifactShard] = {}
    shard_integrity: dict[str, FileIntegrity] = {}
    for shard in shards:
        if not shard.path.strip():
            raise RuntimeError("GMS V1 weight artifact has an empty shard path")
        if shard.path in shard_records:
            raise RuntimeError(f"duplicate GMS V1 shard record {shard.path!r}")
        try:
            shard_integrity[shard.path] = FileIntegrity(shard.byte_count, shard.blake3)
        except ValueError as exc:
            raise RuntimeError(
                f"GMS V1 shard {shard.path!r} has invalid BLAKE3 or size: {exc}"
            ) from exc
        shard_records[shard.path] = shard

    allocation_ids: set[str] = set()
    extents: dict[Path, list[WeightArtifactAllocation]] = defaultdict(list)
    sources = []
    for allocation in manifest.allocations:
        if not allocation.allocation_id:
            raise RuntimeError("GMS V1 weight artifact has an empty allocation ID")
        if allocation.allocation_id in allocation_ids:
            raise RuntimeError(
                f"duplicate GMS V1 allocation ID {allocation.allocation_id!r}"
            )
        allocation_ids.add(allocation.allocation_id)
        if allocation.aligned_size <= 0 or allocation.aligned_size % granularity:
            raise RuntimeError(
                f"GMS V1 allocation {allocation.allocation_id!r} has invalid size"
            )
        if allocation.offset < 0 or allocation.offset % granularity:
            raise RuntimeError(
                f"GMS V1 allocation {allocation.allocation_id!r} has invalid offset"
            )
        shard_path = _resolve_shard_path(root, allocation.shard, allowed_roots)
        extents[shard_path].append(allocation)
        sources.append(
            FileTransferSource(
                allocation_id=allocation.allocation_id,
                file_path=str(shard_path),
                file_offset=allocation.offset,
                byte_count=allocation.aligned_size,
            )
        )
    _validate_file_extents(extents)
    if header.version == 1:
        logger.warning(
            "Loading legacy GMS weight artifact %s (schema version 1): "
            "content-digest integrity verification is unavailable; same-length weight "
            "file corruption cannot be detected. Recapture with the current "
            "saver to enable integrity verification.",
            root,
        )
        return manifest, sources, {}
    if set(shard_records) != {allocation.shard for allocation in manifest.allocations}:
        raise RuntimeError(
            "GMS V1 shard integrity records do not match allocation shards"
        )
    resolved_paths: set[Path] = set()
    integrity_by_path: dict[str, FileIntegrity] = {}
    for raw_path, shard in shard_records.items():
        resolved = _resolve_shard_path(root, raw_path, allowed_roots)
        if resolved in resolved_paths:
            raise RuntimeError(f"duplicate GMS V1 shard record resolves to {resolved}")
        resolved_paths.add(resolved)
        integrity_by_path[str(resolved)] = shard_integrity[raw_path]
        if resolved.stat().st_size != shard.byte_count:
            raise RuntimeError(f"GMS V1 shard length does not match record: {resolved}")
    if {source.allocation_id for source in sources} != allocation_ids:
        raise RuntimeError("GMS V1 artifact source IDs do not match allocations")
    return manifest, sources, integrity_by_path


def _resolve_shard_path(
    root: Path,
    shard: str,
    allowed_roots: Sequence[Path],
) -> Path:
    if not shard.strip():
        raise RuntimeError("GMS V1 weight artifact has an empty shard path")
    path = Path(shard)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise RuntimeError(f"GMS V1 shard must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"GMS V1 shard does not exist: {candidate}") from exc
    roots = allowed_roots if path.is_absolute() else (root,)
    if not any(resolved.is_relative_to(allowed_root) for allowed_root in roots):
        raise RuntimeError(f"GMS V1 shard escapes configured roots: {candidate}")
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise RuntimeError(f"cannot inspect GMS V1 shard: {candidate}") from exc
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"GMS V1 shard is not a regular file: {candidate}")
    return resolved


def _validate_file_extents(
    extents: Mapping[Path, list[WeightArtifactAllocation]],
) -> None:
    for path, allocations in extents.items():
        expected_offset = 0
        for allocation in sorted(allocations, key=lambda item: item.offset):
            if allocation.offset != expected_offset:
                raise RuntimeError(f"GMS V1 shard extents are not contiguous: {path}")
            expected_offset += allocation.aligned_size
        if path.stat().st_size != expected_offset:
            raise RuntimeError(f"GMS V1 shard length does not match extents: {path}")


def _normalize_roots(roots: Sequence[str] | None) -> tuple[str, ...]:
    if isinstance(roots, (str, os.PathLike)):
        roots = (os.fspath(roots),)
    normalized = []
    for root in roots or ():
        value = str(root).strip()
        if not value:
            continue
        path = str(Path(value).expanduser().resolve())
        if path not in normalized:
            normalized.append(path)
    return tuple(normalized)


def _verify_session_device(session: _GMSClientSession, device: int) -> None:
    device_identity.invalidate_device_uuid_cache()
    if session.identity[1] != device_identity.get_device_uuid(device):
        raise RuntimeError("GMS sidecar is on another physical GPU")


def _write_manifest(
    staging_path: Path,
    manifest: WeightArtifactManifest,
) -> None:
    temporary = staging_path / f".{_MANIFEST_NAME}.tmp"
    temporary.write_bytes(msgspec.json.encode(manifest))
    temporary.replace(staging_path / _MANIFEST_NAME)


def _create_external_attempt(root: str, artifact_name: str, attempt_id: str) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    attempt = root_path / f".{artifact_name}.{attempt_id}.attempt"
    attempt.mkdir()
    return attempt


def _remove_attempt_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("failed to remove GMS V1 artifact attempt %s", path)


def _close_backend(backend_holder: list[object]) -> None:
    if backend_holder:
        backend_holder.pop().close()


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
