# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS storage client: save GMS state to disk and load it back."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.protocol.messages import GetAllocationResponse
from gpu_memory_service.common.snapshot_profile import SnapshotProfile
from gpu_memory_service.snapshot.disk import (
    DeviceToFileWriter,
    load_manifest_and_metadata,
    plan_shard_layout,
)
from gpu_memory_service.snapshot.model import AllocationEntry, SaveManifest
from gpu_memory_service.snapshot.transfer import (
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferBackendKind,
    build_file_transfer_sources,
    create_transfer_backend,
)

logger = logging.getLogger(__name__)


class GMSStorageClient:
    """Dump and restore GMS state to/from disk."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        socket_path: Optional[str] = None,
        device: int = 0,
        *,
        timeout_ms: Optional[int] = None,
        shard_size_bytes: int = 4 * 1024**3,
        transfer_backend: str = TransferBackendKind.NIXL.value,
        sharded_ssd_roots: Optional[Sequence[str]] = None,
        sharded_ssd_queues_per_root: int = 2,
        posix_backend_params: Optional[Mapping[str, str]] = None,
        profile: SnapshotProfile | None = None,
        cuda_operations: Any = None,
    ) -> None:
        self.output_dir = output_dir
        self.device = device
        self._timeout_ms = timeout_ms
        self._shard_size = shard_size_bytes
        self._transfer_backend = transfer_backend
        self._sharded_ssd_roots = (
            None
            if sharded_ssd_roots is None
            else [
                os.path.abspath(root) for root in sharded_ssd_roots if str(root).strip()
            ]
        )
        self._profile = profile or SnapshotProfile(
            "loader",
            enabled=False,
            device=device,
        )
        self._cuda_operations = cuda_operations
        self._sharded_ssd_queues_per_root = int(sharded_ssd_queues_per_root)
        if self._sharded_ssd_queues_per_root <= 0:
            raise ValueError("sharded_ssd_queues_per_root must be positive")
        self._posix_backend_params = (
            None
            if posix_backend_params is None
            else {str(key): str(value) for key, value in posix_backend_params.items()}
        )

        if socket_path is None:
            from gpu_memory_service.common.utils import get_socket_path

            socket_path = get_socket_path(device)
        self._socket_path = socket_path

    def save(self, max_workers: int = 4) -> SaveManifest:
        """Connect to GMS in RO mode and save all allocations + metadata to disk."""
        if self.output_dir is None:
            raise ValueError(
                "output_dir must be set to call save(); pass it to GMSStorageClient()"
            )
        with self._profile.phase("checkpoint_output_preparation"):
            (
                output_dir,
                shard_dirs,
                use_absolute_shard_paths,
            ) = self._prepare_output_dir()

        with self._profile.phase("client_memory_manager_construction"):
            mm = GMSClientMemoryManager(
                self._socket_path,
                device=self.device,
                profile=self._profile,
            )
        try:
            mm.connect(RequestedLockType.RO, timeout_ms=self._timeout_ms)
            with self._profile.phase("allocation_enumeration"):
                layout_hash = mm.get_memory_layout_hash()
                if not layout_hash:
                    raise RuntimeError(
                        "GMS server has no committed weights; nothing to dump"
                    )
                allocations_info = mm.list_handles()
            with self._profile.phase(
                "source_mapping_preparation",
                count=len(allocations_info),
                byte_count=sum(
                    int(allocation.aligned_size) for allocation in allocations_info
                ),
            ):
                va_list = [
                    mm.create_mapping(allocation_id=alloc.allocation_id)
                    for alloc in allocations_info
                ]
            logger.info("Imported %d source allocation VAs", len(va_list))
            with self._profile.phase(
                "gpu_host_file_transfer",
                count=len(allocations_info),
                byte_count=sum(
                    int(allocation.aligned_size) for allocation in allocations_info
                ),
            ):
                entries = self._write_shards(
                    shard_dirs,
                    allocations_info,
                    va_list,
                    max_workers=max_workers,
                    use_absolute_shard_paths=use_absolute_shard_paths,
                )
            with self._profile.phase("metadata_enumeration"):
                metadata = self._save_metadata(mm)
            metadata_path = os.path.join(output_dir, "gms_metadata.json")
            if self._profile.enabled:
                with self._profile.phase(
                    "metadata_serialization",
                    count=len(metadata),
                ):
                    metadata_text = json.dumps(metadata, indent=2)
                with self._profile.phase(
                    "metadata_write",
                    count=len(metadata),
                    byte_count=len(metadata_text.encode()),
                ):
                    with open(metadata_path, "w", encoding="utf-8") as handle:
                        handle.write(metadata_text)
            else:
                with open(metadata_path, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2)
            with self._profile.phase("manifest_build", count=len(entries)):
                manifest = SaveManifest(
                    timestamp=time.time(),
                    layout_hash=layout_hash,
                    device=self.device,
                    allocations=entries,
                )
            manifest_path = os.path.join(output_dir, "manifest.json")
            if self._profile.enabled:
                with self._profile.phase(
                    "manifest_serialization",
                    count=len(entries),
                ):
                    manifest_text = json.dumps(asdict(manifest), indent=2)
                with self._profile.phase(
                    "manifest_write",
                    count=len(entries),
                    byte_count=len(manifest_text.encode()),
                ):
                    with open(manifest_path, "w", encoding="utf-8") as handle:
                        handle.write(manifest_text)
            else:
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(asdict(manifest), handle, indent=2)
            logger.info("Wrote manifest with %d allocations", len(entries))
            return manifest
        finally:
            try:
                with self._profile.phase("memory_manager_cleanup"):
                    mm.close()
            finally:
                self._profile.emit_aggregates()

    def _prepare_output_dir(self) -> Tuple[str, list[str], bool]:
        assert self.output_dir is not None
        os.makedirs(self.output_dir, exist_ok=True)
        shard_roots = self._sharded_ssd_roots or [self.output_dir]
        shard_dirs = [os.path.join(root, "shards") for root in shard_roots]
        for shards_dir in shard_dirs:
            os.makedirs(shards_dir, exist_ok=True)
            for name in os.listdir(shards_dir):
                if name.startswith("shard_") and name.endswith(".bin"):
                    os.unlink(os.path.join(shards_dir, name))
        return self.output_dir, shard_dirs, bool(self._sharded_ssd_roots)

    def _write_shards(
        self,
        shard_dirs: list[str],
        allocations_info: Sequence[GetAllocationResponse],
        va_list: Sequence[int],
        *,
        max_workers: int,
        use_absolute_shard_paths: bool = False,
    ) -> list[AllocationEntry]:
        with self._profile.phase(
            "save_shard_layout_planning",
            count=len(allocations_info),
        ):
            layout = plan_shard_layout(allocations_info, self._shard_size)
            shard_groups: Dict[int, list[Tuple[int, int]]] = defaultdict(list)
            for index, (shard_idx, byte_offset) in enumerate(layout):
                shard_groups[shard_idx].append((index, byte_offset))

        entries: list[Optional[AllocationEntry]] = [None] * len(allocations_info)

        def _write_one_shard(
            shard_idx: int, alloc_pairs: list[Tuple[int, int]]
        ) -> None:
            filename = f"shard_{shard_idx:04d}.bin"
            shards_dir = shard_dirs[shard_idx % len(shard_dirs)]
            abs_path = os.path.join(shards_dir, filename)
            rel_path = os.path.join("shards", filename)
            tensor_file = abs_path if use_absolute_shard_paths else rel_path
            with DeviceToFileWriter(
                abs_path,
                device=self.device,
                profile=self._profile,
            ) as writer:
                for index, byte_offset in alloc_pairs:
                    alloc = allocations_info[index]
                    writer.write_device(va_list[index], alloc.aligned_size)
                    entries[index] = AllocationEntry(
                        allocation_id=alloc.allocation_id,
                        size=alloc.size,
                        aligned_size=alloc.aligned_size,
                        tag=alloc.tag,
                        tensor_file=tensor_file,
                        tensor_offset=byte_offset,
                    )

        with self._profile.phase(
            "save_worker_barrier",
            count=len(shard_groups),
        ):
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_write_one_shard, shard_idx, alloc_pairs): shard_idx
                    for shard_idx, alloc_pairs in shard_groups.items()
                }
                for future in as_completed(futures):
                    future.result()

        missing = sum(1 for entry in entries if entry is None)
        if missing:
            raise RuntimeError(
                f"BUG: {missing} allocation(s) missing after shard writers completed"
            )
        logger.info(
            "Wrote %d snapshot shard(s) across %d shard root(s)",
            len(shard_groups),
            len(shard_dirs),
        )
        return [entry for entry in entries if entry is not None]

    def _allocate_restore_targets(
        self,
        mm: Any,
        manifest: SaveManifest,
    ) -> Tuple[Dict[str, str], Dict[str, GMSTransferTarget]]:
        t0 = time.monotonic()
        id_map: Dict[str, str] = {}
        targets: Dict[str, GMSTransferTarget] = {}
        for entry in manifest.allocations:
            old_id = entry.allocation_id
            with self._profile.aggregate(
                "restore_target_mapping_total",
                byte_count=entry.aligned_size,
            ):
                va = mm.create_mapping(size=entry.size, tag=entry.tag)
                id_map[old_id] = mm.mappings[va].allocation_id
                targets[old_id] = GMSTransferTarget(
                    allocation_id=old_id,
                    va=va,
                    device=self.device,
                    byte_count=entry.aligned_size,
                )
        logger.info(
            "Allocated %d restore target GMS VAs in %.3fs",
            len(targets),
            time.monotonic() - t0,
        )
        return id_map, targets

    def load_to_gms(
        self,
        input_dir: str,
        *,
        max_workers: int = 4,
        clear_existing: bool = True,
        transfer_backend: Optional[str] = None,
    ) -> Dict[str, str]:
        backend_name = transfer_backend or self._transfer_backend

        with self._profile.phase(
            "transfer_backend_construction",
            backend=backend_name,
        ):
            backend = create_transfer_backend(
                backend_name,
                GMSSnapshotConfig(
                    device=self.device,
                    max_workers=max_workers,
                    backend_config={
                        "sharded_ssd_roots": self._sharded_ssd_roots,
                        "sharded_ssd_queues_per_root": (
                            self._sharded_ssd_queues_per_root
                        ),
                        "posix_backend_params": self._posix_backend_params,
                        "profile": self._profile,
                        "cuda_operations": self._cuda_operations,
                    },
                ),
            )
        session = None
        id_map: Dict[str, str] = {}

        try:
            manifest, saved_metadata = load_manifest_and_metadata(
                input_dir,
                profile=self._profile,
            )
            with self._profile.phase(
                "restore_source_discovery",
                count=len(manifest.allocations),
            ):
                sources = build_file_transfer_sources(
                    input_dir,
                    manifest.allocations,
                )
            with self._profile.phase(
                "backend_restore_session_planning",
                count=len(sources),
                byte_count=sum(source.byte_count for source in sources),
            ):
                session = backend.start_restore(sources)
            with self._profile.phase("client_memory_manager_construction"):
                mm = GMSClientMemoryManager(
                    self._socket_path,
                    device=self.device,
                    profile=self._profile,
                    cuda_initialized=self._cuda_operations is not None,
                )
            with mm:
                mm.connect(RequestedLockType.RW, timeout_ms=self._timeout_ms)
                if clear_existing:
                    logger.info("RW connect cleared any previously committed GMS state")

                with self._profile.phase(
                    "restore_target_mapping_envelope",
                    count=len(manifest.allocations),
                    byte_count=sum(
                        allocation.aligned_size for allocation in manifest.allocations
                    ),
                ):
                    id_map, targets = self._allocate_restore_targets(mm, manifest)
                with self._profile.phase(
                    "payload_restore_total",
                    backend=backend_name,
                    count=len(sources),
                    byte_count=sum(source.byte_count for source in sources),
                ):
                    session.restore(targets)
                logger.info(
                    "%s restored %d allocation(s) to GMS memory",
                    backend_name,
                    len(manifest.allocations),
                )

                self._restore_metadata(mm, saved_metadata, id_map)
                with self._profile.phase("commit"):
                    if not mm.commit():
                        raise RuntimeError("GMS commit failed after restore")
        finally:
            try:
                if session is not None:
                    with self._profile.phase("restore_session_cleanup"):
                        session.close()
            finally:
                try:
                    with self._profile.phase("transfer_backend_cleanup"):
                        backend.close()
                finally:
                    self._profile.emit_aggregates()

        logger.info(
            "load_to_gms complete: %d allocations, %d metadata keys",
            len(id_map),
            len(saved_metadata),
        )
        return id_map

    def _restore_metadata(
        self,
        mm: Any,
        saved_metadata: Dict[str, Dict[str, Any]],
        id_map: Dict[str, str],
    ) -> None:
        metadata_bytes = sum(
            len(key.encode()) + len(meta["value"])
            for key, meta in saved_metadata.items()
        )
        with self._profile.phase(
            "metadata_put_loop",
            count=len(saved_metadata),
            byte_count=metadata_bytes,
        ):
            for key, meta in saved_metadata.items():
                old_alloc_id = meta["allocation_id"]
                new_alloc_id = id_map.get(old_alloc_id, old_alloc_id)
                ok = mm.metadata_put(
                    key,
                    new_alloc_id,
                    meta["offset_bytes"],
                    meta["value"],
                )
                if not ok:
                    raise RuntimeError(f"Failed to write metadata key={key!r}")
                logger.debug(
                    "Restored metadata key=%s -> alloc=%s",
                    key,
                    new_alloc_id,
                )
        logger.info("Restored %d metadata keys; committing", len(saved_metadata))

    def _save_metadata(self, mm: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in mm.metadata_list():
            got = mm.metadata_get(key)
            if got is None:
                logger.warning("Metadata key disappeared during dump: %s", key)
                continue
            allocation_id, offset_bytes, value = got
            result[key] = {
                "allocation_id": str(allocation_id),
                "offset_bytes": int(offset_bytes),
                "value": base64.b64encode(value).decode("ascii"),
            }
        return result
