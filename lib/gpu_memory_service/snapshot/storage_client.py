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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gpu_memory_service.snapshot.backends.sharded_ssd import (
    DEFAULT_SHARDED_SSD_QUEUES_PER_ROOT,
    SHARDED_SSD_QUEUES_PER_ROOT_CONFIG_KEY,
    SHARDED_SSD_ROOTS_CONFIG_KEY,
)
from gpu_memory_service.snapshot.disk import DeviceToFileWriter
from gpu_memory_service.snapshot.disk import decode_metadata as _decode_metadata_impl
from gpu_memory_service.snapshot.disk import (
    load_manifest_and_metadata as _load_manifest_and_metadata_impl,
)
from gpu_memory_service.snapshot.disk import (
    plan_shard_layout as _plan_shard_layout_impl,
)
from gpu_memory_service.snapshot.model import CURRENT_VERSION as _CURRENT_VERSION
from gpu_memory_service.snapshot.model import AllocationEntry, SaveManifest
from gpu_memory_service.snapshot.transfer import (
    DEFAULT_TRANSFER_BACKEND as _DEFAULT_TRANSFER_BACKEND,
)
from gpu_memory_service.snapshot.transfer import (
    GMSSnapshotConfig,
    GMSTransferTarget,
    build_file_transfer_sources,
    create_transfer_backend,
)

logger = logging.getLogger(__name__)

PACKED_RESTORE_SLAB_SIZE_BYTES = 16 * 1024**3

try:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from gpu_memory_service.common.cuda_utils import (
        align_to_granularity,
        cumem_address_free,
        cumem_address_reserve,
        cumem_import_from_shareable_handle_close_fd,
        cumem_map,
        cumem_release,
        cumem_set_access,
        cumem_unmap,
        cuda_synchronize,
    )
    from gpu_memory_service.common.locks import RequestedLockType
    from gpu_memory_service.common.protocol.messages import (
        CreatePackedLayoutRequest,
        PackedRestorePlacement,
    )

    _GMS_CORE_IMPORTS_AVAILABLE = True
except ImportError:
    _GMS_CORE_IMPORTS_AVAILABLE = False
    GMSClientMemoryManager = None  # type: ignore[assignment,misc]
    RequestedLockType = None  # type: ignore[assignment]
    CreatePackedLayoutRequest = None  # type: ignore[assignment,misc]
    PackedRestorePlacement = None  # type: ignore[assignment,misc]


def _decode_metadata(raw_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # Re-exported for external callers (e.g. multi_ssd_bench.py).
    return _decode_metadata_impl(raw_meta)


def _allocation_record(alloc: Any) -> Dict[str, Any]:
    if isinstance(alloc, dict):
        return alloc
    return {
        "allocation_id": str(alloc.allocation_id),
        "size": int(alloc.size),
        "aligned_size": int(alloc.aligned_size),
        "tag": str(alloc.tag),
        "layout_slot": int(alloc.layout_slot),
    }


def _plan_shard_layout(
    allocations_info: List[Dict[str, Any]],
    shard_size_bytes: int,
) -> List[Tuple[int, int]]:
    return _plan_shard_layout_impl(allocations_info, shard_size_bytes)


def _load_manifest_and_metadata(
    input_dir: str,
) -> Tuple[SaveManifest, Dict[str, Dict[str, Any]]]:
    return _load_manifest_and_metadata_impl(input_dir)


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
        transfer_backend: str = _DEFAULT_TRANSFER_BACKEND,
        sharded_ssd_roots: Optional[Sequence[str]] = None,
        sharded_ssd_queues_per_root: int = DEFAULT_SHARDED_SSD_QUEUES_PER_ROOT,
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
        self._sharded_ssd_queues_per_root = int(sharded_ssd_queues_per_root)

        if socket_path is None:
            from gpu_memory_service.common.utils import get_socket_path

            socket_path = get_socket_path(device)
        self._socket_path = socket_path

    def save(self, max_workers: int = 4) -> SaveManifest:
        """Connect to GMS in RO mode and save all allocations + metadata to disk."""
        self._validate_save_request()
        output_dir, shard_dirs, use_absolute_shard_paths = self._prepare_output_dir()

        mm = GMSClientMemoryManager(self._socket_path, device=self.device)
        try:
            mm.connect(RequestedLockType.RO, timeout_ms=self._timeout_ms)
            layout_hash = mm.get_memory_layout_hash()
            if not layout_hash:
                raise RuntimeError(
                    "GMS server has no committed weights; nothing to dump"
                )
            allocations_info = [
                _allocation_record(alloc) for alloc in mm.list_handles()
            ]
            va_list = self._import_source_mappings(mm, allocations_info)
            entries = self._write_shards(
                shard_dirs,
                allocations_info,
                va_list,
                max_workers=max_workers,
                use_absolute_shard_paths=use_absolute_shard_paths,
            )
            metadata = self._save_metadata(mm)
        except Exception:
            mm.close(best_effort=True)
            raise

        self._write_json(os.path.join(output_dir, "gms_metadata.json"), metadata)
        manifest = SaveManifest(
            version=_CURRENT_VERSION,
            timestamp=time.time(),
            layout_hash=layout_hash,
            device=self.device,
            allocations=entries,
        )
        self._write_json(os.path.join(output_dir, "manifest.json"), manifest.to_dict())
        logger.info("Wrote manifest with %d allocations", len(entries))

        # Best-effort cleanup; CUDA context may be invalid after
        # checkpoint (cuda-checkpoint tears down device state).
        mm.close(best_effort=True)

        return manifest

    def _validate_save_request(self) -> None:
        if not _GMS_CORE_IMPORTS_AVAILABLE:
            raise RuntimeError("GMS client imports unavailable (missing cuda-python)")
        if self.output_dir is None:
            raise ValueError(
                "output_dir must be set to call save(); pass it to GMSStorageClient()"
            )

    def _prepare_output_dir(self) -> Tuple[str, List[str], bool]:
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

    def _import_source_mappings(
        self,
        mm: Any,
        allocations_info: List[Dict[str, Any]],
    ) -> List[int]:
        va_list = [
            mm.create_mapping(allocation_id=alloc["allocation_id"])
            for alloc in allocations_info
        ]
        logger.info("Phase A complete: imported %d allocation VAs", len(va_list))
        return va_list

    def _write_shards(
        self,
        shard_dirs: List[str],
        allocations_info: List[Dict[str, Any]],
        va_list: List[int],
        *,
        max_workers: int,
        use_absolute_shard_paths: bool = False,
    ) -> List[AllocationEntry]:
        layout = _plan_shard_layout(allocations_info, self._shard_size)
        shard_groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for index, (shard_idx, byte_offset) in enumerate(layout):
            shard_groups[shard_idx].append((index, byte_offset))

        entries: List[Optional[AllocationEntry]] = [None] * len(allocations_info)

        def _write_one_shard(
            shard_idx: int, alloc_pairs: List[Tuple[int, int]]
        ) -> None:
            filename = f"shard_{shard_idx:04d}.bin"
            shards_dir = shard_dirs[shard_idx % len(shard_dirs)]
            abs_path = os.path.join(shards_dir, filename)
            rel_path = os.path.join("shards", filename)
            tensor_file = abs_path if use_absolute_shard_paths else rel_path
            with DeviceToFileWriter(abs_path, device=self.device) as writer:
                for index, byte_offset in alloc_pairs:
                    alloc = allocations_info[index]
                    aligned_size = int(alloc["aligned_size"])
                    writer.write_device(va_list[index], aligned_size)
                    entries[index] = AllocationEntry(
                        allocation_id=alloc["allocation_id"],
                        size=int(alloc["size"]),
                        aligned_size=aligned_size,
                        tag=str(alloc.get("tag", "default")),
                        tensor_file=tensor_file,
                        tensor_offset=byte_offset,
                    )

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
            "Phase B complete: wrote %d shards across %d shard root(s)",
            len(shard_groups),
            len(shard_dirs),
        )
        return [entry for entry in entries if entry is not None]

    def _write_json(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

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
            va = mm.create_mapping(size=entry.size, tag=entry.tag)
            id_map[old_id] = mm.mappings[va].allocation_id
            targets[old_id] = GMSTransferTarget(
                allocation_id=old_id,
                va=va,
                device=self.device,
                byte_count=entry.aligned_size,
            )
        logger.info(
            "Phase A complete: allocated %d GMS VAs in %.3fs",
            len(targets),
            time.monotonic() - t0,
        )
        return id_map, targets

    def _pack_restore_allocations(
        self,
        mm: Any,
        manifest: SaveManifest,
    ) -> Tuple[List[int], List[Any]]:
        granularity = int(mm.granularity)
        target_slab_size = align_to_granularity(
            PACKED_RESTORE_SLAB_SIZE_BYTES, granularity
        )
        backing_sizes: List[int] = []
        placements: List[Any] = []
        current_index = -1
        current_offset = 0

        for entry in manifest.allocations:
            aligned_size = align_to_granularity(int(entry.aligned_size), granularity)
            if aligned_size <= 0:
                raise RuntimeError(
                    f"Invalid allocation size in manifest for {entry.allocation_id}: "
                    f"{entry.aligned_size}"
                )

            if (
                current_index < 0
                or current_offset + aligned_size > backing_sizes[current_index]
            ):
                backing_size = max(target_slab_size, aligned_size)
                backing_sizes.append(backing_size)
                current_index = len(backing_sizes) - 1
                current_offset = 0

            placements.append(
                PackedRestorePlacement(
                    size=int(entry.size),
                    aligned_size=aligned_size,
                    tag=str(entry.tag),
                    backing_index=current_index,
                    backing_offset=current_offset,
                )
            )
            current_offset += aligned_size

        if backing_sizes and current_index >= 0:
            backing_sizes[current_index] = max(
                align_to_granularity(current_offset, granularity), granularity
            )

        return backing_sizes, placements

    def _allocate_packed_restore_targets(
        self,
        mm: Any,
        manifest: SaveManifest,
    ) -> Tuple[Dict[str, str], Dict[str, GMSTransferTarget], List[Tuple[int, int, int]]]:
        t0 = time.monotonic()
        backing_sizes, placements = self._pack_restore_allocations(mm, manifest)
        if not placements:
            return {}, {}, []

        response = mm.create_packed_layout(
            CreatePackedLayoutRequest(
                backing_sizes=backing_sizes,
                placements=placements,
            )
        )
        if len(response.allocations) != len(manifest.allocations):
            raise RuntimeError(
                "Packed restore response allocation count mismatch: "
                f"{len(response.allocations)} vs {len(manifest.allocations)}"
            )

        slab_vas: List[int] = []
        slab_handles: List[int] = []
        mapped_slabs: List[Tuple[int, int, int]] = []
        try:
            for backing_index, backing_size in enumerate(backing_sizes):
                # Any published allocation placed in this backing slab can export
                # the slab FD.  Use the first such allocation as the export key.
                export_alloc_index = next(
                    index
                    for index, placement in enumerate(placements)
                    if int(placement.backing_index) == backing_index
                )
                export_alloc = response.allocations[export_alloc_index]
                export_info, fd = mm.export_handle_info(export_alloc.allocation_id)
                if int(export_info.mapping_offset) != int(
                    placements[export_alloc_index].backing_offset
                ):
                    raise RuntimeError(
                        "Packed restore export offset mismatch for backing "
                        f"{backing_index}: {export_info.mapping_offset} vs "
                        f"{placements[export_alloc_index].backing_offset}"
                    )
                handle = cumem_import_from_shareable_handle_close_fd(fd)
                va = cumem_address_reserve(backing_size, int(mm.granularity))
                cumem_map(va, backing_size, handle, 0)
                if mm.granted_lock_type is None:
                    raise RuntimeError("GMS lock is not active while mapping slabs")
                cumem_set_access(va, backing_size, self.device, mm.granted_lock_type)
                slab_vas.append(va)
                slab_handles.append(handle)
                mapped_slabs.append((va, backing_size, handle))

            id_map: Dict[str, str] = {}
            targets: Dict[str, GMSTransferTarget] = {}
            for entry, placement, alloc in zip(
                manifest.allocations, placements, response.allocations, strict=True
            ):
                old_id = entry.allocation_id
                id_map[old_id] = alloc.allocation_id
                targets[old_id] = GMSTransferTarget(
                    allocation_id=old_id,
                    va=slab_vas[int(placement.backing_index)]
                    + int(placement.backing_offset),
                    device=self.device,
                    byte_count=int(entry.aligned_size),
                )
            logger.info(
                "Phase A complete: packed %d GMS VAs into %d slab(s) "
                "(%.2f GiB backing) in %.3fs",
                len(targets),
                len(backing_sizes),
                sum(backing_sizes) / (1 << 30),
                time.monotonic() - t0,
            )
            return id_map, targets, mapped_slabs
        except Exception:
            for va, size, handle in reversed(mapped_slabs):
                cumem_unmap(va, size)
                cumem_release(handle)
                cumem_address_free(va, size)
            raise

    def load_to_gms(
        self,
        input_dir: str,
        *,
        max_workers: int = 4,
        clear_existing: bool = True,
        transfer_backend: Optional[str] = None,
    ) -> Dict[str, str]:
        backend_name = transfer_backend or self._transfer_backend
        self._validate_load_request()

        backend = create_transfer_backend(
            backend_name,
            GMSSnapshotConfig(
                device=self.device,
                max_workers=max_workers,
                backend_config={
                    SHARDED_SSD_ROOTS_CONFIG_KEY: self._sharded_ssd_roots,
                    SHARDED_SSD_QUEUES_PER_ROOT_CONFIG_KEY: (
                        self._sharded_ssd_queues_per_root
                    ),
                },
            ),
        )
        session = None
        id_map: Dict[str, str] = {}
        packed_slab_mappings: List[Tuple[int, int, int]] = []

        try:
            manifest, saved_metadata = _load_manifest_and_metadata(input_dir)
            sources = build_file_transfer_sources(input_dir, manifest.allocations)
            session = backend.start_restore(sources)
            with GMSClientMemoryManager(self._socket_path, device=self.device) as mm:
                mm.connect(RequestedLockType.RW, timeout_ms=self._timeout_ms)
                if clear_existing:
                    logger.info("RW connect cleared any previously committed GMS state")

                try:
                    id_map, targets, packed_slab_mappings = (
                        self._allocate_packed_restore_targets(mm, manifest)
                    )
                    session.restore(targets)
                    logger.info(
                        "Phase B complete: %s restored %d allocations to GMS memory",
                        backend.name,
                        len(manifest.allocations),
                    )

                    self._restore_metadata(mm, saved_metadata, id_map)
                    # The manual slab mappings are not tracked by
                    # GMSClientMemoryManager.  Keep the same publish barrier as
                    # mm.commit(): synchronize writer-side GPU work, unmap the
                    # writer's local slab mappings, then publish the committed
                    # layout.  This lets the restored main remap the published
                    # allocation views while the loader releases its temporary
                    # slab VAs.
                    cuda_synchronize()
                    for va, size, handle in reversed(packed_slab_mappings):
                        cumem_unmap(va, size)
                        cumem_release(handle)
                        cumem_address_free(va, size)
                    packed_slab_mappings = []
                    if not mm.commit():
                        raise RuntimeError("GMS commit failed after restore")
                finally:
                    for va, size, handle in reversed(packed_slab_mappings):
                        cumem_unmap(va, size)
                        cumem_release(handle)
                        cumem_address_free(va, size)
                    packed_slab_mappings = []
        finally:
            if session is not None:
                session.close()
            backend.close()

        logger.info(
            "load_to_gms complete: %d allocations, %d metadata keys",
            len(id_map),
            len(saved_metadata),
        )
        return id_map

    def _validate_load_request(self) -> None:
        if not _GMS_CORE_IMPORTS_AVAILABLE:
            raise RuntimeError("GMS client imports unavailable (missing cuda-python)")

    def _restore_metadata(
        self,
        mm: Any,
        saved_metadata: Dict[str, Dict[str, Any]],
        id_map: Dict[str, str],
    ) -> None:
        for key, meta in saved_metadata.items():
            old_alloc_id = meta["allocation_id"]
            new_alloc_id = id_map.get(old_alloc_id, old_alloc_id)
            ok = mm.metadata_put(key, new_alloc_id, meta["offset_bytes"], meta["value"])
            if not ok:
                raise RuntimeError(f"Failed to write metadata key={key!r}")
            logger.debug("Restored metadata key=%s -> alloc=%s", key, new_alloc_id)
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
