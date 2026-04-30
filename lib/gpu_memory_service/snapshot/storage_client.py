# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS storage client: save GMS state to disk and load it back."""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from gpu_memory_service.snapshot.disk import (  # noqa: F401  re-exported for external callers
    ShardWriter as _ShardWriter,
)
from gpu_memory_service.snapshot.disk import decode_metadata as _decode_metadata_impl
from gpu_memory_service.snapshot.disk import (
    group_entries_by_shard as _group_entries_by_shard_impl,
)
from gpu_memory_service.snapshot.disk import (
    load_manifest_and_metadata as _load_manifest_and_metadata_impl,
)
from gpu_memory_service.snapshot.disk import (
    plan_shard_layout as _plan_shard_layout_impl,
)
from gpu_memory_service.snapshot.disk import (
    read_shard_sequential as _read_shard_sequential_impl,
)
from gpu_memory_service.snapshot.disk import (
    read_shard_to_queue as _read_shard_to_queue_impl,
)
from gpu_memory_service.snapshot.model import CURRENT_VERSION as _CURRENT_VERSION
from gpu_memory_service.snapshot.model import AllocationEntry, SaveManifest
from gpu_memory_service.snapshot.transfer import (
    AIO_TRANSFER_BACKEND as _AIO_TRANSFER_BACKEND,
)
from gpu_memory_service.snapshot.transfer import (
    DEFAULT_TRANSFER_BACKEND as _DEFAULT_TRANSFER_BACKEND,
)
from gpu_memory_service.snapshot.transfer import (
    GMSTransferTarget,
    build_file_transfer_sources,
    create_transfer_backend,
)

logger = logging.getLogger(__name__)

try:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from gpu_memory_service.common.locks import RequestedLockType

    _GMS_CORE_IMPORTS_AVAILABLE = True
except ImportError:
    _GMS_CORE_IMPORTS_AVAILABLE = False
    GMSClientMemoryManager = None  # type: ignore[assignment,misc]
    RequestedLockType = None  # type: ignore[assignment]

try:
    from gpu_memory_service.client.torch.tensor import _tensor_from_pointer

    _GMS_TENSOR_IMPORTS_AVAILABLE = True
except ImportError:
    _GMS_TENSOR_IMPORTS_AVAILABLE = False
    _tensor_from_pointer = None  # type: ignore[assignment]

_GMS_IMPORTS_AVAILABLE = _GMS_CORE_IMPORTS_AVAILABLE and _GMS_TENSOR_IMPORTS_AVAILABLE

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


def _read_shard_sequential(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    device: int,
    pin_memory: bool = False,
) -> Dict[str, "torch.Tensor"]:
    """Facade wrapper kept for test patchability and backwards compatibility."""
    return _read_shard_sequential_impl(
        abs_path,
        sorted_entries,
        device,
        pin_memory=pin_memory,
        os_module=os,
        np_module=_get_numpy_module(),
        torch_module=torch,
        logger=logger,
    )


def _get_numpy_module() -> Any:
    try:
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError("numpy is required to read GMS snapshot shards") from exc
    return np_module


def _decode_metadata(raw_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # Re-exported for external callers (e.g. multi_ssd_bench.py).
    return _decode_metadata_impl(raw_meta)


def _group_entries_by_shard(
    allocations: List[AllocationEntry],
) -> Dict[str, List[AllocationEntry]]:
    return _group_entries_by_shard_impl(allocations)


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


def _read_shard_to_queue(
    abs_path: str,
    sorted_entries: List[AllocationEntry],
    work_q: "queue.Queue[Optional[Tuple[AllocationEntry, 'torch.Tensor']]]",
    *,
    pin_memory: bool,
    cancel_event: Optional[threading.Event] = None,
) -> int:
    return _read_shard_to_queue_impl(
        abs_path,
        sorted_entries,
        work_q,
        pin_memory=pin_memory,
        read_shard=_read_shard_sequential,
        cancel_event=cancel_event,
    )


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
    ) -> None:
        self.output_dir = output_dir
        self.device = device
        self._timeout_ms = timeout_ms
        self._shard_size = shard_size_bytes
        self._transfer_backend = transfer_backend

        if socket_path is None:
            from gpu_memory_service.common.utils import get_socket_path

            socket_path = get_socket_path(device)
        self._socket_path = socket_path

    def save(self, max_workers: int = 4) -> SaveManifest:
        """Connect to GMS in RO mode and save all allocations + metadata to disk."""
        self._validate_save_request()
        output_dir, shards_dir = self._prepare_output_dir()

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
                shards_dir,
                allocations_info,
                va_list,
                max_workers=max_workers,
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
        if not _GMS_IMPORTS_AVAILABLE:
            raise RuntimeError(
                "GMS client imports unavailable (missing cuda-python or torch)"
            )
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for save()")
        if self.output_dir is None:
            raise ValueError(
                "output_dir must be set to call save(); pass it to GMSStorageClient()"
            )

    def _prepare_output_dir(self) -> Tuple[str, str]:
        assert self.output_dir is not None
        os.makedirs(self.output_dir, exist_ok=True)
        shards_dir = os.path.join(self.output_dir, "shards")
        os.makedirs(shards_dir, exist_ok=True)
        for name in os.listdir(shards_dir):
            if name.startswith("shard_") and name.endswith(".bin"):
                os.unlink(os.path.join(shards_dir, name))
        return self.output_dir, shards_dir

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
        shards_dir: str,
        allocations_info: List[Dict[str, Any]],
        va_list: List[int],
        *,
        max_workers: int,
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
            abs_path = os.path.join(shards_dir, filename)
            rel_path = os.path.join("shards", filename)
            with open(abs_path, "wb") as handle:
                for index, byte_offset in alloc_pairs:
                    alloc = allocations_info[index]
                    aligned_size = int(alloc["aligned_size"])
                    tensor = _tensor_from_pointer(
                        va_list[index],
                        [aligned_size],
                        [1],
                        torch.uint8,
                        self.device,
                    )
                    tensor.cpu().numpy().tofile(handle)
                    entries[index] = AllocationEntry(
                        allocation_id=alloc["allocation_id"],
                        size=int(alloc["size"]),
                        aligned_size=aligned_size,
                        tag=str(alloc.get("tag", "default")),
                        tensor_file=rel_path,
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
        logger.info("Phase B complete: wrote %d shards", len(shard_groups))
        return [entry for entry in entries if entry is not None]

    def _write_json(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _allocate_restore_targets(
        self,
        mm: Any,
        manifest: SaveManifest,
    ) -> Tuple[Dict[str, str], Dict[str, GMSTransferTarget]]:
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
            "Phase A complete: allocated %d GMS VAs",
            len(targets),
        )
        return id_map, targets

    def load_to_gms(
        self,
        input_dir: str,
        *,
        max_workers: int = 4,
        clear_existing: bool = True,
        transfer_backend: Optional[str] = None,
        wait_for_socket: bool = False,
    ) -> Dict[str, str]:
        backend_name = transfer_backend or self._transfer_backend
        self._validate_load_request(backend_name)

        manifest, saved_metadata = _load_manifest_and_metadata(input_dir)
        sources = build_file_transfer_sources(input_dir, manifest.allocations)
        backend = create_transfer_backend(
            backend_name,
            device=self.device,
            max_workers=max_workers,
            torch_module=torch if _TORCH_AVAILABLE else None,
            tensor_from_pointer=_tensor_from_pointer
            if _GMS_TENSOR_IMPORTS_AVAILABLE
            else None,
        )
        session = None
        id_map: Dict[str, str] = {}

        try:
            session = backend.start_restore(sources)
            if wait_for_socket:
                from gpu_memory_service.common.utils import wait_for_weights_socket

                wait_for_weights_socket(self.device)
            with GMSClientMemoryManager(self._socket_path, device=self.device) as mm:
                mm.connect(RequestedLockType.RW, timeout_ms=self._timeout_ms)
                if clear_existing:
                    logger.info("RW connect cleared any previously committed GMS state")

                id_map, targets = self._allocate_restore_targets(mm, manifest)
                session.restore(targets)
                logger.info(
                    "Phase B complete: %s restored %d allocations to GMS memory",
                    backend.name,
                    len(manifest.allocations),
                )

                self._restore_metadata(mm, saved_metadata, id_map)
                if not mm.commit():
                    raise RuntimeError("GMS commit failed after restore")
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

    def _validate_load_request(self, transfer_backend: str) -> None:
        if not _GMS_CORE_IMPORTS_AVAILABLE:
            raise RuntimeError("GMS client imports unavailable (missing cuda-python)")
        cpu_staged_backends = {_DEFAULT_TRANSFER_BACKEND, _AIO_TRANSFER_BACKEND}
        if transfer_backend in cpu_staged_backends and not _GMS_IMPORTS_AVAILABLE:
            raise RuntimeError(
                f"{transfer_backend} GMS transfer backend requires cuda-python and torch"
            )
        if transfer_backend in cpu_staged_backends and not _TORCH_AVAILABLE:
            raise RuntimeError(
                f"{transfer_backend} GMS transfer backend requires PyTorch"
            )

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

    @staticmethod
    def load_tensors(
        input_dir: str,
        device: int = 0,
        *,
        max_workers: int = 4,
    ) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, Dict[str, Any]]]:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for load_tensors()")

        manifest, metadata = _load_manifest_and_metadata(input_dir)
        groups = _group_entries_by_shard(manifest.allocations)
        tensors: Dict[str, "torch.Tensor"] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _read_shard_sequential,
                    os.path.join(input_dir, rel_path),
                    sorted_entries,
                    device,
                ): rel_path
                for rel_path, sorted_entries in groups.items()
            }
            for future in as_completed(futures):
                rel_path = futures[future]
                try:
                    tensors.update(future.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load shard {rel_path}: {exc}"
                    ) from exc

        logger.info("Loaded %d allocations from %s", len(tensors), input_dir)
        return tensors, metadata

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
