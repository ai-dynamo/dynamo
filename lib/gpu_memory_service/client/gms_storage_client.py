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
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gpu_memory_service.client._gms_storage_disk import (  # noqa: F401  re-exported for external callers
    ShardWriter as _ShardWriter,
)
from gpu_memory_service.client._gms_storage_disk import (
    decode_metadata as _decode_metadata_impl,
)
from gpu_memory_service.client._gms_storage_disk import (
    group_entries_by_shard as _group_entries_by_shard_impl,
)
from gpu_memory_service.client._gms_storage_disk import (
    load_manifest_and_metadata as _load_manifest_and_metadata_impl,
)
from gpu_memory_service.client._gms_storage_disk import (
    plan_shard_layout as _plan_shard_layout_impl,
)
from gpu_memory_service.client._gms_storage_disk import (
    read_shard_sequential as _read_shard_sequential_impl,
)
from gpu_memory_service.client._gms_storage_disk import (
    read_shard_streaming_to_queue as _read_shard_streaming_to_queue_impl,
)
from gpu_memory_service.client._gms_storage_disk import (
    read_shard_to_queue as _read_shard_to_queue_impl,
)
from gpu_memory_service.client._gms_storage_model import (
    CURRENT_VERSION as _CURRENT_VERSION,
)
from gpu_memory_service.client._gms_storage_model import AllocationEntry, SaveManifest
from gpu_memory_service.client._gms_storage_restore import (
    RestorePipelineContext as _RestorePipelineContext,
)
from gpu_memory_service.client._gms_storage_restore import (
    RestorePipelineResources as _RestorePipelineResources,
)

logger = logging.getLogger(__name__)

try:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
    from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
    from gpu_memory_service.common.locks import RequestedLockType

    _GMS_IMPORTS_AVAILABLE = True
except ImportError:
    _GMS_IMPORTS_AVAILABLE = False
    GMSClientMemoryManager = None  # type: ignore[assignment,misc]
    _tensor_from_pointer = None  # type: ignore[assignment]
    RequestedLockType = None  # type: ignore[assignment]

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
        np_module=np,
        torch_module=torch,
        logger=logger,
    )


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
    # Use streaming reader: overlaps disk I/O with GPU copies within each
    # shard instead of reading the full shard before queuing any entries.
    return _read_shard_streaming_to_queue_impl(
        abs_path,
        sorted_entries,
        work_q,
        pin_memory=pin_memory,
        cancel_event=cancel_event,
        os_module=os,
        np_module=np,
        torch_module=torch,
        logger=logger,
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
    ) -> None:
        self.output_dir = output_dir
        self.device = device
        self._timeout_ms = timeout_ms
        self._shard_size = shard_size_bytes

        if socket_path is None:
            from gpu_memory_service.common.utils import get_socket_path

            socket_path = get_socket_path(device)
        self._socket_path = socket_path

    def save(self, max_workers: int = 4) -> SaveManifest:
        """Connect to GMS in RO mode and save all allocations + metadata to disk."""
        self._validate_save_request()
        output_dir, shards_dir = self._prepare_output_dir()

        with GMSClientMemoryManager(self._socket_path, device=self.device) as mm:
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
        return manifest

    def _validate_save_request(self) -> None:
        if not _GMS_IMPORTS_AVAILABLE:
            raise RuntimeError(
                "GMS client imports unavailable (missing cuda-python or torch)"
            )
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

    def _run_restore_copy_worker(
        self,
        ctx: _RestorePipelineContext,
        stream_idx: int,
    ) -> None:
        while True:
            try:
                item = ctx.work_q.get(timeout=0.1)
            except queue.Empty:
                if ctx.cancel_event.is_set():
                    return
                continue
            if item is None:
                return

            entry, src = item
            try:
                while not ctx.va_events[entry.allocation_id].wait(timeout=0.1):
                    if ctx.cancel_event.is_set():
                        return
                dst = _tensor_from_pointer(
                    ctx.vas[entry.allocation_id],
                    [entry.aligned_size],
                    [1],
                    torch.uint8,
                    self.device,
                )
                if ctx.streams:
                    with torch.cuda.stream(ctx.streams[stream_idx]):
                        dst.copy_(src, non_blocking=src.is_pinned())
                else:
                    dst.copy_(src)
                if ctx.use_streams and src.is_pinned():
                    with ctx.lock:
                        ctx.staged_srcs.append(src)
            except Exception as exc:  # noqa: BLE001
                with ctx.lock:
                    ctx.copy_errors.append(exc)

    def _start_restore_copy_threads(
        self,
        ctx: _RestorePipelineContext,
    ) -> List[threading.Thread]:
        threads = [
            threading.Thread(
                target=self._run_restore_copy_worker,
                args=(ctx, index),
                daemon=True,
            )
            for index in range(ctx.worker_count)
        ]
        for thread in threads:
            thread.start()
        return threads

    def _prepare_restore_pipeline(
        self,
        manifest: SaveManifest,
        groups: Dict[str, List[AllocationEntry]],
        worker_count: int,
        input_dir: str,
    ) -> _RestorePipelineResources:
        ctx = _RestorePipelineContext.build(
            manifest.allocations,
            worker_count,
            device=self.device,
            use_streams=_TORCH_AVAILABLE and torch.cuda.is_available(),
            torch_module=torch,
        )
        copy_threads = self._start_restore_copy_threads(ctx)
        disk_pool = ThreadPoolExecutor(max_workers=worker_count)
        disk_futures = {
            disk_pool.submit(
                _read_shard_to_queue,
                os.path.join(input_dir, rel_path),
                sorted_entries,
                ctx.work_q,
                pin_memory=ctx.use_streams,
                cancel_event=ctx.cancel_event,
            ): rel_path
            for rel_path, sorted_entries in groups.items()
        }
        return _RestorePipelineResources(
            ctx=ctx,
            disk_pool=disk_pool,
            disk_futures=disk_futures,
            copy_threads=copy_threads,
        )

    def _allocate_restore_mappings(
        self,
        mm: Any,
        manifest: SaveManifest,
        ctx: _RestorePipelineContext,
    ) -> Dict[str, str]:
        id_map: Dict[str, str] = {}
        for entry in manifest.allocations:
            old_id = entry.allocation_id
            va = mm.create_mapping(size=entry.size, tag=entry.tag)
            id_map[old_id] = mm.mappings[va].allocation_id
            ctx.vas[old_id] = va
            ctx.va_events[old_id].set()
        logger.info(
            "Phase A complete: allocated %d GMS VAs; waiting for disk/copy pipeline",
            len(ctx.vas),
        )
        return id_map

    def _await_disk_reads(self, disk_futures: Dict[Future[int], str]) -> None:
        for future in as_completed(disk_futures):
            rel_path = disk_futures[future]
            try:
                future.result()
            except CancelledError:
                pass
            except Exception as exc:
                raise RuntimeError(f"Failed to load shard {rel_path}: {exc}") from exc

    def _stop_restore_copy_threads(
        self,
        ctx: _RestorePipelineContext,
        threads: List[threading.Thread],
        *,
        drain_queue: bool = False,
    ) -> None:
        if drain_queue:
            self._drain_restore_queue(ctx)
        for _ in threads:
            if drain_queue:
                # Cancel path: workers may have exited, so drain to make room.
                while True:
                    try:
                        ctx.work_q.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        self._drain_restore_queue(ctx)
            else:
                # Normal path: disk reads are done and workers are alive; block
                # until a slot opens rather than spinning with a timeout.
                ctx.work_q.put(None)
        for thread in threads:
            thread.join()

    def _drain_restore_queue(self, ctx: _RestorePipelineContext) -> None:
        while True:
            try:
                ctx.work_q.get_nowait()
            except queue.Empty:
                return

    def _cancel_restore_pipeline(self, ctx: _RestorePipelineContext) -> None:
        ctx.cancel_event.set()
        for event in ctx.va_events.values():
            event.set()
        self._drain_restore_queue(ctx)

    def _finalize_restore_pipeline(self, ctx: _RestorePipelineContext) -> None:
        if ctx.use_streams:
            torch.cuda.synchronize(device=self.device)
            ctx.staged_srcs.clear()
        if ctx.copy_errors:
            raise RuntimeError(
                f"Failed to copy restored data to GMS: {ctx.copy_errors[0]}"
            )

    def _drain_restore_pipeline(self, resources: _RestorePipelineResources) -> None:
        disk_error: Optional[BaseException] = None
        finalize_error: Optional[BaseException] = None
        drain_queue = False
        try:
            self._await_disk_reads(resources.disk_futures)
        except Exception as exc:
            disk_error = exc
            self._cancel_restore_pipeline(resources.ctx)
            drain_queue = True
            resources.disk_pool.shutdown(wait=True, cancel_futures=True)
        else:
            resources.disk_pool.shutdown(wait=True)
        try:
            self._stop_restore_copy_threads(
                resources.ctx,
                resources.copy_threads,
                drain_queue=drain_queue,
            )
        finally:
            resources.active = False
            try:
                self._finalize_restore_pipeline(resources.ctx)
            except Exception as exc:  # noqa: BLE001
                finalize_error = exc
        if disk_error is not None:
            raise disk_error
        if finalize_error is not None:
            raise finalize_error

    def _shutdown_restore_pipeline(
        self,
        resources: _RestorePipelineResources,
    ) -> None:
        if not resources.active:
            return
        self._cancel_restore_pipeline(resources.ctx)
        resources.disk_pool.shutdown(wait=True, cancel_futures=True)
        self._stop_restore_copy_threads(
            resources.ctx,
            resources.copy_threads,
            drain_queue=True,
        )
        resources.active = False
        # Synchronize async copies to prevent use-after-free of staged pinned
        # buffers, but suppress copy errors — the caller already has an error
        # to propagate and we must not mask it.
        try:
            self._finalize_restore_pipeline(resources.ctx)
        except Exception:  # noqa: BLE001
            pass

    def load_to_gms(
        self,
        input_dir: str,
        *,
        max_workers: int = 4,
        clear_existing: bool = True,
    ) -> Dict[str, str]:
        if not _GMS_IMPORTS_AVAILABLE:
            raise RuntimeError(
                "GMS client imports unavailable (missing cuda-python or torch)"
            )

        manifest, saved_metadata = _load_manifest_and_metadata(input_dir)
        groups = _group_entries_by_shard(manifest.allocations)
        worker_count = max(1, min(max_workers, len(groups) or 1))

        with GMSClientMemoryManager(self._socket_path, device=self.device) as mm:
            mm.connect(RequestedLockType.RW, timeout_ms=self._timeout_ms)
            if clear_existing:
                logger.info("RW connect cleared any previously committed GMS state")

            resources = self._prepare_restore_pipeline(
                manifest,
                groups,
                worker_count,
                input_dir,
            )
            try:
                id_map = self._allocate_restore_mappings(mm, manifest, resources.ctx)
                self._drain_restore_pipeline(resources)
            except Exception:
                self._shutdown_restore_pipeline(resources)
                raise

            logger.info(
                "Phase B complete: streamed %d allocations to GMS memory",
                len(manifest.allocations),
            )
            self._restore_metadata(mm, saved_metadata, id_map)
            if not mm.commit():
                raise RuntimeError("GMS commit failed after restore")

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
