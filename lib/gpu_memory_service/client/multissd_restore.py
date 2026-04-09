# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
import queue
import shutil
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gpu_memory_service.client.gms_storage_client import (
    AllocationEntry,
    SaveManifest,
    _decode_metadata,
    _group_entries_by_shard,
    _read_shard_sequential,
)
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.client.torch.tensor import _tensor_from_pointer
from gpu_memory_service.common.types import RequestedLockType

try:
    import torch

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    torch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
DEFAULT_SHARD_SUBDIR = "gms_shards"
_WORK_QUEUE_DEPTH_MULTIPLIER = 2

_libc_name = ctypes.util.find_library("c")
_libc: Optional[ctypes.CDLL] = (
    ctypes.CDLL(_libc_name, use_errno=True) if _libc_name else None
)


def distribute_shards(
    input_dir: Path,
    nvme_dirs: List[Path],
    *,
    shard_subdir: str = DEFAULT_SHARD_SUBDIR,
) -> None:
    shard_files = sorted((input_dir / "shards").glob("shard_*.bin"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {input_dir / 'shards'}")
    if not nvme_dirs:
        raise ValueError("nvme_dirs is empty")

    logger.info(
        "Distributing %d shards across %d NVMes into %s",
        len(shard_files),
        len(nvme_dirs),
        shard_subdir,
    )
    for nvme_dir in nvme_dirs:
        target_dir = nvme_dir / shard_subdir
        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)

    def copy_one(index: int, source: Path) -> None:
        target = nvme_dirs[index % len(nvme_dirs)] / shard_subdir / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    with ThreadPoolExecutor(max_workers=len(nvme_dirs)) as pool:
        futures = {
            pool.submit(copy_one, index, source): index
            for index, source in enumerate(shard_files)
        }
        for future in as_completed(futures):
            future.result()

    os.sync()
    logger.info("Shard distribution complete")


def load_to_gms_multissd(
    input_dir: Path,
    nvme_dirs: List[Path],
    *,
    socket_path: str,
    device: int,
    timeout_ms: Optional[int] = None,
    clear_existing: bool = True,
    drop_cache: bool = True,
    shard_subdir: str = DEFAULT_SHARD_SUBDIR,
) -> Dict[str, float | int]:
    if not nvme_dirs:
        raise ValueError("nvme_dirs is empty")

    manifest = SaveManifest.from_dict(
        json.loads((input_dir / "manifest.json").read_text())
    )
    raw_meta = {}
    metadata_path = input_dir / "gms_metadata.json"
    if metadata_path.exists():
        raw_meta = json.loads(metadata_path.read_text())
    total_gib = sum(entry.aligned_size for entry in manifest.allocations) / 1024**3
    nvme_groups = build_nvme_groups(
        manifest,
        nvme_dirs,
        shard_subdir=shard_subdir,
    )
    if not nvme_groups:
        raise RuntimeError("No NVMe shard groups were discovered")

    logger.info(
        "Multi-SSD restore: %.2f GiB | %d shards | %d NVMes | 1 thread/NVMe",
        total_gib,
        sum(len(group) for group in nvme_groups.values()),
        len(nvme_groups),
    )
    for nvme_dir, shard_group in sorted(nvme_groups.items()):
        logger.info("  %s <- %d shards", nvme_dir, len(shard_group))

    if drop_cache:
        drop_shard_caches(nvme_groups)

    disk_seconds = float("nan")
    phase_a_seconds = float("nan")
    combined_seconds = float("nan")
    saved_metadata = _decode_metadata(raw_meta)

    with GMSClientMemoryManager(socket_path, device=device) as mm:
        mm.connect(RequestedLockType.RW, timeout_ms=timeout_ms)
        if clear_existing:
            cleared = mm.clear_all_handles()
            if cleared:
                logger.info("Cleared %d pre-existing allocations", cleared)

        libcudart = ctypes.CDLL("libcudart.so", use_errno=True)
        libcudart.cudaHostRegister.restype = ctypes.c_int
        libcudart.cudaHostUnregister.restype = ctypes.c_int

        worker_count = len(nvme_groups)
        streams = (
            [torch.cuda.Stream(device=device) for _ in range(worker_count)]
            if _TORCH_OK and torch.cuda.is_available()
            else []
        )
        entry_by_id = {
            entry.allocation_id: entry for entry in manifest.allocations
        }
        work_queue: queue.Queue = queue.Queue(
            maxsize=max(1, worker_count * _WORK_QUEUE_DEPTH_MULTIPLIER)
        )
        state_lock = threading.Lock()
        cancel_event = threading.Event()
        registered_ptrs: list = []
        staged_srcs: list = []
        copy_errors: list[BaseException] = []
        id_map: Dict[str, str] = {}
        vas: Dict[str, int] = {}
        va_events: Dict[str, threading.Event] = {
            entry.allocation_id: threading.Event() for entry in manifest.allocations
        }

        def register_and_copy(stream_index: int) -> None:
            while True:
                try:
                    item = work_queue.get(timeout=0.1)
                except queue.Empty:
                    if cancel_event.is_set():
                        return
                    continue
                if item is None:
                    return
                entry, src = item
                try:
                    while not va_events[entry.allocation_id].wait(timeout=0.1):
                        if cancel_event.is_set():
                            return
                    if streams and not src.is_pinned():
                        pointer = ctypes.c_void_p(src.data_ptr())
                        result = libcudart.cudaHostRegister(
                            pointer,
                            ctypes.c_size_t(src.nbytes),
                            ctypes.c_uint(0),
                        )
                        if result == 0:
                            with state_lock:
                                registered_ptrs.append(pointer)
                    dst = _tensor_from_pointer(
                        vas[entry.allocation_id],
                        [entry.aligned_size],
                        [1],
                        torch.uint8,
                        device,
                    )
                    if streams:
                        with torch.cuda.stream(streams[stream_index]):
                            dst.copy_(src, non_blocking=True)
                    else:
                        dst.copy_(src)
                    with state_lock:
                        staged_srcs.append(src)
                except Exception as exc:  # noqa: BLE001
                    with state_lock:
                        copy_errors.append(exc)

        def drain_queue() -> None:
            while True:
                try:
                    work_queue.get_nowait()
                except queue.Empty:
                    return

        def cancel_pipeline() -> None:
            cancel_event.set()
            for event in va_events.values():
                event.set()
            drain_queue()

        def stop_copy_threads(*, drain: bool = False) -> None:
            if drain:
                drain_queue()
            for _ in copy_threads:
                while True:
                    try:
                        work_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        if drain:
                            drain_queue()
            for thread in copy_threads:
                thread.join()

        copy_threads = [
            threading.Thread(target=register_and_copy, args=(index,), daemon=True)
            for index in range(worker_count)
        ]
        for thread in copy_threads:
            thread.start()

        combined_started = time.monotonic()
        disk_pool = ThreadPoolExecutor(max_workers=worker_count)
        disk_futures = {
            disk_pool.submit(
                read_and_enqueue,
                shard_group,
                str(nvme_dir),
                work_queue,
                entry_by_id,
                cancel_event=cancel_event,
            ): nvme_dir
            for nvme_dir, shard_group in nvme_groups.items()
        }

        phase_started = time.monotonic()
        phase_a_step_seconds: list[float] = []
        try:
            for entry in manifest.allocations:
                step_started = time.monotonic()
                va = mm.create_mapping(size=entry.size, tag=entry.tag)
                phase_a_step_seconds.append(time.monotonic() - step_started)
                new_id = mm.get_allocation_id(va)
                id_map[entry.allocation_id] = new_id
                vas[entry.allocation_id] = va
                va_events[entry.allocation_id].set()
        except Exception:
            cancel_pipeline()
            disk_pool.shutdown(wait=True, cancel_futures=True)
            stop_copy_threads(drain=True)
            raise
        phase_a_seconds = time.monotonic() - phase_started
        logger.info(
            "Phase A complete: %.3fs  (%d GMS VAs allocated)",
            phase_a_seconds,
            len(vas),
        )
        if phase_a_step_seconds:
            sorted_steps = sorted(phase_a_step_seconds)
            logger.info(
                "Phase A step timings: avg=%.4fs p50=%.4fs p95=%.4fs max=%.4fs",
                statistics.fmean(sorted_steps),
                statistics.median(sorted_steps),
                sorted_steps[min(len(sorted_steps) - 1, int(len(sorted_steps) * 0.95))],
                sorted_steps[-1],
            )
        create_mapping_step_summary = mm.take_create_mapping_step_summary()
        if create_mapping_step_summary:
            for step_name, step_stats in create_mapping_step_summary.items():
                logger.info(
                    "Phase A create_mapping %s: avg=%.4fs p50=%.4fs p95=%.4fs max=%.4fs",
                    step_name,
                    step_stats["avg"],
                    step_stats["p50"],
                    step_stats["p95"],
                    step_stats["max"],
                )

        try:
            for future in as_completed(disk_futures):
                future.result()
        finally:
            disk_pool.shutdown(wait=True)
        disk_seconds = time.monotonic() - combined_started
        logger.info(
            "Disk reads complete: %.2fs  (%.2f GiB/s)",
            disk_seconds,
            total_gib / disk_seconds if disk_seconds > 0 else float("nan"),
        )

        try:
            stop_copy_threads()
            if streams:
                torch.cuda.synchronize(device=device)
            combined_seconds = time.monotonic() - combined_started
        finally:
            for pointer in registered_ptrs:
                libcudart.cudaHostUnregister(pointer)
            staged_srcs.clear()

        if copy_errors:
            raise copy_errors[0]

        logger.info(
            "Disk+B combined: %.2fs  (%.2f GiB/s effective)",
            combined_seconds,
            total_gib / combined_seconds if combined_seconds > 0 else float("nan"),
        )

        for key, meta in saved_metadata.items():
            old_id = meta["allocation_id"]
            new_id = id_map.get(old_id, old_id)
            mm.metadata_put(key, new_id, meta["offset_bytes"], meta["value"])
        if not mm.commit():
            raise RuntimeError("GMS commit failed after restore")

    result: Dict[str, float | int] = {
        "n_nvmes": len(nvme_groups),
        "total_gib": total_gib,
        "disk_s": disk_seconds,
        "phaseA_s": phase_a_seconds,
        "combined_s": combined_seconds,
        "total_s": combined_seconds,
    }
    logger.info("Multi-SSD restore summary: %s", json.dumps(result, sort_keys=True))
    return result


def build_nvme_groups(
    manifest: SaveManifest,
    nvme_dirs: List[Path],
    *,
    shard_subdir: str = DEFAULT_SHARD_SUBDIR,
) -> Dict[Path, List[Tuple[str, List[AllocationEntry]]]]:
    by_shard = _group_entries_by_shard(manifest.allocations)
    sorted_rel_paths = sorted(by_shard.keys())
    result: Dict[Path, List[Tuple[str, List[AllocationEntry]]]] = defaultdict(list)
    for index, rel_path in enumerate(sorted_rel_paths):
        nvme_dir = nvme_dirs[index % len(nvme_dirs)]
        shard_name = Path(rel_path).name
        abs_path = str(nvme_dir / shard_subdir / shard_name)
        result[nvme_dir].append((abs_path, by_shard[rel_path]))
    return dict(result)


def drop_shard_caches(
    nvme_groups: Dict[Path, List[Tuple[str, List[AllocationEntry]]]],
) -> None:
    if _libc is None:
        return
    evicted = 0
    for shard_list in nvme_groups.values():
        for abs_path, _ in shard_list:
            try:
                fd = os.open(abs_path, os.O_RDONLY)
                try:
                    _libc.posix_fadvise(
                        fd,
                        ctypes.c_int64(0),
                        ctypes.c_int64(0),
                        4,
                    )
                finally:
                    os.close(fd)
                evicted += 1
            except OSError:
                continue
    logger.info("Page-cache evicted for %d shard files", evicted)


def read_and_enqueue(
    shard_list: List[Tuple[str, List[AllocationEntry]]],
    nvme_label: str,
    work_queue: "queue.Queue",
    entry_by_id: Dict[str, AllocationEntry],
    *,
    cancel_event: Optional[threading.Event] = None,
) -> int:
    loaded = 0
    for abs_path, sorted_entries in shard_list:
        logger.debug("%s reading %s (%d allocs)", nvme_label, abs_path, len(sorted_entries))
        shard_result = _read_shard_sequential(abs_path, sorted_entries, device=-1)
        for allocation_id, src in shard_result.items():
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("%s cancelled mid-enqueue", nvme_label)
                    return loaded
                try:
                    work_queue.put((entry_by_id[allocation_id], src), timeout=0.1)
                    break
                except queue.Full:
                    pass
            loaded += 1
    logger.info("%s done - %d allocations enqueued", nvme_label, loaded)
    return loaded
