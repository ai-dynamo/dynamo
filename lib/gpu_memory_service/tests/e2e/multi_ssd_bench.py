#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-SSD GMS restore benchmark with exclusive per-SSD thread affinity.

The ThreadPoolExecutor submits ONE task per NVMe (not one per shard).
Each task reads ALL shards on its assigned drive sequentially — so concurrent
reads to the same SSD are structurally impossible.

Shard distribution (round-robin by sorted shard index):
    shard_N  →  /mnt/nvme{2 + N % 8}/gms_shards/shard_NNNN.bin

Restore phases
--------------
  Disk  : 8 threads, one per NVMe, each reads its shards front-to-back.
  Phase A: serial GMS VA allocation (cuMemCreate × N allocs).
  Phase B: parallel CPU→GPU copy (one CUDA stream per worker thread).

Usage
-----
    # First run: save Qwen2.5-72B to nvme2, distribute, then benchmark.
    .venv/bin/python multi_ssd_bench.py

    # Subsequent runs (shards already distributed):
    .venv/bin/python multi_ssd_bench.py --skip-distribute

    # Keep page cache warm (removes cold-read penalty for debugging):
    .venv/bin/python multi_ssd_bench.py --skip-distribute --no-drop-cache
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_WORK_QUEUE_DEPTH_MULTIPLIER = 2

# Cache the libc handle once at import time — ctypes.util.find_library +
# CDLL are non-trivial and would otherwise run once per shard file.
_libc_name = ctypes.util.find_library("c")
_libc: Optional[ctypes.CDLL] = (
    ctypes.CDLL(_libc_name, use_errno=True) if _libc_name else None
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NVME_DIRS: List[Path] = [Path(f"/mnt/nvme{i}") for i in range(2, 10)]  # nvme2..nvme9
SHARD_SUBDIR = "gms_shards"  # per-NVMe subdir that holds copied shard files

# ---------------------------------------------------------------------------
# Shared process helpers (imported from adjacent bench script)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from bench_gms_restore import (  # noqa: E402
    ManagedProcess,
    _gms_server_cmd,
    _gms_socket_path,
    _wait_gpu_free,
    run_save,
)

# ---------------------------------------------------------------------------
# GMS / PyTorch imports
# ---------------------------------------------------------------------------

try:
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

    _GMS_OK = True
except ImportError as _e:
    logger.error("GMS imports unavailable: %s", _e)
    _GMS_OK = False

try:
    import torch

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# Shard distribution: copy shards round-robin to /mnt/nvme{2..9}/gms_shards/
# ---------------------------------------------------------------------------


def distribute_shards(save_dir: Path, nvme_dirs: List[Path]) -> None:
    """Copy shard files round-robin to per-NVMe directories.

    shard_N  →  nvme_dirs[N % len(nvme_dirs)] / SHARD_SUBDIR / shard_NNNN.bin

    Uses hard links when source and destination are on the same filesystem
    (instant, no data copy).  Falls back to shutil.copy2() across devices.
    Already-present destination files are skipped.
    Copies run in parallel (one thread per NVMe) to saturate source bandwidth.
    """
    shard_files = sorted((save_dir / "shards").glob("shard_*.bin"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {save_dir / 'shards'}")

    n = len(nvme_dirs)
    logger.info(
        "Distributing %d shards across %d NVMes (~%d shards/NVMe) …",
        len(shard_files),
        n,
        (len(shard_files) + n - 1) // n,
    )

    for d in nvme_dirs:
        (d / SHARD_SUBDIR).mkdir(parents=True, exist_ok=True)

    def _copy_one(idx: int, src: Path) -> None:
        dest = nvme_dirs[idx % n] / SHARD_SUBDIR / src.name
        if dest.exists():
            dest.unlink()
        # Hard-link if same filesystem (instant); copy otherwise.
        try:
            os.link(src, dest)
            logger.debug("  shard_%04d refreshed via hard link → %s", idx, dest.parent)
        except OSError:  # EXDEV — different device
            logger.info("  copying shard_%04d → %s", idx, dest.parent)
            shutil.copy2(src, dest)

    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(_copy_one, i, f): i for i, f in enumerate(shard_files)}
        for fut in as_completed(futs):
            fut.result()  # re-raise any errors

    # Flush dirty pages to disk so O_DIRECT reads in the benchmark see clean
    # data; without this, competing write-back tanks read throughput.
    os.sync()
    logger.info("Shard distribution complete.")


# ---------------------------------------------------------------------------
# Build per-NVMe shard groups (mirrors distribute_shards assignment)
# ---------------------------------------------------------------------------


def _build_nvme_groups(
    manifest: SaveManifest,
    nvme_dirs: List[Path],
) -> Dict[Path, List[Tuple[str, List[AllocationEntry]]]]:
    """Map each NVMe to the list of (abs_path, sorted_entries) it owns.

    Assignment rule is identical to distribute_shards: shard at sorted
    index N belongs to nvme_dirs[N % len(nvme_dirs)].

    IMPORTANT: both distribute_shards and this function sort shard filenames
    lexicographically (``sorted(glob(...))`` vs ``sorted(by_shard.keys())``).
    The round-robin index N must be consistent between the two — if the sort
    order here ever diverges from distribute_shards, shards will be read from
    the wrong NVMe.

    Returns:
        { nvme_dir: [(abs_shard_path, sorted_entries), …] }
        One entry per shard file assigned to that NVMe.
    """
    by_shard = _group_entries_by_shard(manifest.allocations)
    sorted_rel_paths = sorted(by_shard.keys())  # "shards/shard_NNNN.bin" in order

    result: Dict[Path, List[Tuple[str, List[AllocationEntry]]]] = defaultdict(list)
    for idx, rel_path in enumerate(sorted_rel_paths):
        nvme = nvme_dirs[idx % len(nvme_dirs)]
        shard_name = Path(rel_path).name  # "shard_NNNN.bin"
        abs_path = str(nvme / SHARD_SUBDIR / shard_name)
        result[nvme].append((abs_path, by_shard[rel_path]))

    return dict(result)


# ---------------------------------------------------------------------------
# Page-cache eviction
# ---------------------------------------------------------------------------


def _fadvise_dontneed(path: str) -> None:
    """Evict a single file from the page cache (POSIX_FADV_DONTNEED)."""
    if _libc is None:
        return
    POSIX_FADV_DONTNEED = 4
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            _libc.posix_fadvise(
                fd, ctypes.c_int64(0), ctypes.c_int64(0), POSIX_FADV_DONTNEED
            )
        finally:
            os.close(fd)
    except OSError:
        pass


def drop_shard_caches(
    nvme_groups: Dict[Path, List[Tuple[str, List[AllocationEntry]]]],
) -> None:
    """Call fadvise(DONTNEED) on every shard file assigned to each NVMe."""
    evicted = 0
    for shard_list in nvme_groups.values():
        for abs_path, _ in shard_list:
            _fadvise_dontneed(abs_path)
            evicted += 1
    logger.info("Page-cache evicted for %d shard files", evicted)


# ---------------------------------------------------------------------------
# Disk-read worker: one thread owns one NVMe, reads all its shards in order
# ---------------------------------------------------------------------------


def _read_and_enqueue(
    shard_list: List[Tuple[str, List[AllocationEntry]]],
    nvme_label: str,
    work_q: "queue.Queue",
    entry_by_id: Dict[str, "AllocationEntry"],
    cancel_event: Optional[threading.Event] = None,
) -> int:
    """Read shards for one NVMe, putting per-allocation items on work_q after each shard.

    Reads each shard as one large O_DIRECT read (fast sequential I/O), then
    immediately enqueues all allocations.  Copy threads see data after each
    shard completes (~shard_size / NVMe_speed, e.g. 0.3 s for 1 GiB shards).

    NOTE: shard_size_bytes < ~1.2 GiB is required for the copy tail to reach zero
    and combined time to approach disk_time (speed-of-light ≈ 5.45 s total).

    Uses timed put() + cancel_event checks so that a Phase A exception that
    calls _cancel_pipeline() is guaranteed to unblock this function; without
    this, a full queue would cause this function to block forever.
    """
    n = 0
    for abs_path, sorted_entries in shard_list:
        logger.debug(
            "%s  reading %s (%d allocs)", nvme_label, abs_path, len(sorted_entries)
        )
        shard_result = _read_shard_sequential(abs_path, sorted_entries, device=-1)
        for alloc_id, src in shard_result.items():
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("%s  cancelled mid-enqueue", nvme_label)
                    return n
                try:
                    work_q.put((entry_by_id[alloc_id], src), timeout=0.1)
                    break
                except queue.Full:
                    pass
            n += 1
    logger.info("%s  done — %d allocations enqueued", nvme_label, n)
    return n


# ---------------------------------------------------------------------------
# Full restore with per-NVMe thread affinity
# ---------------------------------------------------------------------------


def restore_multissd(
    save_dir: Path,
    nvme_dirs: List[Path],
    device: int = 0,
    drop_cache: bool = True,
    log_dir: Optional[Path] = None,
) -> dict:
    """Run a full GMS restore with one exclusive reader thread per NVMe.

    Returns a dict with timing breakdown and throughput numbers.
    """
    if not _GMS_OK:
        raise RuntimeError("GMS imports not available — check venv installation")

    if log_dir is None:
        log_dir = save_dir.parent / "gms_bench_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest + metadata
    manifest = SaveManifest.from_dict(
        json.loads((save_dir / "manifest.json").read_text())
    )
    total_gib = sum(e.aligned_size for e in manifest.allocations) / 1024**3
    raw_meta: dict = {}
    meta_path = save_dir / "gms_metadata.json"
    if meta_path.exists():
        raw_meta = json.loads(meta_path.read_text())

    # Build per-NVMe shard groups (same assignment as distribute_shards)
    nvme_groups = _build_nvme_groups(manifest, nvme_dirs)
    n_threads = len(nvme_groups)

    logger.info(
        "Multi-SSD restore: %.2f GiB  |  %d shards  |  %d NVMes  |  1 thread/NVMe",
        total_gib,
        sum(len(v) for v in nvme_groups.values()),
        n_threads,
    )
    for nvme, shard_list in sorted(nvme_groups.items()):
        logger.info(
            "  %s  ←  %d shards  (%.2f GiB)",
            nvme,
            len(shard_list),
            sum(
                sum(e.aligned_size for e in entries) / 1024**3
                for _, entries in shard_list
            ),
        )

    if drop_cache:
        drop_shard_caches(nvme_groups)

    sock = _gms_socket_path(device)
    gms: Optional[ManagedProcess] = None

    disk_s = phaseA_s = combined_s = float("nan")

    try:
        # Start a fresh GMS server
        gms = ManagedProcess(_gms_server_cmd(device), log_dir / "gms_multissd.log")
        time.sleep(2)
        assert gms.is_running(), f"GMS server exited immediately.\n{gms.tail()}"

        # ── Pipelined Disk + Phase A + B (one RW lock session) ──
        # Phase A (serial cuMemCreate, ~0.6 s) and disk reads start simultaneously.
        # Phase A runs in the main thread; disk reads run in the thread pool.
        # Copy threads wait for each allocation's VA via _va_events before copying.
        # This hides Phase A entirely under the ~5 s disk phase.
        with GMSClientMemoryManager(str(sock), device=device) as mm:
            mm.connect(RequestedLockType.RW)
            cleared = mm.clear_all_handles()
            if cleared:
                logger.info("Cleared %d pre-existing allocations", cleared)

            # Setup for pipelined disk + registration + copy
            import queue as _queue
            import threading as _threading

            _libcudart = ctypes.CDLL("libcudart.so", use_errno=True)
            _libcudart.cudaHostRegister.restype = ctypes.c_int
            _libcudart.cudaHostUnregister.restype = ctypes.c_int

            streams = (
                [torch.cuda.Stream(device=device) for _ in range(n_threads)]
                if _TORCH_OK and torch.cuda.is_available()
                else []
            )

            entry_by_id = {e.allocation_id: e for e in manifest.allocations}
            _work_q: _queue.Queue = _queue.Queue(
                maxsize=max(1, n_threads * _WORK_QUEUE_DEPTH_MULTIPLIER)
            )
            _state_lock = _threading.Lock()
            _cancel_event = _threading.Event()
            _registered: list = []
            _srcs: list = []  # keep src tensors alive until after CUDA sync
            _copy_exc: list = []
            id_map: Dict[str, str] = {}
            vas: Dict[str, int] = {}

            # One event per allocation: set by Phase A after VA is ready
            _va_events: Dict[str, _threading.Event] = {
                e.allocation_id: _threading.Event() for e in manifest.allocations
            }

            def _register_and_copy(stream_idx: int) -> None:
                while True:
                    try:
                        item = _work_q.get(timeout=0.1)
                    except _queue.Empty:
                        if _cancel_event.is_set():
                            return
                        continue
                    if item is None:
                        return
                    entry, src = item
                    try:
                        # Wait until Phase A has allocated the VA for this allocation
                        while not _va_events[entry.allocation_id].wait(timeout=0.1):
                            if _cancel_event.is_set():
                                return
                        if streams and not src.is_pinned():
                            ptr = ctypes.c_void_p(src.data_ptr())
                            ret = _libcudart.cudaHostRegister(
                                ptr, ctypes.c_size_t(src.nbytes), ctypes.c_uint(0)
                            )
                            if ret == 0:
                                with _state_lock:
                                    _registered.append(ptr)
                        dst = _tensor_from_pointer(
                            vas[entry.allocation_id],
                            [entry.aligned_size],
                            [1],
                            torch.uint8,
                            device,
                        )
                        if streams:
                            with torch.cuda.stream(streams[stream_idx]):
                                dst.copy_(src, non_blocking=True)
                        else:
                            dst.copy_(src)
                        # Keep src alive until torch.cuda.synchronize() below;
                        # async DMA may still be reading it after copy_ returns.
                        with _state_lock:
                            _srcs.append(src)
                    except Exception as exc:  # noqa: BLE001
                        with _state_lock:
                            _copy_exc.append(exc)

            def _drain_queue() -> None:
                while True:
                    try:
                        _work_q.get_nowait()
                    except _queue.Empty:
                        break

            def _cancel_pipeline() -> None:
                _cancel_event.set()
                for event in _va_events.values():
                    event.set()
                _drain_queue()

            def _stop_copy_threads(*, drain_queue: bool = False) -> None:
                if drain_queue:
                    _drain_queue()
                for _ in range(n_threads):
                    while True:
                        try:
                            _work_q.put(None, timeout=0.1)
                            break
                        except _queue.Full:
                            if drain_queue:
                                _drain_queue()
                for t in copy_threads:
                    t.join()

            # Start copy threads before disk and Phase A
            copy_threads = [
                _threading.Thread(target=_register_and_copy, args=(i,), daemon=True)
                for i in range(n_threads)
            ]
            for t in copy_threads:
                t.start()

            # Start disk and Phase A simultaneously
            t_combined = time.monotonic()

            # Launch disk pool in background (non-blocking submit)
            disk_pool = ThreadPoolExecutor(max_workers=n_threads)
            disk_futs = {
                disk_pool.submit(
                    _read_and_enqueue,
                    shard_list,
                    str(nvme),
                    _work_q,
                    entry_by_id,
                    _cancel_event,
                ): nvme
                for nvme, shard_list in nvme_groups.items()
            }

            # Phase A: serial cuMemCreate — runs concurrently with disk reads above
            t_a = time.monotonic()
            try:
                for entry in manifest.allocations:
                    va = mm.create_mapping(size=entry.size, tag=entry.tag)
                    new_id = mm.get_allocation_id(va)
                    id_map[entry.allocation_id] = new_id
                    vas[entry.allocation_id] = va
                    # Signal copy threads that VA for this allocation is ready
                    _va_events[entry.allocation_id].set()
            except Exception:
                _cancel_pipeline()
                disk_pool.shutdown(wait=True, cancel_futures=True)
                _stop_copy_threads(drain_queue=True)
                raise
            phaseA_s = time.monotonic() - t_a
            logger.info(
                "Phase A complete: %.3fs  (%d GMS VAs allocated)", phaseA_s, len(vas)
            )

            # Wait for all disk reads to finish
            try:
                for fut in as_completed(disk_futs):
                    fut.result()  # propagate any read exceptions
            finally:
                disk_pool.shutdown(wait=True)
            disk_s = time.monotonic() - t_combined
            logger.info(
                "Disk reads complete: %.2fs  (%.2f GiB/s)", disk_s, total_gib / disk_s
            )

            # Signal copy threads and wait for all DMAs to be enqueued
            _stop_copy_threads()

            try:
                if streams:
                    torch.cuda.synchronize(device=device)
                combined_s = time.monotonic() - t_combined
            finally:
                for ptr in _registered:
                    _libcudart.cudaHostUnregister(ptr)
                _srcs.clear()

            if _copy_exc:
                raise _copy_exc[0]
            logger.info(
                "Disk+B combined: %.2fs  (%.2f GiB/s  effective)",
                combined_s,
                total_gib / combined_s if combined_s > 0 else float("nan"),
            )

            # Restore metadata and commit
            saved_metadata = _decode_metadata(raw_meta)
            for key, meta in saved_metadata.items():
                old_id = meta["allocation_id"]
                new_id = id_map.get(old_id, old_id)
                mm.metadata_put(key, new_id, meta["offset_bytes"], meta["value"])
            ok = mm.commit()
            if not ok:
                raise RuntimeError("GMS commit failed after restore")

    finally:
        if gms:
            gms.terminate("GMS")
        _wait_gpu_free(exclude_pids={os.getpid()}, timeout=60)
        if sock.exists():
            sock.unlink()

    total_s = combined_s  # Phase A overlaps with disk; combined_s already covers both
    return {
        "n_nvmes": n_threads,
        "total_gib": total_gib,
        "disk_s": disk_s,
        "phaseA_s": phaseA_s,
        "combined_s": combined_s,
        "total_s": total_s,
        "disk_gib_s": total_gib / disk_s if disk_s > 0 else float("nan"),
        "combined_gib_s": total_gib / combined_s if combined_s > 0 else float("nan"),
        "total_gib_s": total_gib / total_s if total_s > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-SSD GMS restore benchmark (exclusive per-SSD thread affinity)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-dir",
        default="/mnt/nvme2/gms_bench_save_72b",
        help="Directory for the initial (single-SSD) GMS save "
        "(default: /mnt/nvme2/gms_bench_save_72b)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="HuggingFace model for the save phase (used only if save is missing)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--skip-distribute",
        action="store_true",
        help="Skip copying shards to per-NVMe dirs (assume already done)",
    )
    parser.add_argument(
        "--no-drop-cache",
        action="store_true",
        help="Do NOT evict page cache before measuring (warm-cache run)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)

    # 1. Save phase (if save data is missing)
    if not (source_dir / "manifest.json").exists():
        logger.info(
            "=== Save phase: loading model and saving GMS state → %s ===", source_dir
        )
        run_save(source_dir, args.model, args.device)
    else:
        manifest = json.loads((source_dir / "manifest.json").read_text())
        total_gib = sum(a["aligned_size"] for a in manifest["allocations"]) / 1024**3
        logger.info(
            "Existing save found at %s  (%.2f GiB, %d allocs, %d shards)",
            source_dir,
            total_gib,
            len(manifest["allocations"]),
            len({a["tensor_file"] for a in manifest["allocations"]}),
        )

    # 2. Distribute shards across NVMes
    if not args.skip_distribute:
        distribute_shards(source_dir, NVME_DIRS)
    else:
        logger.info("--skip-distribute: using shards already on NVMes")

    # 3. Benchmark restore
    logger.info(
        "=== Multi-SSD restore benchmark: %d NVMes, 1 thread/NVMe, drop_cache=%s ===",
        len(NVME_DIRS),
        not args.no_drop_cache,
    )
    result = restore_multissd(
        source_dir,
        NVME_DIRS,
        device=args.device,
        drop_cache=not args.no_drop_cache,
    )

    # Results
    W = 60
    print()
    print("─" * W)
    print(
        f"  Multi-SSD GMS restore  ({result['n_nvmes']} NVMes, 1 exclusive thread/NVMe)"
    )
    print("─" * W)
    print(f"  Data     : {result['total_gib']:.2f} GiB")
    print(f"  Phase A  : {result['phaseA_s']:.3f}s  (overlapped with disk reads)")
    print(
        f"  Disk+B   : {result['combined_s']:.2f}s"
        f"   →  {result['combined_gib_s']:.2f} GiB/s  (disk+PhaseA+copy overlapped)"
    )
    print(
        f"    disk   : {result['disk_s']:.2f}s"
        f"   →  {result['disk_gib_s']:.2f} GiB/s  (aggregate across all NVMes)"
    )
    print(
        f"  Total    : {result['total_s']:.2f}s   →  {result['total_gib_s']:.2f} GiB/s"
    )
    print("─" * W)
    print()


if __name__ == "__main__":
    main()
