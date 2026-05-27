#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark GMS restore Phase-A primitives.

This benchmark intentionally avoids model/vLLM/checkpoint work.  It connects to
an already-running GMS server in RW mode, allocates a synthetic layout, and times
exactly the primitives that dominate snapshot restore Phase A:

  * server-side allocate_many
  * per-allocation FD export over the GMS UDS
  * CUDA FD import
  * CUDA VA reservation
  * cuMemMap
  * cuMemSetAccess
  * metadata_put loop and commit tail

It can also run a loader-only VA-arena variant: reserve one contiguous VA range,
map each independent CUDA handle at successive offsets, and optionally probe a
single cuMemSetAccess over the whole arena.  This keeps one backing GMS/CUDA
allocation per logical allocation; it does not rely on unsupported partial
mapping of one backing allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from cuda.bindings import driver as cuda

from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.common.cuda_utils import (
    align_to_granularity,
    cuda_ensure_initialized,
    cuda_runtime_set_device,
    cuda_synchronize,
    cumem_address_free,
    cumem_address_reserve,
    cumem_get_allocation_granularity,
    cumem_import_from_shareable_handle_close_fd,
    cumem_map,
    cumem_release,
    cumem_set_access,
    cumem_unmap,
)
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path


@dataclass(frozen=True)
class StepTiming:
    name: str
    seconds: float
    count: int = 0
    bytes: int = 0
    extra: dict[str, object] | None = None


class Timer:
    def __init__(self, timings: list[StepTiming], name: str, *, count: int = 0, bytes: int = 0, extra: dict[str, object] | None = None):
        self.timings = timings
        self.name = name
        self.count = count
        self.bytes = bytes
        self.extra = extra
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timings.append(
            StepTiming(
                name=self.name,
                seconds=time.perf_counter() - self.t0,
                count=self.count,
                bytes=self.bytes,
                extra=self.extra,
            )
        )
        return False


def _cuda_error_string(result) -> str:
    err_result, err_str = cuda.cuGetErrorString(result)
    if err_result == cuda.CUresult.CUDA_SUCCESS and err_str:
        return err_str.decode() if isinstance(err_str, bytes) else str(err_str)
    return str(result)


def _cuda_check_nonfatal(result, name: str) -> tuple[bool, str]:
    if result == cuda.CUresult.CUDA_SUCCESS:
        return True, "success"
    return False, f"{name}: {_cuda_error_string(result)}"


def _raw_set_access(va: int, size: int, device: int, access: GrantedLockType) -> tuple[bool, str]:
    access_desc = cuda.CUmemAccessDesc()
    access_desc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access_desc.location.id = device
    access_desc.flags = (
        cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
        if access == GrantedLockType.RO
        else cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    )
    (result,) = cuda.cuMemSetAccess(va, size, [access_desc], 1)
    return _cuda_check_nonfatal(result, "cuMemSetAccess")


def _parse_total_bytes(value: str) -> int:
    s = value.strip().lower()
    suffixes = {
        "gib": 1 << 30,
        "gb": 10**9,
        "mib": 1 << 20,
        "mb": 10**6,
        "kib": 1 << 10,
        "kb": 10**3,
        "b": 1,
    }
    for suffix, scale in suffixes.items():
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * scale)
    return int(float(s))


def _sizes_from_total(count: int, total_bytes: int, granularity: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be > 0")
    total_granules = max(count, int(round(total_bytes / granularity)))
    base, remainder = divmod(total_granules, count)
    sizes = [base * granularity] * count
    for i in range(remainder):
        sizes[i] += granularity
    return sizes


def _sizes_from_manifest(path: str, granularity: int) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    allocations = manifest.get("allocations") or []
    if not allocations:
        raise ValueError(f"manifest {path!r} has no allocations")
    sizes: list[int] = []
    for entry in allocations:
        size = int(entry.get("aligned_size") or entry.get("size"))
        sizes.append(align_to_granularity(size, granularity))
    return sizes


def _summarize_sizes(sizes: Sequence[int]) -> dict[str, object]:
    gib = 1 << 30
    mib = 1 << 20
    return {
        "count": len(sizes),
        "total_bytes": sum(sizes),
        "total_gib": sum(sizes) / gib,
        "min_mib": min(sizes) / mib if sizes else 0,
        "max_mib": max(sizes) / mib if sizes else 0,
        "mean_mib": statistics.mean(sizes) / mib if sizes else 0,
        "median_mib": statistics.median(sizes) / mib if sizes else 0,
    }


def _close_fds(fds: Iterable[int]) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _cleanup_mappings(vas: Sequence[int], sizes: Sequence[int], handles: Sequence[int]) -> None:
    for va, size in zip(vas, sizes, strict=False):
        if va:
            try:
                cumem_unmap(va, size)
            except BaseException:
                pass
    for handle in handles:
        if handle:
            try:
                cumem_release(handle)
            except BaseException:
                pass
    for va, size in zip(vas, sizes, strict=False):
        if va:
            try:
                cumem_address_free(va, size)
            except BaseException:
                pass


def _cleanup_arena(arena_va: int, arena_size: int, offsets: Sequence[int], sizes: Sequence[int], handles: Sequence[int]) -> None:
    for offset, size in zip(offsets, sizes, strict=False):
        try:
            cumem_unmap(arena_va + offset, size)
        except BaseException:
            pass
    for handle in handles:
        if handle:
            try:
                cumem_release(handle)
            except BaseException:
                pass
    if arena_va:
        try:
            cumem_address_free(arena_va, arena_size)
        except BaseException:
            pass


def _allocate(mm: GMSClientMemoryManager, sizes: Sequence[int], timings: list[StepTiming]):
    with Timer(timings, "server_allocate_many", count=len(sizes), bytes=sum(sizes)):
        allocations = mm.allocate_handles([(size, "weights") for size in sizes])
    return allocations


def _metadata_put_loop(mm: GMSClientMemoryManager, allocations, count: int, timings: list[StepTiming]) -> None:
    payload = b'{"shape":[1],"dtype":"torch.float16","stride":[1],"tensor_type":"parameter"}'
    with Timer(timings, "metadata_put_loop", count=count, bytes=len(payload) * count):
        for i, alloc in enumerate(allocations[:count]):
            ok = mm.metadata_put(f"bench_tensor_{i}", alloc.allocation_id, 0, payload)
            if not ok:
                raise RuntimeError(f"metadata_put failed at index {i}")


def _run_per_allocation(mm: GMSClientMemoryManager, sizes: Sequence[int], *, metadata_count: int, timings: list[StepTiming]) -> None:
    allocations = _allocate(mm, sizes, timings)
    fds: list[int] = []
    vas: list[int] = []
    handles: list[int] = []
    try:
        with Timer(timings, "export_loop", count=len(allocations)):
            for alloc in allocations:
                fds.append(mm.export_handle(alloc.allocation_id))

        with Timer(timings, "reserve_va_loop", count=len(sizes), bytes=sum(sizes)):
            for size in sizes:
                vas.append(mm.reserve_va(size))

        with Timer(timings, "cuda_import_loop", count=len(fds)):
            for fd in fds:
                handles.append(cumem_import_from_shareable_handle_close_fd(fd))
            fds.clear()

        with Timer(timings, "cu_mem_map_loop", count=len(handles), bytes=sum(sizes)):
            for va, size, handle in zip(vas, sizes, handles, strict=True):
                cumem_map(va, size, handle)

        with Timer(timings, "cu_mem_set_access_loop", count=len(handles), bytes=sum(sizes)):
            for va, size in zip(vas, sizes, strict=True):
                cumem_set_access(va, size, mm.device, GrantedLockType.RW)

        if metadata_count:
            _metadata_put_loop(mm, allocations, min(metadata_count, len(allocations)), timings)

        with Timer(timings, "local_unmap_release_free_loop", count=len(handles), bytes=sum(sizes)):
            _cleanup_mappings(vas, sizes, handles)
            vas.clear()
            handles.clear()

        with Timer(timings, "commit", count=len(allocations), bytes=sum(sizes)):
            mm.commit()
    finally:
        _close_fds(fds)
        _cleanup_mappings(vas, sizes, handles)


def _run_arena(mm: GMSClientMemoryManager, sizes: Sequence[int], *, one_set_access: bool, metadata_count: int, timings: list[StepTiming]) -> None:
    allocations = _allocate(mm, sizes, timings)
    fds: list[int] = []
    handles: list[int] = []
    arena_va = 0
    offsets: list[int] = []
    total = sum(sizes)
    pos = 0
    for size in sizes:
        offsets.append(pos)
        pos += size
    try:
        with Timer(timings, "export_loop", count=len(allocations)):
            for alloc in allocations:
                fds.append(mm.export_handle(alloc.allocation_id))

        with Timer(timings, "arena_reserve_va", count=1, bytes=total):
            arena_va = cumem_address_reserve(total, mm.granularity)

        with Timer(timings, "cuda_import_loop", count=len(fds)):
            for fd in fds:
                handles.append(cumem_import_from_shareable_handle_close_fd(fd))
            fds.clear()

        with Timer(timings, "cu_mem_map_loop", count=len(handles), bytes=total):
            for offset, size, handle in zip(offsets, sizes, handles, strict=True):
                cumem_map(arena_va + offset, size, handle)

        if one_set_access:
            ok = False
            detail = ""
            with Timer(timings, "arena_cu_mem_set_access_once", count=1, bytes=total):
                ok, detail = _raw_set_access(arena_va, total, mm.device, GrantedLockType.RW)
            timings.append(
                StepTiming(
                    name="arena_cu_mem_set_access_once_result",
                    seconds=0.0,
                    count=1,
                    bytes=total,
                    extra={"ok": ok, "detail": detail},
                )
            )
            if not ok:
                with Timer(timings, "cu_mem_set_access_loop_after_once_failed", count=len(handles), bytes=total):
                    for offset, size in zip(offsets, sizes, strict=True):
                        cumem_set_access(arena_va + offset, size, mm.device, GrantedLockType.RW)
        else:
            with Timer(timings, "cu_mem_set_access_loop", count=len(handles), bytes=total):
                for offset, size in zip(offsets, sizes, strict=True):
                    cumem_set_access(arena_va + offset, size, mm.device, GrantedLockType.RW)

        if metadata_count:
            _metadata_put_loop(mm, allocations, min(metadata_count, len(allocations)), timings)

        with Timer(timings, "local_unmap_release_arena_free", count=len(handles), bytes=total):
            _cleanup_arena(arena_va, total, offsets, sizes, handles)
            arena_va = 0
            handles.clear()

        with Timer(timings, "commit", count=len(allocations), bytes=total):
            mm.commit()
    finally:
        _close_fds(fds)
        if arena_va:
            _cleanup_arena(arena_va, total, offsets, sizes, handles)


def run_once(args: argparse.Namespace, variant: str, sizes: Sequence[int], iteration: int) -> dict[str, object]:
    socket_path = args.socket_path or get_socket_path(args.device, "weights")
    timings: list[StepTiming] = []
    mm = GMSClientMemoryManager(socket_path, device=args.device, tag="weights")
    connected = False
    try:
        with Timer(timings, "connect_rw"):
            mm.connect(RequestedLockType.RW, timeout_ms=args.timeout_ms)
            connected = True
        if variant == "per-allocation":
            _run_per_allocation(mm, sizes, metadata_count=args.metadata_count, timings=timings)
            connected = False  # commit closed it
        elif variant == "arena-loop-access":
            _run_arena(mm, sizes, one_set_access=False, metadata_count=args.metadata_count, timings=timings)
            connected = False
        elif variant == "arena-once-access":
            _run_arena(mm, sizes, one_set_access=True, metadata_count=args.metadata_count, timings=timings)
            connected = False
        else:
            raise ValueError(f"unknown variant: {variant}")
    finally:
        if connected:
            try:
                mm.abort()
            except Exception:
                pass

    return {
        "variant": variant,
        "iteration": iteration,
        "socket_path": socket_path,
        "device": args.device,
        "sizes": _summarize_sizes(sizes),
        "timings": [asdict(t) for t in timings],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--socket-path", default=None)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--count", type=int, default=3435)
    parser.add_argument("--total-bytes", default="134.28GiB")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--metadata-count", type=int, default=723)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--variant",
        action="append",
        choices=("per-allocation", "arena-loop-access", "arena-once-access"),
        help="Variant to run. May be passed multiple times. Default: per-allocation.",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    cuda_ensure_initialized()
    cuda_runtime_set_device(args.device)
    granularity = cumem_get_allocation_granularity(args.device)
    if args.manifest:
        sizes = _sizes_from_manifest(args.manifest, granularity)
    else:
        sizes = _sizes_from_total(args.count, _parse_total_bytes(args.total_bytes), granularity)

    variants = args.variant or ["per-allocation"]
    results: list[dict[str, object]] = []
    print(
        json.dumps(
            {
                "event": "config",
                "device": args.device,
                "granularity": granularity,
                "variants": variants,
                "iterations": args.iterations,
                "sizes": _summarize_sizes(sizes),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for iteration in range(args.iterations):
        for variant in variants:
            result = run_once(args, variant, sizes, iteration)
            results.append(result)
            print(json.dumps({"event": "result", **result}, sort_keys=True), flush=True)

    payload = {"results": results}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
