# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Populate or verify one committed GMS allocation for the CustomStorage probe."""

from __future__ import annotations

import argparse

import numpy as np
from cuda.bindings import driver as cuda
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm.cuda_utils import cuda_check_result

_METADATA_KEY = "custom_storage_probe.pattern"
_TAG = "weights"
_CHUNK_BYTES = 1024 * 1024


def _expected(offset: int, size: int) -> np.ndarray:
    positions = np.arange(
        offset,
        offset + size,
        dtype=np.uint64,
    )
    return ((positions * 1315423911) ^ (positions >> 7) ^ 0x5A).astype(np.uint8)


def _copy_to_device(pointer: int, data: np.ndarray) -> None:
    (result,) = cuda.cuMemcpyHtoD(pointer, data.ctypes.data, data.nbytes)
    cuda_check_result(result, "cuMemcpyHtoD")


def _copy_from_device(pointer: int, size: int) -> np.ndarray:
    data = np.empty(size, dtype=np.uint8)
    (result,) = cuda.cuMemcpyDtoH(data.ctypes.data, pointer, data.nbytes)
    cuda_check_result(result, "cuMemcpyDtoH")
    return data


def _write(socket_path: str, device: int, size: int) -> None:
    manager = GMSClientMemoryManager(socket_path, device=device, tag=_TAG)
    try:
        manager.connect(RequestedLockType.RW, timeout_ms=30_000)
        pointer = manager.create_mapping(size=size, tag=_TAG)
        allocation = manager.mappings[pointer]
        for offset in range(0, size, _CHUNK_BYTES):
            count = min(_CHUNK_BYTES, size - offset)
            _copy_to_device(pointer + offset, _expected(offset, count))
        manager.metadata_put(
            _METADATA_KEY,
            allocation.allocation_id,
            0,
            str(size).encode("ascii"),
        )
        manager.commit()
        print(
            "gms_write=passed "
            f"allocation_id={allocation.allocation_id} "
            f"layout_slot={allocation.layout_slot} bytes={size}",
            flush=True,
        )
    finally:
        manager.close()


def _verify(socket_path: str, device: int, size: int) -> None:
    manager = GMSClientMemoryManager(socket_path, device=device, tag=_TAG)
    try:
        manager.connect(RequestedLockType.RO, timeout_ms=30_000)
        allocations = manager.list_handles(_TAG)
        if len(allocations) != 1:
            raise RuntimeError(
                f"expected one committed GMS allocation, found {len(allocations)}"
            )
        allocation = allocations[0]
        metadata = manager.metadata_get(_METADATA_KEY)
        expected_metadata = (allocation.allocation_id, 0, str(size).encode("ascii"))
        if metadata != expected_metadata:
            raise RuntimeError(
                f"restored GMS metadata differs: {metadata!r} != {expected_metadata!r}"
            )
        layout_hash = manager.get_memory_layout_hash()
        if not layout_hash:
            raise RuntimeError("restored GMS layout hash is empty")

        pointer = manager.create_mapping(allocation_id=allocation.allocation_id)
        for offset in range(0, size, _CHUNK_BYTES):
            count = min(_CHUNK_BYTES, size - offset)
            expected = _expected(offset, count)
            actual = _copy_from_device(pointer + offset, count)
            mismatch = actual != expected
            if np.any(mismatch):
                first = int(np.flatnonzero(mismatch)[0])
                raise RuntimeError(
                    f"restored GMS bytes differ at offset {offset + first}"
                )

        print(
            "gms_verify=passed "
            f"allocation_id={allocation.allocation_id} "
            f"layout_slot={allocation.layout_slot} bytes={size} "
            f"layout_hash={layout_hash}",
            flush=True,
        )
    finally:
        manager.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    if args.bytes <= 0:
        parser.error("--bytes must be greater than zero")

    (result,) = cuda.cuInit(0)
    cuda_check_result(result, "cuInit")
    result, cuda_device = cuda.cuDeviceGet(args.device)
    cuda_check_result(result, "cuDeviceGet")
    result, context = cuda.cuDevicePrimaryCtxRetain(cuda_device)
    cuda_check_result(result, "cuDevicePrimaryCtxRetain")
    try:
        (result,) = cuda.cuCtxSetCurrent(context)
        cuda_check_result(result, "cuCtxSetCurrent")
        if args.mode == "write":
            _write(args.socket_path, args.device, args.bytes)
        else:
            _verify(args.socket_path, args.device, args.bytes)
    finally:
        (result,) = cuda.cuDevicePrimaryCtxRelease(cuda_device)
        cuda_check_result(result, "cuDevicePrimaryCtxRelease")


if __name__ == "__main__":
    main()
