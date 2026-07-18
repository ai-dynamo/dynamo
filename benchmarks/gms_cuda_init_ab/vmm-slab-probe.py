# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate-process CUDA VMM slab-count scaling probe."""

from __future__ import annotations

import argparse
import array
import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cuda_runtime

from gpu_memory_service.common.cuda_utils import (
    cuda_check_result,
    cuda_runtime_check_result,
)

_HEADER = struct.Struct("!I")
_MAX_FDS_PER_MESSAGE = 128


def _stamp() -> tuple[int, int]:
    return time.time_ns(), time.monotonic_ns()


def _timed(phases: dict[str, dict[str, int]], name: str, operation):
    wall_start_ns, monotonic_start_ns = _stamp()
    result = operation()
    wall_end_ns, monotonic_end_ns = _stamp()
    phases[name] = {
        "wall_start_ns": wall_start_ns,
        "wall_end_ns": wall_end_ns,
        "duration_ns": monotonic_end_ns - monotonic_start_ns,
    }
    return result


def _allocation_property() -> cuda.CUmemAllocationProp:
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0
    prop.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    return prop


def _cu_init() -> None:
    (result,) = cuda.cuInit(0)
    cuda_check_result(result, "cuInit")


def _runtime_set_device() -> None:
    cuda_runtime_check_result(cuda_runtime.cudaSetDevice(0), "cudaSetDevice(0)")


def _create(size: int, prop: cuda.CUmemAllocationProp) -> int:
    result, handle = cuda.cuMemCreate(size, prop, 0)
    cuda_check_result(result, "cuMemCreate")
    return int(handle)


def _export(handle: int) -> int:
    result, fd = cuda.cuMemExportToShareableHandle(
        handle,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        0,
    )
    cuda_check_result(result, "cuMemExportToShareableHandle")
    return int(fd)


def _import(fd: int) -> int:
    result, handle = cuda.cuMemImportFromShareableHandle(
        fd,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    )
    cuda_check_result(result, "cuMemImportFromShareableHandle")
    return int(handle)


def _reserve(size: int, granularity: int) -> int:
    result, va = cuda.cuMemAddressReserve(size, granularity, 0, 0)
    cuda_check_result(result, "cuMemAddressReserve")
    return int(va)


def _map(va: int, size: int, handle: int) -> None:
    (result,) = cuda.cuMemMap(va, size, 0, handle, 0)
    cuda_check_result(result, "cuMemMap")


def _set_access(va: int, size: int) -> None:
    desc = cuda.CUmemAccessDesc()
    desc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = 0
    desc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    (result,) = cuda.cuMemSetAccess(va, size, [desc], 1)
    cuda_check_result(result, "cuMemSetAccess")


def _unmap(va: int, size: int) -> None:
    (result,) = cuda.cuMemUnmap(va, size)
    cuda_check_result(result, "cuMemUnmap")


def _release(handle: int) -> None:
    (result,) = cuda.cuMemRelease(handle)
    cuda_check_result(result, "cuMemRelease")


def _free_address(va: int, size: int) -> None:
    (result,) = cuda.cuMemAddressFree(va, size)
    cuda_check_result(result, "cuMemAddressFree")


def _send_packet(sock: socket.socket, payload: dict[str, Any], fds: list[int]) -> None:
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    ancillary = []
    if fds:
        ancillary.append(
            (
                socket.SOL_SOCKET,
                socket.SCM_RIGHTS,
                array.array("i", fds).tobytes(),
            )
        )
    sock.sendmsg([_HEADER.pack(len(encoded)), encoded], ancillary)


def _recv_packet(sock: socket.socket) -> tuple[dict[str, Any], list[int]]:
    data, ancillary, _flags, _address = sock.recvmsg(
        1 << 20,
        socket.CMSG_SPACE(_MAX_FDS_PER_MESSAGE * array.array("i").itemsize),
    )
    if len(data) < _HEADER.size:
        while len(data) < _HEADER.size:
            chunk = sock.recv(_HEADER.size - len(data))
            if not chunk:
                raise RuntimeError("truncated VMM probe packet header")
            data += chunk
    (payload_size,) = _HEADER.unpack(data[: _HEADER.size])
    payload = data[_HEADER.size :]
    while len(payload) < payload_size:
        chunk = sock.recv(payload_size - len(payload))
        if not chunk:
            raise RuntimeError("truncated VMM probe packet payload")
        payload += chunk
    fds = []
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            received = array.array("i")
            received.frombytes(value[: len(value) - (len(value) % received.itemsize)])
            fds.extend(received)
    return json.loads(payload[:payload_size]), fds


def _coalesce_sizes(logical_sizes: list[int], physical_count: int) -> list[int]:
    if not 1 <= physical_count <= len(logical_sizes):
        raise ValueError(
            "physical_count must be between 1 and logical allocation count"
        )
    base, extra = divmod(len(logical_sizes), physical_count)
    sizes = []
    offset = 0
    for group in range(physical_count):
        logical_count = base + (1 if group < extra else 0)
        sizes.append(sum(logical_sizes[offset : offset + logical_count]))
        offset += logical_count
    return sizes


def _write_result(path: str, result: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def exporter(args: argparse.Namespace) -> None:
    distribution = json.loads(Path(args.sizes).read_text(encoding="utf-8"))
    logical_sizes = [
        int(entry["size"]) for entry in distribution for _ in range(int(entry["count"]))
    ]
    physical_sizes = _coalesce_sizes(logical_sizes, args.physical_count)
    phases: dict[str, dict[str, int]] = {}
    handles: list[int] = []
    export_fds: list[int] = []
    listen_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    Path(args.socket).unlink(missing_ok=True)
    try:
        _timed(phases, "cu_init", _cu_init)
        prop = _allocation_property()
        wall_start_ns, monotonic_start_ns = _stamp()
        for size in physical_sizes:
            handles.append(_create(size, prop))
        wall_end_ns, monotonic_end_ns = _stamp()
        phases["cu_mem_create"] = {
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "duration_ns": monotonic_end_ns - monotonic_start_ns,
        }
        wall_start_ns, monotonic_start_ns = _stamp()
        for handle in handles:
            export_fds.append(_export(handle))
        wall_end_ns, monotonic_end_ns = _stamp()
        phases["cu_mem_export"] = {
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "duration_ns": monotonic_end_ns - monotonic_start_ns,
        }
        listen_socket.bind(args.socket)
        listen_socket.listen(1)
        connection, _ = listen_socket.accept()
        with connection:
            wall_start_ns, monotonic_start_ns = _stamp()
            for first in range(0, len(export_fds), _MAX_FDS_PER_MESSAGE):
                batch_fds = export_fds[first : first + _MAX_FDS_PER_MESSAGE]
                _send_packet(
                    connection,
                    {
                        "first": first,
                        "sizes": physical_sizes[first : first + len(batch_fds)],
                        "total_count": len(export_fds),
                    },
                    batch_fds,
                )
            wall_end_ns, monotonic_end_ns = _stamp()
            phases["ipc_send"] = {
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "duration_ns": monotonic_end_ns - monotonic_start_ns,
            }
            acknowledgement, acknowledgement_fds = _recv_packet(connection)
            if acknowledgement != {"done": True} or acknowledgement_fds:
                raise RuntimeError("invalid importer acknowledgement")
    finally:
        listen_socket.close()
        Path(args.socket).unlink(missing_ok=True)
        for fd in export_fds:
            os.close(fd)
        wall_start_ns, monotonic_start_ns = _stamp()
        for handle in handles:
            _release(handle)
        wall_end_ns, monotonic_end_ns = _stamp()
        phases["server_release"] = {
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "duration_ns": monotonic_end_ns - monotonic_start_ns,
        }
    _write_result(
        args.result,
        {
            "role": "exporter",
            "device": args.device,
            "physical_count": len(physical_sizes),
            "logical_count": len(logical_sizes),
            "bytes": sum(physical_sizes),
            "phases": phases,
        },
    )


def importer(args: argparse.Namespace) -> None:
    phases: dict[str, dict[str, int]] = {}
    received_fds: list[int] = []
    sizes: list[int] = []
    handles: list[int] = []
    va = 0
    total_size = 0
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        _timed(phases, "cuda_set_device", _runtime_set_device)
        deadline = time.monotonic() + 120
        while True:
            try:
                connection.connect(args.socket)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"exporter socket not ready: {args.socket}")
                time.sleep(0.01)
        wall_start_ns, monotonic_start_ns = _stamp()
        expected_count = None
        while expected_count is None or len(received_fds) < expected_count:
            payload, batch_fds = _recv_packet(connection)
            if payload["first"] != len(received_fds):
                raise RuntimeError("out-of-order VMM probe FD batch")
            expected_count = int(payload["total_count"])
            batch_sizes = [int(size) for size in payload["sizes"]]
            if len(batch_sizes) != len(batch_fds):
                raise RuntimeError("VMM probe FD/size count mismatch")
            sizes.extend(batch_sizes)
            received_fds.extend(batch_fds)
        wall_end_ns, monotonic_end_ns = _stamp()
        phases["ipc_receive"] = {
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "duration_ns": monotonic_end_ns - monotonic_start_ns,
        }
        total_size = sum(sizes)
        wall_start_ns, monotonic_start_ns = _stamp()
        for fd in received_fds:
            handles.append(_import(fd))
            os.close(fd)
        received_fds.clear()
        wall_end_ns, monotonic_end_ns = _stamp()
        phases["cu_mem_import"] = {
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "duration_ns": monotonic_end_ns - monotonic_start_ns,
        }
        prop = _allocation_property()
        result, granularity = cuda.cuMemGetAllocationGranularity(
            prop,
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        cuda_check_result(result, "cuMemGetAllocationGranularity")
        va = _timed(
            phases,
            "cu_mem_address_reserve",
            lambda: _reserve(total_size, int(granularity)),
        )
        wall_start_ns, monotonic_start_ns = _stamp()
        offset = 0
        for size, handle in zip(sizes, handles):
            _map(va + offset, size, handle)
            offset += size
        wall_end_ns, monotonic_end_ns = _stamp()
        phases["cu_mem_map"] = {
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "duration_ns": monotonic_end_ns - monotonic_start_ns,
        }
        _timed(phases, "cu_mem_set_access", lambda: _set_access(va, total_size))
        _send_packet(connection, {"done": True}, [])
    finally:
        connection.close()
        for fd in received_fds:
            os.close(fd)
        if va:
            _timed(phases, "cu_mem_unmap", lambda: _unmap(va, total_size))
        wall_start_ns, monotonic_start_ns = _stamp()
        for handle in handles:
            _release(handle)
        wall_end_ns, monotonic_end_ns = _stamp()
        phases["client_release"] = {
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "duration_ns": monotonic_end_ns - monotonic_start_ns,
        }
        if va:
            _timed(
                phases,
                "cu_mem_address_free",
                lambda: _free_address(va, total_size),
            )
    _write_result(
        args.result,
        {
            "role": "importer",
            "device": args.device,
            "physical_count": len(sizes),
            "bytes": total_size,
            "phases": phases,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["exporter", "importer"])
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--physical-count", type=int, required=True)
    parser.add_argument("--sizes", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--result", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.role == "exporter":
        exporter(args)
    else:
        importer(args)


if __name__ == "__main__":
    main()
