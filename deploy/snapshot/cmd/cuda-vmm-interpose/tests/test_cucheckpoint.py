# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing.reduction import recv_handle, send_handle
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cuda.bindings import driver

WORLD_SIZE = 2
NUMEL = 2048
COMMAND_TIMEOUT_SECONDS = 60
CHECKPOINT_TIMEOUT_SECONDS = 120
WORKER_TIMEOUT_SECONDS = 240
PARENT_PARTICIPANT_ID = "a" * 32
LOGICAL_HANDLE_TAG = 0xD94D000000000000
LOGICAL_HANDLE_TAG_MASK = 0xFFFF000000000000
POSIX_FD_HANDLE_TYPE = (
    driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
)


class _ExternalAllocation(NamedTuple):
    device: driver.CUdevice
    context: driver.CUcontext
    address: int
    size: int
    handle: driver.CUmemGenericAllocationHandle
    fd: int


def _cuda_call(function, *arguments):
    status, *outputs = function(*arguments)
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{function.__name__} failed: {status.name} ({int(status)})")
    if not outputs:
        return None
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def _assert_no_current_context(process: str) -> None:
    status, context = driver.cuCtxGetCurrent()
    if status == driver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        return
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed: {status.name} ({int(status)})")
    if int(context) != 0:
        raise AssertionError(f"{process} has a current CUDA context")


def _allocation_properties(device) -> driver.CUmemAllocationProp:
    properties = driver.CUmemAllocationProp()
    properties.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    properties.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    properties.location.id = int(device)
    properties.requestedHandleTypes = POSIX_FD_HANDLE_TYPE
    return properties


def _allocation_size(properties: driver.CUmemAllocationProp) -> int:
    return int(
        _cuda_call(
            driver.cuMemGetAllocationGranularity,
            properties,
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    )


def _assert_handle_namespace(handle, logical: bool, stage: str) -> None:
    tagged = int(handle) & LOGICAL_HANDLE_TAG_MASK == LOGICAL_HANDLE_TAG
    if tagged != logical:
        expected = "logical" if logical else "raw"
        raise AssertionError(f"{stage}: handle {int(handle):#x} is not {expected}")


def _map_allocation(handle, size: int, device) -> int:
    address = int(_cuda_call(driver.cuMemAddressReserve, size, size, 0, 0))
    mapped = False
    try:
        _cuda_call(driver.cuMemMap, address, size, 0, handle, 0)
        mapped = True
        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = int(device)
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _cuda_call(driver.cuMemSetAccess, address, size, [access], 1)
    except Exception:
        if mapped:
            _cuda_call(driver.cuMemUnmap, address, size)
        _cuda_call(driver.cuMemAddressFree, address, size)
        raise
    return address


def _write_bytes(address: int, expected: bytes) -> None:
    source = ctypes.create_string_buffer(expected)
    _cuda_call(driver.cuMemcpyHtoD, address, ctypes.addressof(source), len(expected))


def _assert_bytes(address: int, expected: bytes, stage: str) -> None:
    actual = ctypes.create_string_buffer(len(expected))
    _cuda_call(driver.cuMemcpyDtoH, ctypes.addressof(actual), address, len(expected))
    if actual.raw != expected:
        raise AssertionError(f"{stage}: got {actual.raw!r}, expected {expected!r}")


def _destroy_mapped_allocation(address: int, size: int, handle) -> None:
    _cuda_call(driver.cuMemUnmap, address, size)
    _cuda_call(driver.cuMemAddressFree, address, size)
    _cuda_call(driver.cuMemRelease, handle)


def _worker(
    rank: int,
    raw_fds: tuple[int, int],
    restore_fds: tuple[int, int],
    sync_dir: Path,
    store_path: Path,
) -> None:
    _cuda_call(driver.cuInit, 0)
    device = _cuda_call(driver.cuDeviceGet, rank)
    properties = _allocation_properties(device)
    private_size = _allocation_size(properties)
    _assert_no_current_context("worker before private cuMemCreate")
    private_handle = _cuda_call(driver.cuMemCreate, private_size, properties, 0)
    _assert_handle_namespace(private_handle, True, "managed cuMemCreate")

    torch.cuda.set_device(rank)
    for other_rank, fd in enumerate(raw_fds):
        if other_rank != rank:
            os.close(fd)
    for other_rank, fd in enumerate(restore_fds):
        if other_rank != rank:
            os.close(fd)
    restore_socket = socket.socket(fileno=restore_fds[rank])
    raw_fd = raw_fds[rank]
    external_handle = None
    external_address = 0
    try:
        external_handle = _cuda_call(
            driver.cuMemImportFromShareableHandle,
            raw_fd,
            POSIX_FD_HANDLE_TYPE,
        )
        _assert_handle_namespace(external_handle, False, "raw external import")
    finally:
        os.close(raw_fd)
    try:
        external_address = _map_allocation(external_handle, private_size, device)
        _assert_bytes(
            external_address,
            bytes([rank + 1]) * 32,
            "raw external allocation",
        )
    finally:
        if external_address:
            _destroy_mapped_allocation(
                external_address,
                private_size,
                external_handle,
            )
        elif external_handle is not None:
            _cuda_call(driver.cuMemRelease, external_handle)

    private_address = _map_allocation(private_handle, private_size, device)
    retained_handle = _cuda_call(
        driver.cuMemRetainAllocationHandle,
        private_address,
    )
    _assert_handle_namespace(retained_handle, True, "managed retain")
    _cuda_call(driver.cuMemRelease, retained_handle)
    private_expected = bytes([0xA0 + rank]) * 32
    _write_bytes(private_address, private_expected)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    group_name = dist.group.WORLD.group_name
    input_tensor = symm_mem.empty(NUMEL, dtype=torch.float32, device="cuda")
    input_tensor.fill_(rank + 1)
    symm_handle = symm_mem.rendezvous(input_tensor, group=group_name)
    output = torch.empty_like(input_tensor)

    torch.ops.symm_mem.one_shot_all_reduce_out(input_tensor, "sum", group_name, output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.symm_mem.one_shot_all_reduce_out(
            input_tensor, "sum", group_name, output
        )
    graph.replay()
    torch.cuda.synchronize()
    _assert_exact_result(output, "before checkpoint")
    (sync_dir / f"ready-{rank}").touch()

    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    while not (sync_dir / "continue").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for restore")
        time.sleep(0.05)

    fresh_fd = recv_handle(restore_socket)
    restore_socket.close()
    fresh_handle = None
    fresh_address = 0
    try:
        fresh_handle = _cuda_call(
            driver.cuMemImportFromShareableHandle,
            fresh_fd,
            POSIX_FD_HANDLE_TYPE,
        )
        _assert_handle_namespace(fresh_handle, False, "fresh raw import after restore")
    finally:
        os.close(fresh_fd)
    try:
        fresh_address = _map_allocation(fresh_handle, private_size, device)
        _assert_bytes(
            fresh_address,
            bytes([0x40 + rank]) * 32,
            "fresh raw allocation after restore",
        )
    finally:
        if fresh_address:
            _destroy_mapped_allocation(
                fresh_address,
                private_size,
                fresh_handle,
            )
        elif fresh_handle is not None:
            _cuda_call(driver.cuMemRelease, fresh_handle)

    _assert_bytes(private_address, private_expected, "private allocation after restore")
    graph.replay()
    torch.cuda.synchronize()
    _assert_exact_result(output, "after restore")
    (sync_dir / f"done-{rank}").touch()

    dist.barrier()
    del graph, output, symm_handle, input_tensor
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    _destroy_mapped_allocation(private_address, private_size, private_handle)


def _assert_exact_result(output: torch.Tensor, stage: str) -> None:
    expected = torch.full((NUMEL,), 3.0, dtype=torch.float32)
    actual = output.cpu()
    if not torch.equal(actual, expected):
        mismatch = torch.nonzero(actual != expected)[0].item()
        raise AssertionError(
            f"{stage}: output[{mismatch}] is {actual[mismatch].item()}, expected 3.0"
        )


def _visible_gpus() -> tuple[str, str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        return "0", "1"
    devices = [entry.strip() for entry in configured.split(",") if entry.strip()]
    if len(devices) < WORLD_SIZE:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain two GPUs")
    if devices[0] == devices[1]:
        raise RuntimeError("the first two CUDA_VISIBLE_DEVICES entries must differ")
    return devices[0], devices[1]


def _create_external_allocations(byte_base: int) -> list[_ExternalAllocation]:
    _cuda_call(driver.cuInit, 0)
    allocations = []
    try:
        for rank in range(WORLD_SIZE):
            allocations.append(_create_external_allocation(rank, byte_base))
    except Exception:
        _destroy_external_allocations(allocations)
        raise
    return allocations


def _create_external_allocation(rank: int, byte_base: int) -> _ExternalAllocation:
    device = _cuda_call(driver.cuDeviceGet, rank)
    context = _cuda_call(driver.cuDevicePrimaryCtxRetain, device)
    handle = None
    address = 0
    try:
        _cuda_call(driver.cuCtxPushCurrent, context)
        try:
            properties = _allocation_properties(device)
            size = _allocation_size(properties)
            handle = _cuda_call(driver.cuMemCreate, size, properties, 0)
            address = _map_allocation(handle, size, device)
            _write_bytes(address, bytes([byte_base + rank]) * 32)
            fd = int(
                _cuda_call(
                    driver.cuMemExportToShareableHandle,
                    handle,
                    POSIX_FD_HANDLE_TYPE,
                    0,
                )
            )
        except Exception:
            if address:
                _destroy_mapped_allocation(address, size, handle)
            elif handle is not None:
                _cuda_call(driver.cuMemRelease, handle)
            raise
        finally:
            _cuda_call(driver.cuCtxPopCurrent)
    except Exception:
        _cuda_call(driver.cuDevicePrimaryCtxRelease, device)
        raise
    return _ExternalAllocation(device, context, address, size, handle, fd)


def _destroy_external_allocations(allocations: list[_ExternalAllocation]) -> None:
    while allocations:
        allocation = allocations.pop()
        os.close(allocation.fd)
        _cuda_call(driver.cuCtxPushCurrent, allocation.context)
        try:
            _destroy_mapped_allocation(
                allocation.address,
                allocation.size,
                allocation.handle,
            )
        finally:
            _cuda_call(driver.cuCtxPopCurrent)
            _cuda_call(driver.cuDevicePrimaryCtxRelease, allocation.device)


def _build_native_tools(tmp_path: Path) -> tuple[Path, Path]:
    interposer_dir = Path(__file__).resolve().parents[1]
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    compiler = os.environ.get("CC", "cc")
    missing = []
    if shutil.which("make") is None:
        missing.append("make on PATH")
    if shutil.which("readelf") is None:
        missing.append("readelf on PATH")
    if shutil.which(compiler) is None:
        missing.append(f"{compiler} on PATH")
    cuda_header = cuda_home / "include" / "cuda.h"
    if not cuda_header.is_file():
        missing.append(str(cuda_header))
    if missing:
        pytest.fail(
            "missing native build prerequisites: "
            f"{', '.join(missing)}; provide make, readelf, a C compiler, and "
            "CUDA headers, or set CUDA_HOME to the CUDA toolkit root"
        )
    build_dir = tmp_path / "native"
    result = subprocess.run(
        [
            "make",
            "-C",
            str(interposer_dir),
            f"BUILD_DIR={build_dir}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        pytest.fail(
            f"native build failed ({result.returncode}):\n"
            f"{result.stdout}{result.stderr}"
        )
    interposer = (build_dir / "libsnapshot_cuda_vmm.so").resolve()
    coordinator = (build_dir / "snapshot-cuda-vmm").resolve()
    if not interposer.is_file() or not coordinator.is_file():
        pytest.fail("native build did not produce the interposer and coordinator")
    return interposer, coordinator


def _fork_workers(
    raw_fds: tuple[int, int],
    restore_fds: tuple[int, int],
    sync_dir: Path,
    store_path: Path,
) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("parent initialized CUDA before forking workers")
    _assert_no_current_context("parent before forking workers")

    children = []
    for rank in range(WORLD_SIZE):
        child = os.fork()
        if child == 0:
            try:
                _worker(rank, raw_fds, restore_fds, sync_dir, store_path)
            except BaseException:  # noqa: BLE001 -- report child failures to parent
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        children.append((rank, child))
        (sync_dir / f"pid-{rank}").write_text(f"{child}\n")

    for fd in raw_fds:
        os.close(fd)
    for fd in restore_fds:
        os.close(fd)

    remaining = dict(children)
    while remaining:
        failures = []
        for rank, child in tuple(remaining.items()):
            waited, status = os.waitpid(child, os.WNOHANG)
            if waited == 0:
                continue
            del remaining[rank]
            exit_code = os.waitstatus_to_exitcode(status)
            if not os.WIFEXITED(status) or exit_code != 0:
                failures.append(f"rank {rank} PID {child}: {exit_code}")
        if failures:
            for child in remaining.values():
                try:
                    os.kill(child, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for child in remaining.values():
                os.waitpid(child, 0)
            raise RuntimeError(f"forked workers failed: {', '.join(failures)}")
        time.sleep(0.05)


def _start_parent(
    gpus: tuple[str, str],
    interposer: Path,
    control_dir: Path,
    sync_dir: Path,
    store_path: Path,
    raw_fds: tuple[int, int],
    restore_fds: tuple[int, int],
) -> subprocess.Popen[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpus),
            "DYN_SNAPSHOT_CUDA_VMM_INTERPOSE": "1",
            "DYN_SNAPSHOT_PARTICIPANT_ID": PARENT_PARTICIPANT_ID,
            "DYN_SNAPSHOT_CONTROL_DIR": str(control_dir),
            "LD_PRELOAD": str(interposer),
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "TORCH_SYMMEM_IMPLICIT_POOL": "0",
            "TORCH_SYMM_MEM_DISABLE_MULTICAST": "1",
        }
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-u",
            str(Path(__file__).resolve()),
            "--parent",
            *(str(fd) for fd in raw_fds),
            *(str(fd) for fd in restore_fds),
            str(sync_dir),
            str(store_path),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=raw_fds + restore_fds,
    )


def _wait_for_child_pids(
    parent: subprocess.Popen[str], sync_dir: Path
) -> tuple[int, int]:
    paths = [sync_dir / f"pid-{rank}" for rank in range(WORLD_SIZE)]
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    while True:
        try:
            pids = tuple(int(path.read_text()) for path in paths)
        except (FileNotFoundError, ValueError):
            pids = ()
        if len(pids) == WORLD_SIZE:
            if len(set(pids)) != WORLD_SIZE:
                raise AssertionError(f"forked worker PIDs are not unique: {pids}")
            return pids
        if parent.poll() is not None:
            raise RuntimeError(
                f"parent {parent.pid} exited before publishing child PIDs "
                f"with {parent.returncode}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for forked worker PIDs")
        time.sleep(0.05)


def _wait_for_workers(
    parent: subprocess.Popen[str],
    sync_dir: Path,
    marker: str,
) -> None:
    expected = [sync_dir / f"{marker}-{rank}" for rank in range(WORLD_SIZE)]
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    while not all(path.exists() for path in expected):
        if parent.poll() is not None:
            raise RuntimeError(
                f"parent {parent.pid} exited before workers reached {marker} "
                f"with {parent.returncode}"
            )
        if time.monotonic() >= deadline:
            missing = [str(path) for path in expected if not path.exists()]
            raise TimeoutError(f"timed out waiting for {marker}: {', '.join(missing)}")
        time.sleep(0.05)


def _assert_worker_runtime(
    process_id: int, interposer: Path, control_dir: Path
) -> None:
    maps = Path(f"/proc/{process_id}/maps").read_text().splitlines()
    mapped_paths = {
        fields[5] for line in maps if len(fields := line.split(maxsplit=5)) == 6
    }
    if str(interposer) not in mapped_paths:
        raise AssertionError(f"{interposer} is not loaded in process {process_id}")
    endpoint = control_dir / f"cuda-vmm-{process_id}.sock"
    if not endpoint.is_socket():
        raise AssertionError(f"interposer endpoint does not exist: {endpoint}")


def _run_coordinator(
    coordinator: Path,
    operation: str,
    checkpoint_dir: Path,
    control_dir: Path,
    process_ids: tuple[int, int],
) -> None:
    command = [
        str(coordinator),
        operation,
        "--proc-root",
        "",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    for process_id in process_ids:
        command.extend(["--process", str(process_id), str(process_id)])
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    environment.pop("DYN_SNAPSHOT_CUDA_VMM_INTERPOSE", None)
    environment["DYN_SNAPSHOT_CONTROL_DIR"] = str(control_dir)
    result = subprocess.run(
        command,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"coordinator {operation} failed ({result.returncode}):\n"
            f"{result.stdout}{result.stderr}"
        )


def _expect_state(process_id: int, expected) -> None:
    actual = _cuda_call(driver.cuCheckpointProcessGetState, process_id)
    if actual != expected:
        raise AssertionError(
            f"CUDA process {process_id} is {actual.name}, expected {expected.name}"
        )


def _native_checkpoint(process_ids: tuple[int, int]) -> None:
    _cuda_call(driver.cuInit, 0)
    running = driver.CUprocessState.CU_PROCESS_STATE_RUNNING
    locked = driver.CUprocessState.CU_PROCESS_STATE_LOCKED
    checkpointed = driver.CUprocessState.CU_PROCESS_STATE_CHECKPOINTED
    for process_id in process_ids:
        _expect_state(process_id, running)

    lock_arguments = driver.CUcheckpointLockArgs()
    lock_arguments.timeoutMs = COMMAND_TIMEOUT_SECONDS * 1000
    for process_id in process_ids:
        _cuda_call(driver.cuCheckpointProcessLock, process_id, lock_arguments)
    for process_id in process_ids:
        _expect_state(process_id, locked)

    checkpoint_arguments = driver.CUcheckpointCheckpointArgs()
    for process_id in process_ids:
        _cuda_call(
            driver.cuCheckpointProcessCheckpoint,
            process_id,
            checkpoint_arguments,
        )
    for process_id in process_ids:
        _expect_state(process_id, checkpointed)

    restore_arguments = driver.CUcheckpointRestoreArgs()
    for process_id in process_ids:
        _cuda_call(driver.cuCheckpointProcessRestore, process_id, restore_arguments)
    for process_id in process_ids:
        _expect_state(process_id, locked)

    unlock_arguments = driver.CUcheckpointUnlockArgs()
    for process_id in process_ids:
        _cuda_call(driver.cuCheckpointProcessUnlock, process_id, unlock_arguments)
    for process_id in process_ids:
        _expect_state(process_id, running)


def _native_checkpoint_with_timeout(
    process_ids: tuple[int, int],
) -> None:
    outcomes: queue.Queue[Exception | None] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            _native_checkpoint(process_ids)
        except Exception as error:  # noqa: BLE001
            outcomes.put(error)
        else:
            outcomes.put(None)

    threading.Thread(target=run, daemon=True).start()
    try:
        outcome = outcomes.get(timeout=CHECKPOINT_TIMEOUT_SECONDS)
    except queue.Empty as error:
        raise TimeoutError(
            f"native CUDA checkpoint exceeded {CHECKPOINT_TIMEOUT_SECONDS} seconds"
        ) from error
    if outcome is not None:
        raise outcome


def test_cucheckpoint_preserves_symmetric_memory_cuda_graph(tmp_path: Path) -> None:
    interposer, coordinator = _build_native_tools(tmp_path)
    gpus = _visible_gpus()
    control_dir = tmp_path / "control"
    checkpoint_dir = tmp_path / "checkpoint"
    sync_dir = tmp_path / "sync"
    control_dir.mkdir()
    checkpoint_dir.mkdir()
    sync_dir.mkdir()
    store_path = tmp_path / "torch-distributed-store"

    parent: subprocess.Popen[str] | None = None
    child_pids: tuple[int, int] = ()
    output = ("", "")
    output_collected = False
    failure: Exception | None = None
    external_allocations: list[_ExternalAllocation] = []
    restore_channels = [socket.socketpair() for _ in range(WORLD_SIZE)]

    try:
        external_allocations = _create_external_allocations(1)
        parent = _start_parent(
            gpus,
            interposer,
            control_dir,
            sync_dir,
            store_path,
            tuple(allocation.fd for allocation in external_allocations),
            tuple(receiver.fileno() for _, receiver in restore_channels),
        )
        for _, receiver in restore_channels:
            receiver.close()
        child_pids = _wait_for_child_pids(parent, sync_dir)
        _wait_for_workers(parent, sync_dir, "ready")
        _destroy_external_allocations(external_allocations)
        _assert_worker_runtime(parent.pid, interposer, control_dir)
        for child_pid in child_pids:
            _assert_worker_runtime(child_pid, interposer, control_dir)

        _run_coordinator(
            coordinator,
            "--prepare",
            checkpoint_dir,
            control_dir,
            child_pids,
        )
        state = checkpoint_dir / "cuda-vmm.state"
        if not state.is_file() or state.stat().st_size == 0:
            raise AssertionError("coordinator did not write nonempty cuda-vmm.state")

        _native_checkpoint_with_timeout(child_pids)
        _run_coordinator(
            coordinator,
            "--restore",
            checkpoint_dir,
            control_dir,
            child_pids,
        )
        external_allocations = _create_external_allocations(0x40)
        for rank, (sender, _) in enumerate(restore_channels):
            send_handle(sender, external_allocations[rank].fd, child_pids[rank])
        (sync_dir / "continue").touch()
        _wait_for_workers(parent, sync_dir, "done")

        output = parent.communicate(timeout=COMMAND_TIMEOUT_SECONDS)
        output_collected = True
        if parent.returncode != 0:
            raise RuntimeError(f"parent {parent.pid} exited with {parent.returncode}")
    except Exception as error:  # noqa: BLE001
        failure = error
    finally:
        if parent is not None and (failure is not None or parent.poll() is None):
            try:
                os.killpg(parent.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if parent is not None and not output_collected:
            try:
                output = parent.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(parent.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                output = parent.communicate(timeout=10)
        _destroy_external_allocations(external_allocations)
        for sender, receiver in restore_channels:
            sender.close()
            receiver.close()

    if failure is not None:
        parent_pid = parent.pid if parent is not None else "not started"
        parent_returncode = parent.returncode if parent is not None else "not started"
        sockets = sorted(path.name for path in control_dir.glob("*.sock"))
        raise AssertionError(
            f"{failure}\n"
            f"parent PID/return code: {parent_pid}/{parent_returncode}\n"
            f"forked child PIDs: {child_pids}\n"
            f"control sockets: {sockets}\n"
            f"parent and worker stdout:\n{output[0]}\n"
            f"parent and worker stderr:\n{output[1]}"
        ) from failure


if __name__ == "__main__":
    if len(sys.argv) != 8 or sys.argv[1] != "--parent":
        raise SystemExit(
            "usage: test_cucheckpoint.py --parent RAW_FD_0 RAW_FD_1 "
            "RESTORE_FD_0 RESTORE_FD_1 SYNC_DIR STORE_PATH"
        )
    _fork_workers(
        (int(sys.argv[2]), int(sys.argv[3])),
        (int(sys.argv[4]), int(sys.argv[5])),
        Path(sys.argv[6]),
        Path(sys.argv[7]),
    )
