# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

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
JOB_ID = "1" * 32
PARTICIPANT_IDS = ("a" * 32, "b" * 32)


def _cuda_call(function, *arguments):
    status, *outputs = function(*arguments)
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{function.__name__} failed: {status.name} ({int(status)})")
    if not outputs:
        return None
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def _worker(rank: int, sync_dir: Path, store_path: Path) -> None:
    torch.cuda.set_device(rank)
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

    graph.replay()
    torch.cuda.synchronize()
    _assert_exact_result(output, "after restore")
    (sync_dir / f"done-{rank}").touch()

    dist.barrier()
    del graph, output, symm_handle, input_tensor
    torch.cuda.empty_cache()
    dist.destroy_process_group()


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


def _start_worker(
    rank: int,
    gpus: tuple[str, str],
    interposer: Path,
    control_dir: Path,
    sync_dir: Path,
    store_path: Path,
) -> subprocess.Popen[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpus),
            "DYN_SNAPSHOT_CUDA_VMM_INTERPOSE": "1",
            "DYN_SNAPSHOT_JOB_ID": JOB_ID,
            "DYN_SNAPSHOT_PARTICIPANT_ID": PARTICIPANT_IDS[rank],
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
            "--worker",
            str(rank),
            str(sync_dir),
            str(store_path),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_workers(
    processes: list[subprocess.Popen[str]],
    sync_dir: Path,
    marker: str,
) -> None:
    expected = [sync_dir / f"{marker}-{rank}" for rank in range(WORLD_SIZE)]
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    while not all(path.exists() for path in expected):
        exited = [
            f"{process.pid} ({process.returncode})"
            for process in processes
            if process.poll() is not None
        ]
        if exited:
            raise RuntimeError(f"workers exited before {marker}: {', '.join(exited)}")
        if time.monotonic() >= deadline:
            missing = [str(path) for path in expected if not path.exists()]
            raise TimeoutError(f"timed out waiting for {marker}: {', '.join(missing)}")
        time.sleep(0.05)


def _assert_worker_runtime(
    process: subprocess.Popen[str], interposer: Path, control_dir: Path
) -> None:
    maps = Path(f"/proc/{process.pid}/maps").read_text().splitlines()
    mapped_paths = {
        fields[5] for line in maps if len(fields := line.split(maxsplit=5)) == 6
    }
    if str(interposer) not in mapped_paths:
        raise AssertionError(f"{interposer} is not loaded in process {process.pid}")
    endpoint = control_dir / f"cuda-vmm-{process.pid}.sock"
    if not endpoint.is_socket():
        raise AssertionError(f"interposer endpoint does not exist: {endpoint}")


def _run_coordinator(
    coordinator: Path,
    operation: str,
    checkpoint_dir: Path,
    control_dir: Path,
    processes: list[subprocess.Popen[str]],
) -> None:
    command = [
        str(coordinator),
        operation,
        "--proc-root",
        "",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    for process in processes:
        command.extend(["--process", str(process.pid), str(process.pid)])
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


def _expect_state(process: subprocess.Popen[str], expected) -> None:
    actual = _cuda_call(driver.cuCheckpointProcessGetState, process.pid)
    if actual != expected:
        raise AssertionError(
            f"CUDA process {process.pid} is {actual.name}, expected {expected.name}"
        )


def _native_checkpoint(processes: list[subprocess.Popen[str]]) -> None:
    _cuda_call(driver.cuInit, 0)
    running = driver.CUprocessState.CU_PROCESS_STATE_RUNNING
    locked = driver.CUprocessState.CU_PROCESS_STATE_LOCKED
    checkpointed = driver.CUprocessState.CU_PROCESS_STATE_CHECKPOINTED
    for process in processes:
        _expect_state(process, running)

    lock_arguments = driver.CUcheckpointLockArgs()
    lock_arguments.timeoutMs = COMMAND_TIMEOUT_SECONDS * 1000
    for process in processes:
        _cuda_call(driver.cuCheckpointProcessLock, process.pid, lock_arguments)
    for process in processes:
        _expect_state(process, locked)

    checkpoint_arguments = driver.CUcheckpointCheckpointArgs()
    for process in processes:
        _cuda_call(
            driver.cuCheckpointProcessCheckpoint,
            process.pid,
            checkpoint_arguments,
        )
    for process in processes:
        _expect_state(process, checkpointed)

    restore_arguments = driver.CUcheckpointRestoreArgs()
    for process in processes:
        _cuda_call(driver.cuCheckpointProcessRestore, process.pid, restore_arguments)
    for process in processes:
        _expect_state(process, locked)

    unlock_arguments = driver.CUcheckpointUnlockArgs()
    for process in processes:
        _cuda_call(driver.cuCheckpointProcessUnlock, process.pid, unlock_arguments)
    for process in processes:
        _expect_state(process, running)


def _native_checkpoint_with_timeout(
    processes: list[subprocess.Popen[str]],
) -> None:
    outcomes: queue.Queue[Exception | None] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            _native_checkpoint(processes)
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

    processes: list[subprocess.Popen[str]] = []
    outputs: dict[int, tuple[str, str]] = {}
    failure: Exception | None = None

    try:
        for rank in range(WORLD_SIZE):
            process = _start_worker(
                rank,
                gpus,
                interposer,
                control_dir,
                sync_dir,
                store_path,
            )
            processes.append(process)

        _wait_for_workers(processes, sync_dir, "ready")
        for process in processes:
            _assert_worker_runtime(process, interposer, control_dir)

        _run_coordinator(
            coordinator,
            "--prepare",
            checkpoint_dir,
            control_dir,
            processes,
        )
        state = checkpoint_dir / "cuda-vmm.state"
        if not state.is_file() or state.stat().st_size == 0:
            raise AssertionError("coordinator did not write nonempty cuda-vmm.state")

        _native_checkpoint_with_timeout(processes)
        _run_coordinator(
            coordinator,
            "--restore",
            checkpoint_dir,
            control_dir,
            processes,
        )
        (sync_dir / "continue").touch()
        _wait_for_workers(processes, sync_dir, "done")

        for process in processes:
            stdout, stderr = process.communicate(timeout=COMMAND_TIMEOUT_SECONDS)
            outputs[process.pid] = (stdout, stderr)
            if process.returncode != 0:
                raise RuntimeError(
                    f"worker {process.pid} exited with {process.returncode}"
                )
    except Exception as error:  # noqa: BLE001
        failure = error
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.pid in outputs:
                continue
            try:
                outputs[process.pid] = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                outputs[process.pid] = process.communicate(timeout=10)

    if failure is not None:
        diagnostics = []
        for process in processes:
            stdout, stderr = outputs.get(process.pid, ("", ""))
            diagnostics.append(
                f"worker {process.pid} return code: {process.returncode}\n"
                f"worker {process.pid} stdout:\n{stdout}\n"
                f"worker {process.pid} stderr:\n{stderr}"
            )
        raise AssertionError(f"{failure}\n" + "\n".join(diagnostics)) from failure


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit(
            "usage: test_cucheckpoint.py --worker RANK SYNC_DIR STORE_PATH"
        )
    _worker(int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
