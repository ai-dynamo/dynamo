#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded sibling-process CUDA checkpoint probe; no model or driver-version assumptions."""

import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

SIZE = 1024 * 1024
SENTINEL = 0xA5


def emit(**record):
    print(json.dumps(record), flush=True)


def target(driver):
    cuda = ctypes.CDLL(driver)

    def call(name, types, *args):
        function = getattr(cuda, name)
        function.argtypes = types
        function.restype = ctypes.c_int
        status = function(*args)
        if status:
            raise RuntimeError(f"{name} returned CUDA status {status}")

    context = ctypes.c_void_p()
    pointer = ctypes.c_uint64()
    call("cuInit", [ctypes.c_uint], 0)
    call("cuDevicePrimaryCtxRetain", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int], ctypes.byref(context), 0)
    call("cuCtxSetCurrent", [ctypes.c_void_p], context)
    call("cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t], ctypes.byref(pointer), SIZE)
    call("cuMemsetD8_v2", [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t], pointer, SENTINEL, SIZE)
    call("cuCtxSynchronize", [])
    libraries = sorted({
        line.split(maxsplit=5)[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(maxsplit=5)) == 6 and Path(line.split(maxsplit=5)[5]).name.startswith("libcuda.so.")
    })
    emit(event="ready", pid=os.getpid(), parent=os.getppid(), mapped_drivers=libraries)
    if sys.stdin.readline().strip() != "verify":
        raise RuntimeError("verification command missing")
    data = (ctypes.c_ubyte * SIZE)()
    call("cuMemcpyDtoH_v2", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t], data, pointer, SIZE)
    verified = bytes(data) == bytes([SENTINEL]) * SIZE
    emit(event="verified", bytes=SIZE, success=verified)
    call("cuMemFree_v2", [ctypes.c_uint64], pointer)
    call("cuDevicePrimaryCtxRelease_v2", [ctypes.c_int], 0)
    return 0 if verified else 1


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=1)


def probe(driver):
    mode = os.environ["PROBE_MODE"]
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    expected_uid = 1000 if mode == "nonroot-no-caps" else 0
    emit(event="permissions", mode=mode, uid=os.getuid(), pid=os.getpid(), requested_driver=driver, cap_effective=status["CapEff"].strip(), seccomp=status["Seccomp"].strip())
    if int(status["CapEff"].strip(), 16) != 0 or os.getuid() != expected_uid or os.getpid() == 1:
        raise RuntimeError("expected runtime exec with the requested UID and no effective capabilities")
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    process = subprocess.Popen(
        [sys.executable, "-S", __file__, "--target", driver], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=environment, start_new_session=True,
    )
    try:
        if not select.select([process.stdout], [], [], 10)[0]:
            raise RuntimeError("target readiness exceeded 10s")
        ready = json.loads(process.stdout.readline())
        if ready.get("event") != "ready" or ready.get("parent") != os.getpid():
            raise RuntimeError(f"target readiness failed: {ready}")
        libraries = ready["mapped_drivers"]
        if len(libraries) != 1 or not Path(libraries[0]).is_file():
            raise RuntimeError(f"expected one accessible mapped CUDA library: {libraries}")
        if driver != "libcuda.so.1" and not Path(libraries[0]).samefile(driver):
            raise RuntimeError(f"target did not load the requested compatibility library: {libraries}")
        emit(mode=mode, **ready)
        environment["LD_DEBUG"] = "libs"
        for action in ("lock", "checkpoint", "restore", "unlock"):
            started = time.monotonic()
            emit(event="helper_start", mode=mode, action=action, driver=libraries[0], target_pid=process.pid, parent_pid=os.getpid())
            helper = subprocess.Popen(
                ["/helpers/cuda-checkpoint-helper", "--driver-library", libraries[0], "--action", action, "--pid", str(process.pid)],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            try:
                output, _ = helper.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                stop(helper)
                output, _ = helper.communicate(timeout=1)
                emit(event="helper_end", mode=mode, action=action, status="timeout", elapsed=round(time.monotonic() - started, 3), output=output[-4096:])
                return 1
            finally:
                stop(helper)
            matched = any("calling init:" in line and line.rstrip().endswith(libraries[0]) for line in output.splitlines())
            emit(event="helper_end", mode=mode, action=action, status=helper.returncode, driver_confirmed=matched, elapsed=round(time.monotonic() - started, 3), output=output[-4096:])
            if not matched:
                raise RuntimeError("loader trace did not confirm the exact target CUDA library")
            if helper.returncode:
                return 1
        process.stdin.write("verify\n")
        process.stdin.flush()
        output, _ = process.communicate(timeout=3)
        verified = any(json.loads(line).get("success") is True for line in output.splitlines() if line.startswith("{"))
        emit(event="result", mode=mode, success=process.returncode == 0 and verified, output=output[-4096:])
        return 0 if process.returncode == 0 and verified else 1
    finally:
        stop(process)


def expired(_signum, _frame):
    raise TimeoutError("probe exceeded its 90s hard deadline")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(90)
    try:
        if sys.argv[1:2] == ["--target"]:
            raise SystemExit(target(sys.argv[2]))
        compat = sorted({str(path.resolve()) for path in Path("/usr/local").glob("cuda*/compat/lib.real/libcuda.so.*") if path.is_file()})
        if not compat:
            raise RuntimeError("the pinned image has no discoverable CUDA compatibility libraries")
        emit(event="driver_inventory", compatibility_libraries=compat)
        results = [probe(driver) for driver in ["libcuda.so.1", *compat]]
        emit(event="summary", success=not any(results), cases=len(results))
        raise SystemExit(1 if any(results) else 0)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        emit(event="error", error=str(error))
        raise SystemExit(1)
