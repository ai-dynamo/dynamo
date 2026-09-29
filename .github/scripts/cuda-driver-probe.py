#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded CUDA checkpoint ABI comparison; requires GPU access and checkpoint privileges."""

import argparse
import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

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
    call(
        "cuDevicePrimaryCtxRetain",
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        ctypes.byref(context),
        0,
    )
    call("cuCtxSetCurrent", [ctypes.c_void_p], context)
    call(
        "cuMemAlloc_v2",
        [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
        ctypes.byref(pointer),
        SIZE,
    )
    call(
        "cuMemsetD8_v2",
        [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t],
        pointer,
        SENTINEL,
        SIZE,
    )
    call("cuCtxSynchronize", [])
    maps = sorted(
        {
            line.split(maxsplit=5)[5].strip()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libcuda.so" in line
        }
    )
    emit(event="ready", pid=os.getpid(), requested_driver=driver, mapped_drivers=maps)
    if sys.stdin.readline().strip() != "verify":
        return 2
    data = (ctypes.c_ubyte * SIZE)()
    call(
        "cuMemcpyDtoH_v2",
        [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
        data,
        pointer,
        SIZE,
    )
    verified = bytes(data) == bytes([SENTINEL]) * SIZE
    emit(event="verified", pid=os.getpid(), bytes=SIZE, success=verified)
    call("cuMemFree_v2", [ctypes.c_uint64], pointer)
    call("cuDevicePrimaryCtxRelease_v2", [ctypes.c_int], 0)
    return 0 if verified else 1


def case(name, driver, steps, helper, expect_success):
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    process = subprocess.Popen(
        [sys.executable, "-S", __file__, "--target", driver],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    try:
        if not select.select([process.stdout], [], [], 10)[0]:
            raise RuntimeError("target readiness exceeded 10s")
        ready_line = process.stdout.readline()
        try:
            ready = json.loads(ready_line)
        except ValueError as error:
            raise RuntimeError(
                f"target did not emit readiness: {ready_line.strip()}"
            ) from error
        if ready.get("event") != "ready":
            raise RuntimeError(f"unexpected target readiness: {ready}")
        mapped = ready.get("mapped_drivers", [])
        if len(mapped) != 1 or Path(mapped[0]).name != Path(driver).resolve().name:
            raise RuntimeError(f"target loaded unexpected CUDA driver: {mapped}")
        emit(case=name, **ready)
        for index, (action, helper_driver) in enumerate(steps):
            helper_environment = os.environ.copy()
            helper_environment["LD_PRELOAD"] = helper_driver
            started = time.monotonic()
            emit(
                event="helper_start",
                case=name,
                pid=process.pid,
                action=action,
                driver=helper_driver,
            )
            try:
                completed = subprocess.run(
                    [helper, "--action", action, "--pid", str(process.pid)],
                    env=helper_environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3,
                    check=False,
                )
                status, output = completed.returncode, completed.stdout
            except subprocess.TimeoutExpired as error:
                status, output = "timeout", error.output or b""
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
            emit(
                event="helper_end",
                case=name,
                pid=process.pid,
                action=action,
                driver=helper_driver,
                status=status,
                elapsed=round(time.monotonic() - started, 3),
                output=output[-4096:],
            )
            if status != 0:
                # Only the final, deliberately mismatched operation is expected
                # to fail; failure preparing it is an inconclusive experiment.
                expected = not expect_success and index == len(steps) - 1
                emit(
                    event="case_end",
                    case=name,
                    success=expected,
                    outcome="expected_mismatch_failure"
                    if expected
                    else "unexpected_failure",
                )
                return expected
        if not expect_success:
            emit(
                event="case_end", case=name, success=False, outcome="mismatch_succeeded"
            )
            return False
        process.stdin.write("verify\n")
        process.stdin.flush()
        output, _ = process.communicate(timeout=3)
        verified = any(
            json.loads(line).get("success") is True
            for line in output.splitlines()
            if line.startswith("{")
        )
        success = process.returncode == 0 and verified
        emit(
            event="case_end",
            case=name,
            success=success,
            outcome="memory_verified" if success else "verification_failed",
            output=output[-4096:],
        )
        return success
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        emit(event="case_end", case=name, success=False, error=str(error))
        return False
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in (process.stdin, process.stdout):
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help=argparse.SUPPRESS)
    parser.add_argument("--driver-595")
    parser.add_argument("--driver-615")
    parser.add_argument("--helper", default="/helpers/cuda-checkpoint-helper")
    args = parser.parse_args()
    if args.target:
        try:
            return target(args.target)
        except (OSError, RuntimeError, AttributeError) as error:
            emit(event="target_error", error=str(error))
            return 1
    for label, path in (
        ("driver-595", args.driver_595),
        ("driver-615", args.driver_615),
        ("helper", args.helper),
    ):
        if not path or not Path(path).is_file():
            parser.error(f"{label}: missing file: {path}")
    if not os.access(args.helper, os.X_OK):
        parser.error(f"helper is not executable: {args.helper}")
    old, new = args.driver_595, args.driver_615
    actions = ("lock", "checkpoint", "restore", "unlock")
    matrix = (
        ("matched595", old, [(action, old) for action in actions], True),
        ("matched615", new, [(action, new) for action in actions], True),
        ("helper595_target615_lock", new, [("lock", old)], False),
        (
            "helper615_target595_restore",
            old,
            [("lock", old), ("checkpoint", old), ("restore", new)],
            False,
        ),
    )
    results = [
        case(name, driver, steps, args.helper, expected)
        for name, driver, steps, expected in matrix
    ]
    emit(
        event="summary",
        matched_controls_passed=all(results[:2]),
        all_expectations_met=all(results),
    )
    return 0 if all(results) else 1


if __name__ == "__main__":

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
