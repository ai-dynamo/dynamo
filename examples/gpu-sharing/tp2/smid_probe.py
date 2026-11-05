#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded execution diagnostic: record client-visible SM IDs from actual CUDA blocks."""
import argparse
import collections
import ctypes as C
import datetime
import json
import os
import time
import uuid
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--expected-sm", type=int)
p.add_argument("--uuid", required=True)
p.add_argument(
    "--mode",
    choices=[
        "assigned",
        "missing-socket-valid",
        "missing-socket-omitted",
        "correct-socket-omitted",
        "empty-socket",
        "unset-socket",
        "wrong-token",
    ],
    required=True,
)
p.add_argument("--selection", required=True)
p.add_argument("--launches", type=int, default=8)
p.add_argument("--work-seconds", type=float, default=0)
p.add_argument("--ready-file")
p.add_argument("--all-visible", action="store_true")
a = p.parse_args()
assert a.launches == 8 and 0 <= a.work_seconds <= 30
original = {
    k: os.getenv(k)
    for k in [
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_SM_PARTITION",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "CUDA_MPS_PROTOCOL_VERSION",
    ]
}
os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
os.environ.pop("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", None)
if a.mode in ["missing-socket-omitted", "correct-socket-omitted"]:
    os.environ.pop("CUDA_MPS_SM_PARTITION", None)
elif a.selection == "dynamic":
    os.environ.pop("CUDA_MPS_SM_PARTITION", None)
else:
    os.environ["CUDA_MPS_SM_PARTITION"] = a.selection
if a.mode.startswith("missing-socket"):
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/tp2-smid-nonexistent-socket"
if a.mode == "empty-socket":
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = ""
if a.mode == "unset-socket":
    os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
if a.mode == "wrong-token":
    os.environ["CUDA_MPS_SM_PARTITION"] = a.uuid + "/invalid_partition_token"
if a.all_visible:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = a.uuid
assert not os.path.exists("/tmp/tp2-smid-nonexistent-socket")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def emit(event, **data):
    print(json.dumps({"event": event, "utc": utc(), **data}), flush=True)


env = {k: os.getenv(k) for k in original}
emit(
    "process",
    pid=os.getpid(),
    uid=os.getuid(),
    euid=os.geteuid(),
    gid=os.getgid(),
    uuid=a.uuid,
    mode=a.mode,
    original_environment=original,
    effective_environment=env,
    grid_blocks=8192,
    threads_per_block=256,
    launches=a.launches,
    work_seconds=a.work_seconds,
)
c = C.CDLL("libcuda.so.1")


def call(name, types, *args):
    f = getattr(c, name)
    f.argtypes = types
    f.restype = C.c_int
    return f(*args)


def error(code):
    name, message = C.c_char_p(), C.c_char_p()
    call("cuGetErrorName", [C.c_int, C.POINTER(C.c_char_p)], code, C.byref(name))
    call("cuGetErrorString", [C.c_int, C.POINTER(C.c_char_p)], code, C.byref(message))
    return {
        "code": code,
        "name": name.value.decode() if name.value else None,
        "message": message.value.decode() if message.value else None,
    }


def checked(name, types, *args):
    code = call(name, types, *args)
    emit("cuda_call", function=name, **error(code))
    if code:
        raise RuntimeError(name + " failed " + str(error(code)))
    return code


# PTX stores one SM ID per block. Only thread0 burns a bounded clock interval;
# all blocks finish, with no persistent kernel. No throughput/timing comparison.
PTX = r""".version 7.0
.target sm_80
.address_size 64
.visible .entry record_smid(.param .u64 output, .param .u64 cycles) {
 .reg .pred %p<3>;
 .reg .b32 %r<4>;
 .reg .b64 %d<8>;
 mov.u32 %r0, %tid.x;
 setp.ne.u32 %p0, %r0, 0;
 @%p0 bra DONE;
 ld.param.u64 %d0, [output];
 ld.param.u64 %d1, [cycles];
 mov.u32 %r1, %ctaid.x;
 mul.wide.u32 %d2, %r1, 4;
 add.u64 %d3, %d0, %d2;
 mov.u32 %r2, %smid;
 st.global.u32 [%d3], %r2;
 mov.u64 %d4, %clock64;
LOOP:
 mov.u64 %d5, %clock64;
 sub.u64 %d6, %d5, %d4;
 setp.lt.u64 %p1, %d6, %d1;
 @%p1 bra LOOP;
DONE:
 ret;
}
"""
ctx = C.c_void_p()
module = C.c_void_p()
ptr = C.c_uint64()
device = None
union = set()
launch_records = []
failed = False
try:
    checked("cuInit", [C.c_uint], 0)
    driver_version = C.c_int()
    checked("cuDriverGetVersion", [C.POINTER(C.c_int)], C.byref(driver_version))
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcuda.so" in line
        }
    )
    emit(
        "driver_client_libraries",
        cuda_driver_api=driver_version.value,
        library_paths=libraries,
    )

    n = C.c_int()
    checked("cuDeviceGetCount", [C.POINTER(C.c_int)], C.byref(n))

    class U(C.Structure):
        _fields_ = [("data", C.c_ubyte * 16)]

    uuids = []
    for i in range(n.value):
        u = U()
        checked("cuDeviceGetUuid_v2", [C.POINTER(U), C.c_int], C.byref(u), i)
        uuids.append("GPU-" + str(uuid.UUID(bytes=bytes(u.data))))
    assert a.uuid in uuids
    device = uuids.index(a.uuid)
    emit(
        "device_mapping",
        ordinal=device,
        selected_uuid=uuids[device],
        visible_uuids=uuids,
    )
    checked(
        "cuDevicePrimaryCtxRetain",
        [C.POINTER(C.c_void_p), C.c_int],
        C.byref(ctx),
        device,
    )
    checked("cuCtxSetCurrent", [C.c_void_p], ctx)
    sm_count = C.c_int()
    checked(
        "cuDeviceGetAttribute",
        [C.POINTER(C.c_int), C.c_int, C.c_int],
        C.byref(sm_count),
        16,
        device,
    )
    emit("device_properties", sm_count=sm_count.value)
    ptx = C.create_string_buffer(PTX.encode())
    checked(
        "cuModuleLoadData",
        [C.POINTER(C.c_void_p), C.c_void_p],
        C.byref(module),
        C.cast(ptx, C.c_void_p),
    )
    fn = C.c_void_p()
    checked(
        "cuModuleGetFunction",
        [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
        C.byref(fn),
        module,
        b"record_smid",
    )
    blocks = 8192
    checked(
        "cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t], C.byref(ptr), blocks * 4
    )
    clock_cycles = C.c_uint64(1_000_000)
    params = (C.c_void_p * 2)(
        C.cast(C.byref(ptr), C.c_void_p), C.cast(C.byref(clock_cycles), C.c_void_p)
    )
    start = time.monotonic()
    deadline = start + a.work_seconds
    limit = 1000 if a.work_seconds else a.launches
    for iteration in range(limit):
        if a.work_seconds and iteration and time.monotonic() >= deadline:
            break
        reset = call(
            "cuMemsetD32_v2",
            [C.c_uint64, C.c_uint, C.c_size_t],
            ptr,
            0xFFFFFFFF,
            blocks,
        )
        launch = (
            call(
                "cuLaunchKernel",
                [
                    C.c_void_p,
                    C.c_uint,
                    C.c_uint,
                    C.c_uint,
                    C.c_uint,
                    C.c_uint,
                    C.c_uint,
                    C.c_uint,
                    C.c_void_p,
                    C.POINTER(C.c_void_p),
                    C.POINTER(C.c_void_p),
                ],
                fn,
                blocks,
                1,
                1,
                256,
                1,
                1,
                0,
                None,
                params,
                None,
            )
            if reset == 0
            else None
        )
        sync = call("cuCtxSynchronize", []) if launch == 0 else None
        event = {
            "iteration": iteration,
            "reset": error(reset),
            "launch": error(launch) if launch is not None else None,
            "synchronize": error(sync) if sync is not None else None,
        }
        if reset or launch or sync:
            emit("kernel", **event)
            failed = True
            break
        output = (C.c_uint * blocks)()
        copy = call(
            "cuMemcpyDtoH_v2",
            [C.c_void_p, C.c_uint64, C.c_size_t],
            C.cast(output, C.c_void_p),
            ptr,
            blocks * 4,
        )
        event["copy"] = error(copy)
        if copy:
            emit("kernel", **event)
            failed = True
            break
        hist = collections.Counter(output)
        missing = hist.pop(0xFFFFFFFF, 0)
        ids = sorted(hist)
        union.update(ids)
        event.update(
            blocks_recorded=sum(hist.values()),
            unwritten_blocks=missing,
            sm_ids=ids,
            distinct_sm_count=len(ids),
            block_count_by_sm=dict(sorted(hist.items())),
        )
        emit("kernel", **event)
        launch_records.append(
            {
                "iteration": iteration,
                "distinct_sm_count": len(ids),
                "unwritten_blocks": missing,
                "sm_ids": ids,
                "utc": utc(),
            }
        )
        if missing:
            failed = True
            break
        if a.work_seconds and iteration == 0:
            if a.ready_file:
                Path(a.ready_file).write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "first_kernel_completed_utc": utc(),
                            "uuid": a.uuid,
                            "sm_ids": ids,
                        }
                    )
                )
            emit("active_work_started", uuid=a.uuid)
        if a.work_seconds:
            time.sleep(0.02)
    emit(
        "execution_result",
        uuid=a.uuid,
        mode=a.mode,
        euid=os.geteuid(),
        successful_launches=len(launch_records),
        execution_failed=failed,
        union_sm_ids=sorted(union),
        union_distinct_sm_count=len(union),
        per_launch_counts=[x["distinct_sm_count"] for x in launch_records],
        work_started_monotonic=start,
        work_completed_monotonic=time.monotonic(),
    )
except Exception as exc:
    failed = True
    emit(
        "probe_exception",
        exception=str(exc),
        union_sm_ids=sorted(union),
        union_distinct_sm_count=len(union),
    )
finally:
    if ptr.value:
        emit(
            "cleanup",
            function="cuMemFree_v2",
            **error(call("cuMemFree_v2", [C.c_uint64], ptr)),
        )
    if module.value:
        emit(
            "cleanup",
            function="cuModuleUnload",
            **error(call("cuModuleUnload", [C.c_void_p], module)),
        )
    if ctx.value:
        call("cuCtxSetCurrent", [C.c_void_p], None)
        emit(
            "cleanup",
            function="cuDevicePrimaryCtxRelease_v2",
            **error(call("cuDevicePrimaryCtxRelease_v2", [C.c_int], device)),
        )
if not failed and a.expected_sm is not None:
    failed = sm_count.value != a.expected_sm or len(union) != a.expected_sm
    emit(
        "expected_share",
        expected_sm=a.expected_sm,
        properties_sm=sm_count.value,
        observed_sm=len(union),
        passed=not failed,
    )
if failed:
    raise SystemExit(1)
