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

"""One bounded driver allocation; host physical free measured immediately before exec."""
import argparse
import ctypes as C
import json
import os
import uuid

p = argparse.ArgumentParser()
p.add_argument("--uuid", required=True)
p.add_argument("--mib", type=int, required=True)
p.add_argument("--expected-code", type=int, required=True)
p.add_argument("--host-free-mib", type=int, required=True)
p.add_argument("--expected-sm", type=int, required=True)
p.add_argument("--selection", required=True)
p.add_argument("--hostile", action="store_true")
p.add_argument("--narrow", action="store_true")
a = p.parse_args()
if a.selection == "dynamic":
    os.environ.pop("CUDA_MPS_SM_PARTITION", None)
else:
    os.environ["CUDA_MPS_SM_PARTITION"] = a.selection
original = {
    k: os.getenv(k)
    for k in [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
        "NVIDIA_GPU_MEMORY_LIMIT",
    ]
}
if a.hostile:
    os.environ.pop("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", None)
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
if a.narrow:
    os.environ["CUDA_VISIBLE_DEVICES"] = a.uuid
assert a.host_free_mib > a.mib + 1024, "Inconclusive physical headroom; do not allocate"
c = C.CDLL("libcuda.so.1")


def call(name, types, *args):
    f = getattr(c, name)
    f.argtypes = types
    f.restype = C.c_int
    return f(*args)


assert call("cuInit", [C.c_uint], 0) == 0
n = C.c_int()
assert call("cuDeviceGetCount", [C.POINTER(C.c_int)], C.byref(n)) == 0


class U(C.Structure):
    _fields_ = [("bytes", C.c_ubyte * 16)]


devs = []
for i in range(n.value):
    u = U()
    assert call("cuDeviceGetUuid_v2", [C.POINTER(U), C.c_int], C.byref(u), i) == 0
    devs.append("GPU-" + str(uuid.UUID(bytes=bytes(u.bytes))))
assert len(devs) == (1 if a.narrow else 2) and a.uuid in devs
ordinal = devs.index(a.uuid)
ctx = C.c_void_p()
ptr = C.c_uint64()
sm = C.c_int()
assert (
    call(
        "cuDevicePrimaryCtxRetain",
        [C.POINTER(C.c_void_p), C.c_int],
        C.byref(ctx),
        ordinal,
    )
    == 0
)
try:
    assert call("cuCtxSetCurrent", [C.c_void_p], ctx) == 0
    assert (
        call(
            "cuDeviceGetAttribute",
            [C.POINTER(C.c_int), C.c_int, C.c_int],
            C.byref(sm),
            16,
            ordinal,
        )
        == 0
    )
    assert sm.value == a.expected_sm

    def info():
        free, total = C.c_size_t(), C.c_size_t()
        assert (
            call(
                "cuMemGetInfo_v2",
                [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)],
                C.byref(free),
                C.byref(total),
            )
            == 0
        )
        return {"free_bytes": free.value, "total_bytes": total.value}

    before = info()
    code = call(
        "cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t], C.byref(ptr), a.mib << 20
    )
    after = info()
    if code == 0:
        try:
            for off in [0, (a.mib << 20) - 256]:
                assert (
                    call(
                        "cuMemsetD8_v2",
                        [C.c_uint64, C.c_ubyte, C.c_size_t],
                        ptr.value + off,
                        0x5A,
                        256,
                    )
                    == 0
                )
            assert call("cuCtxSynchronize", []) == 0
        finally:
            assert call("cuMemFree_v2", [C.c_uint64], ptr) == 0
    print(
        json.dumps(
            {
                "event": "allocation",
                "uuid": a.uuid,
                "ordinal": ordinal,
                "visible_uuids": devs,
                "requested_mib": a.mib,
                "cuda_code": code,
                "expected_code": a.expected_code,
                "host_physical_free_mib": a.host_free_mib,
                "visible_sms": sm.value,
                "hostile": a.hostile,
                "narrow": a.narrow,
                "original_env": original,
                "effective_client_pinned": os.getenv(
                    "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"
                ),
                "effective_client_compute": os.getenv(
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
                ),
                "cuda_before": before,
                "cuda_after": after,
            }
        ),
        flush=True,
    )
    assert code == a.expected_code, "Unexpected allocation result"
finally:
    call("cuCtxSetCurrent", [C.c_void_p], None)
    call("cuDevicePrimaryCtxRelease_v2", [C.c_int], ordinal)
