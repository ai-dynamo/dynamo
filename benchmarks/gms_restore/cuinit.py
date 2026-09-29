# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process cuInit probe; use actual DRA container isolation for comparison."""

import ctypes
import json
import os
import time

started = time.perf_counter()
cuda = ctypes.CDLL("libcuda.so.1")
loaded = time.perf_counter()
result = cuda.cuInit(0)
initialized = time.perf_counter()
if result:
    raise RuntimeError(f"cuInit failed: {result}")
count = ctypes.c_int()
assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0
uuids = []
for ordinal in range(count.value):
    device = ctypes.c_int()
    uuid = (ctypes.c_ubyte * 16)()
    assert cuda.cuDeviceGet(ctypes.byref(device), ordinal) == 0
    assert cuda.cuDeviceGetUuid(ctypes.byref(uuid), device) == 0
    from uuid import UUID

    uuids.append("GPU-" + str(UUID(bytes=bytes(uuid))))
print(
    json.dumps(
        {
            "visible_devices": count.value,
            "uuids": uuids,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dlopen_s": loaded - started,
            "cuinit_s": initialized - loaded,
            "total_s": time.perf_counter() - started,
        }
    ),
    flush=True,
)
