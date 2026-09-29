# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental CUDA Python bridge through LD_PRELOAD's VMM entrypoints.

The evidence image's cuda-python dispatch bypasses the interposed allocation
entrypoints (the captured GMS cuinterpose state contains zero memory blocks).
Explicitly route only the VMM ownership APIs through the loaded frontend.
Must be installed before any VMM allocations. Benchmark-only, CUDA/Linux only.
"""

import ctypes as C

from cuda.bindings import driver as cuda


def install():
    lib = C.CDLL(None)

    def bind(name, args):
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = C.c_int
        return fn

    create = bind(
        "cuMemCreate", [C.POINTER(C.c_uint64), C.c_size_t, C.c_void_p, C.c_ulonglong]
    )
    export = bind(
        "cuMemExportToShareableHandle", [C.c_void_p, C.c_uint64, C.c_int, C.c_ulonglong]
    )
    release = bind("cuMemRelease", [C.c_uint64])
    map_ = bind(
        "cuMemMap", [C.c_uint64, C.c_size_t, C.c_size_t, C.c_uint64, C.c_ulonglong]
    )
    unmap = bind("cuMemUnmap", [C.c_uint64, C.c_size_t])

    def mem_create(size, prop, flags):
        handle = C.c_uint64()
        result = create(C.byref(handle), size, prop.getPtr(), flags)
        return cuda.CUresult(result), handle.value

    def mem_export(handle, kind, flags):
        if int(kind) != 1:
            raise ValueError("POSIX FD only")
        fd = C.c_int(-1)
        result = export(C.byref(fd), int(handle), int(kind), flags)
        return cuda.CUresult(result), fd.value

    cuda.cuMemCreate = mem_create
    cuda.cuMemExportToShareableHandle = mem_export
    cuda.cuMemRelease = lambda handle: (cuda.CUresult(release(int(handle))),)
    cuda.cuMemMap = lambda va, size, offset, handle, flags: (
        cuda.CUresult(map_(int(va), size, offset, int(handle), flags)),
    )
    cuda.cuMemUnmap = lambda va, size: (cuda.CUresult(unmap(int(va), size)),)
