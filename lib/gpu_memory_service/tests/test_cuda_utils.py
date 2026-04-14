# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from unittest import mock

from gpu_memory_service.common import cuda_utils


class _Prop:
    def __init__(self) -> None:
        self.type = None
        self.location = types.SimpleNamespace(type=None, id=None)
        self.requestedHandleTypes = None


def test_cuda_set_current_device_does_not_initialize_driver() -> None:
    calls: list[object] = []
    fake_cuda = types.SimpleNamespace(
        CUresult=types.SimpleNamespace(CUDA_SUCCESS=0),
        cuDevicePrimaryCtxRetain=lambda device: calls.append(("retain", device))
        or (0, f"ctx-{device}"),
        cuCtxSetCurrent=lambda ctx: calls.append(("set", ctx)) or (0,),
        cuDevicePrimaryCtxRelease=lambda device: (0,),
    )

    with mock.patch.object(cuda_utils, "cuda", fake_cuda):
        with mock.patch.object(cuda_utils, "_primary_contexts", {}):
            with mock.patch.object(
                cuda_utils,
                "_primary_context_release_registered",
                False,
            ):
                cuda_utils.cuda_set_current_device(3)

    assert calls == [("retain", 3), ("set", "ctx-3")]


def test_cumem_get_allocation_granularity_uses_existing_cuda_init() -> None:
    calls: list[object] = []
    fake_cuda = types.SimpleNamespace(
        CUresult=types.SimpleNamespace(CUDA_SUCCESS=0),
        CUmemAllocationProp=_Prop,
        CUmemAllocationType=types.SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1),
        CUmemLocationType=types.SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE=2),
        CUmemAllocationHandleType=types.SimpleNamespace(
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR=3
        ),
        CUmemAllocationGranularity_flags=types.SimpleNamespace(
            CU_MEM_ALLOC_GRANULARITY_MINIMUM=4
        ),
        cuMemGetAllocationGranularity=lambda prop, flag: calls.append(
            ("granularity", prop.location.id, flag)
        )
        or (0, 2097152),
    )

    with mock.patch.object(cuda_utils, "cuda", fake_cuda):
        granularity = cuda_utils.cumem_get_allocation_granularity(5)

    assert granularity == 2097152
    assert calls == [("granularity", 5, 4)]
