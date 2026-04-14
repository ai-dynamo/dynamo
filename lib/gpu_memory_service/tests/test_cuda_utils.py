# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from unittest import mock

import pytest

from gpu_memory_service.common import cuda_utils

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _Prop:
    def __init__(self) -> None:
        self.type = None
        self.location = types.SimpleNamespace(type=None, id=None)
        self.requestedHandleTypes = None


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
