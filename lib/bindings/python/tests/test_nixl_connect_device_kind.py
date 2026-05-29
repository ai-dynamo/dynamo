# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.nixl_connect import DeviceKind

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge]


def test_host_nixl_mem_type():
    """DeviceKind.HOST.nixl_mem_type must return the canonical NIXL DRAM segment name."""
    assert DeviceKind.HOST.nixl_mem_type == "DRAM"


@pytest.mark.gpu_0
def test_cuda_nixl_mem_type():
    """DeviceKind.CUDA.nixl_mem_type must return the canonical NIXL VRAM segment name."""
    assert DeviceKind.CUDA.nixl_mem_type == "VRAM"


def test_str_unchanged():
    """Adding nixl_mem_type must not alter the existing __str__ representations."""
    assert str(DeviceKind.HOST) == "cpu"
    assert str(DeviceKind.CUDA) == "cuda"


def test_unspecified_raises():
    """Accessing nixl_mem_type on UNSPECIFIED must raise ValueError."""
    with pytest.raises(ValueError):
        _ = DeviceKind.UNSPECIFIED.nixl_mem_type
