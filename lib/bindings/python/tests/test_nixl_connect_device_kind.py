# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DeviceKind enum in dynamo.nixl_connect.

nixl bindings are mocked so these tests run on CPU-only machines without
NIXL installed, following the same pattern as test_nixl_connect_unit.py.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge]


@pytest.fixture
def device_kind():
    """Yield DeviceKind with nixl and cupy bindings mocked out."""
    with patch.dict(
        sys.modules,
        {
            "nixl": MagicMock(),
            "nixl._api": MagicMock(),
            "nixl._bindings": MagicMock(),
            "cupy": MagicMock(),
            "cupy_backends": MagicMock(),
            "cupy_backends.cuda": MagicMock(),
            "cupy_backends.cuda.api": MagicMock(),
            "cupy_backends.cuda.api.runtime": MagicMock(),
        },
    ):
        from dynamo.nixl_connect import DeviceKind

        yield DeviceKind


def test_host_nixl_mem_type(device_kind):
    """DeviceKind.HOST.nixl_mem_type must return the canonical NIXL DRAM segment name."""
    assert device_kind.HOST.nixl_mem_type == "DRAM"


@pytest.mark.gpu_0
def test_cuda_nixl_mem_type(device_kind):
    """DeviceKind.CUDA.nixl_mem_type must return the canonical NIXL VRAM segment name."""
    assert device_kind.CUDA.nixl_mem_type == "VRAM"


def test_str_unchanged(device_kind):
    """Adding nixl_mem_type must not alter the existing __str__ representations."""
    assert str(device_kind.HOST) == "cpu"
    assert str(device_kind.CUDA) == "cuda"


def test_unspecified_raises(device_kind):
    """Accessing nixl_mem_type on UNSPECIFIED must raise ValueError."""
    with pytest.raises(ValueError):
        _ = device_kind.UNSPECIFIED.nixl_mem_type
