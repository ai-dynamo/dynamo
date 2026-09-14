# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-device smoke coverage for the GMS XPU virtual memory backend."""

from __future__ import annotations

import pytest
from _deps import HAS_GMS, HAS_SYCL_VMM, HAS_XPU

if not HAS_GMS:
    pytest.skip("gpu_memory_service package is not available", allow_module_level=True)

if not HAS_XPU:
    pytest.skip("an Intel XPU is required", allow_module_level=True)

if not HAS_SYCL_VMM:
    pytest.skip("the _sycl_vmm native extension is required", allow_module_level=True)

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.vmm import (
    VMMDeviceType,
    _reset_vmm_singleton,
    get_vmm,
    get_vmm_device_type,
    init_vmm,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
]


def test_xpu_vmm_allocates_maps_and_releases_memory():
    _reset_vmm_singleton()
    vmm = None
    handle = None
    virtual_address = None
    mapped = False
    try:
        init_vmm(VMMDeviceType.XPU)
        vmm = get_vmm()
        assert get_vmm_device_type() is VMMDeviceType.XPU

        vmm.ensure_initialized()
        assert 0 in vmm.list_devices()
        granularity = vmm.get_allocation_granularity(0)
        assert granularity > 0

        allocated, handle = vmm.create_tolerate_oom(granularity, 0)
        assert allocated
        assert handle

        virtual_address = vmm.address_reserve(granularity, granularity)
        vmm.map(virtual_address, granularity, handle)
        mapped = True
        vmm.set_access(virtual_address, granularity, 0, GrantedLockType.RW)
        vmm.validate_pointer(virtual_address)
    finally:
        if vmm is not None:
            if mapped:
                vmm.unmap(virtual_address, granularity)
            if virtual_address is not None:
                vmm.address_free(virtual_address, granularity)
            if handle is not None:
                vmm.release(handle)
        _reset_vmm_singleton()
