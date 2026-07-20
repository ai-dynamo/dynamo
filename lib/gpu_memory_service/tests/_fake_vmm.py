# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared VMMDevice mock for unit tests.

Provides a device-agnostic ``FakeVMM(VMMDevice)`` that stubs all abstract
methods with in-memory counters and ``os.pipe()`` for FD simulation.
Import this in any test that needs to monkeypatch the VMM singleton.
"""

from __future__ import annotations

import itertools
import os

from gpu_memory_service.common.vmm import VMMDevice


class FakeVMM(VMMDevice):
    """Device-agnostic VMMDevice mock for unit tests.

    Works regardless of whether the real backend is CUDA or XPU — all
    VMMDevice methods are stubbed with in-memory counters and os.pipe()
    for FD export/import simulation.
    """

    def __init__(self, devices: list[int] | None = None):
        self._handles = itertools.count(1000)
        self._vas = itertools.count(0x100000, 0x10000)
        self._devices = devices if devices is not None else [0]
        self.calls: list[tuple] = []

    def ensure_initialized(self):
        pass

    def synchronize(self):
        pass

    def list_devices(self):
        return self._devices

    def device_memory_info(self, device):
        return (8 * 1024**3, 16 * 1024**3)

    def get_allocation_granularity(self, device):
        return 4096

    def create_tolerate_oom(self, size, device):
        return (True, next(self._handles))

    def release(self, handle):
        pass

    def export_to_shareable_handle(self, handle):
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        return read_fd

    def import_shareable_handle_close_fd(self, fd):
        os.close(fd)
        return next(self._handles)

    def address_reserve(self, size, granularity):
        return next(self._vas)

    def address_free(self, va, size):
        pass

    def map(self, va, size, handle):
        pass

    def unmap(self, va, size):
        pass

    def set_access(self, va, size, device, access):
        pass

    def validate_pointer(self, va):
        pass

    def runtime_check_result(self, result, name):
        pass

    def runtime_set_device(self, device):
        self.calls.append(("set_device", device))

    def host_register(self, ptr, size):
        pass

    def host_unregister(self, ptr):
        pass

    def stream_create_nonblocking(self):
        return "fake_stream"

    def stream_destroy(self, stream):
        pass

    def stream_synchronize(self, stream):
        pass

    def event_create(self):
        return next(self._handles)

    def event_record(self, event, stream):
        pass

    def event_elapsed_ns(self, start_event, end_event):
        return 0

    def event_destroy(self, event):
        pass

    def memcpy_h2d_async(self, dst_ptr, src_ptr, size, stream):
        pass

    def memcpy_d2h_async(self, dst_ptr, src_ptr, size, stream):
        pass
