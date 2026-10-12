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

    def __init__(
        self,
        devices: list[int] | None = None,
        granularity: int = 4096,
    ):
        self._handles = itertools.count(1000)
        self._vas = itertools.count(0x100000, 0x10000)
        self._devices = devices if devices is not None else [0]
        self._granularity = granularity
        self.calls: list[tuple] = []
        self.server_handles: set[int] = set()
        self.imports: set[int] = set()
        self.reservations: dict[int, int] = {}
        self.mapped: dict[int, tuple[int, int]] = {}
        self.access: dict[int, object] = {}

    def ensure_initialized(self):
        pass

    def synchronize(self):
        pass

    def list_devices(self):
        return self._devices

    def device_memory_info(self, device):
        return (8 * 1024**3, 16 * 1024**3)

    def get_allocation_granularity(self, device):
        return self._granularity

    def create_tolerate_oom(self, size, device):
        handle = next(self._handles)
        self.server_handles.add(handle)
        return True, handle

    def release(self, handle):
        self.server_handles.discard(handle)
        self.imports.discard(handle)

    def export_to_shareable_handle(self, handle):
        if handle not in self.server_handles:
            raise AssertionError("unknown server handle")
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        return read_fd

    def import_shareable_handle_close_fd(self, fd, import_size=0):
        os.close(fd)
        handle = next(self._handles)
        self.imports.add(handle)
        return handle

    def address_reserve(self, size, granularity):
        va = next(self._vas)
        self.reservations[va] = size
        return va

    def address_free(self, va, size):
        if self.reservations.pop(va) != size:
            raise AssertionError("reservation size mismatch")

    def map(self, va, size, handle):
        if handle not in self.imports and handle not in self.server_handles:
            raise AssertionError("unknown handle")
        self.mapped[va] = size, handle

    def unmap(self, va, size):
        if self.mapped.pop(va)[0] != size:
            raise AssertionError("mapping size mismatch")
        self.access.pop(va, None)

    def set_access(self, va, size, device, access):
        if self.mapped[va][0] != size:
            raise AssertionError("access size mismatch")
        self.access[va] = access

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

    def memcpy_h2d_async(self, dst_ptr, src_ptr, size, stream):
        pass

    def memcpy_d2h_async(self, dst_ptr, src_ptr, size, stream):
        pass
