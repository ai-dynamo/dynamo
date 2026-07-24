# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful VMMDevice fake for GMS V1 ownership tests."""

from __future__ import annotations

import os

from gpu_memory_service.common.vmm import VMMDevice
from gpu_memory_service.v1.server.allocations import AllocationStore


class V1FakeVMM(VMMDevice):
    def __init__(self, granularity: int = 64):
        self.granularity = granularity
        self.next_server_handle = 10
        self.next_import = 100
        self.next_base = 0x100000
        self.server_handles: set[int] = set()
        self.imports: set[int] = set()
        self.reservations: dict[int, int] = {}
        self.mapped: dict[int, tuple[int, int]] = {}
        self.access: dict[int, object] = {}
        self.events: list[tuple[object, ...]] = []
        self.fail_access_call: int | None = None
        self.access_calls = 0
        self.fail_map_call: int | None = None
        self.map_calls = 0
        self.fail_unmap: set[int] = set()
        self.fail_release: dict[int, int] = {}

    def ensure_initialized(self):
        pass

    def synchronize(self):
        self.events.append(("synchronize",))

    def list_devices(self):
        return [0]

    def device_memory_info(self, device):
        return 1 << 30, 2 << 30

    def get_allocation_granularity(self, device):
        return self.granularity

    def create_tolerate_oom(self, size, device):
        if size % self.granularity:
            raise AssertionError("unaligned fake allocation")
        handle = self.next_server_handle
        self.next_server_handle += 1
        self.server_handles.add(handle)
        self.events.append(("create", size, device, handle))
        return True, handle

    def release(self, handle):
        self.events.append(("release", handle))
        remaining = self.fail_release.get(handle, 0)
        if remaining:
            self.fail_release[handle] = remaining - 1
            raise RuntimeError("release failed")
        if handle >= 100:
            self.imports.remove(handle)
        else:
            self.server_handles.remove(handle)

    def export_to_shareable_handle(self, handle):
        if handle not in self.server_handles:
            raise AssertionError("unknown server handle")
        return os.open("/dev/null", os.O_RDONLY)

    def import_shareable_handle_close_fd(self, fd):
        try:
            handle = self.next_import
            self.next_import += 1
            self.imports.add(handle)
            self.events.append(("import", handle))
            return handle
        finally:
            os.close(fd)

    def address_reserve(self, size, granularity):
        if granularity != self.granularity:
            raise AssertionError("wrong granularity")
        base = self.next_base
        self.next_base += size + 0x1000
        self.reservations[base] = size
        self.events.append(("reserve", base, size))
        return base

    def address_free(self, va, size):
        self.events.append(("address_free", va, size))
        if va in self.mapped:
            raise RuntimeError("reservation remains mapped")
        if self.reservations.pop(va) != size:
            raise AssertionError("reservation size mismatch")

    def map(self, va, size, handle):
        self.map_calls += 1
        if self.map_calls == self.fail_map_call:
            raise RuntimeError("map failed")
        if handle not in self.imports:
            raise AssertionError("unknown import")
        self.mapped[va] = size, handle
        self.events.append(("map", va, size))

    def unmap(self, va, size):
        self.events.append(("unmap", va, size))
        if va in self.fail_unmap:
            self.fail_unmap.remove(va)
            raise RuntimeError("unmap failed")
        if self.mapped.pop(va)[0] != size:
            raise AssertionError("mapping size mismatch")
        self.access.pop(va, None)

    def set_access(self, va, size, device, access):
        self.access_calls += 1
        self.events.append(("access", va, size, device, access))
        if self.access_calls == self.fail_access_call:
            raise RuntimeError("access failed")
        if self.mapped[va][0] != size:
            raise AssertionError("access size mismatch")
        self.access[va] = access

    def validate_pointer(self, va):
        pass

    def runtime_check_result(self, result, name):
        pass

    def runtime_set_device(self, device):
        self.events.append(("device", device))

    def host_register(self, ptr, size):
        pass

    def host_unregister(self, ptr):
        pass

    def stream_create_nonblocking(self):
        return object()

    def stream_destroy(self, stream):
        pass

    def stream_synchronize(self, stream):
        pass

    def memcpy_h2d_async(self, dst_ptr, src_ptr, size, stream):
        pass

    def memcpy_d2h_async(self, dst_ptr, src_ptr, size, stream):
        pass


class V1FakeService:
    def __init__(self, store: AllocationStore):
        self.store = store
        self.events: list[tuple[object, ...]] = []
        self.connected = True

    def hello(self):
        if not self.connected:
            raise RuntimeError("client is disconnected")
        self.events.append(("hello",))
        return self.store.hello()

    def allocate(self, allocation_id, aligned_size):
        if not self.connected:
            raise RuntimeError("client is disconnected")
        self.events.append(("allocate", allocation_id))
        return self.store.allocate(allocation_id, aligned_size)

    def export(self, allocation_id):
        if not self.connected:
            raise RuntimeError("client is disconnected")
        self.events.append(("export", allocation_id))
        return self.store.export(allocation_id)

    def free(self, allocation_id):
        if not self.connected:
            raise RuntimeError("client is disconnected")
        self.events.append(("free", allocation_id))
        return self.store.free(allocation_id)

    def disconnect(self):
        self.events.append(("disconnect",))
        self.connected = False

    def reconnect(self):
        self.events.append(("reconnect",))
        self.connected = True
        return self.store.hello()
