# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful VMMDevice fake for GMS V1 ownership tests."""

from __future__ import annotations

import os

from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import VMMDevice
from gpu_memory_service.core.errors import FatalGMSError
from gpu_memory_service.core.server.gms import GMS


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
        self.export_fds: list[int] = []
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
        fd = os.open("/dev/null", os.O_RDONLY)
        self.export_fds.append(fd)
        return fd

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


class V1FakeSession:
    def __init__(
        self,
        gms: GMS,
        requested: RequestedLockType,
        events: list[tuple[object, ...]],
    ):
        self.gms = gms
        self.events = events
        self._server_session = gms.acquire(requested)
        self._connected = True
        self.events.append(("handshake", requested))

    @property
    def identity(self):
        return self.gms.identity

    @property
    def lock_type(self):
        return self._server_session.mode

    @property
    def is_connected(self):
        return self._connected

    def allocate(self, allocation_id, aligned_size):
        self.events.append(("allocate", allocation_id))
        self.gms.dispatch(
            self._server_session,
            "allocate",
            [allocation_id, aligned_size],
        )

    def export(self, allocation_id):
        self.events.append(("export", allocation_id))
        _, fd = self.gms.dispatch(
            self._server_session,
            "export",
            [allocation_id],
        )
        return fd

    def free(self, allocation_id):
        self.events.append(("free", allocation_id))
        self.gms.dispatch(self._server_session, "free", [allocation_id])

    def commit(self):
        self.events.append(("commit",))
        self.gms.dispatch(self._server_session, "commit", [])

    def close(self):
        if not self._connected:
            return
        self.events.append(("close", self._server_session.mode))
        self.gms.close(self._server_session)
        self._connected = False


class V1FakeSessionFactory:
    def __init__(self, gms: GMS, events: list[tuple[object, ...]]):
        self.gms = gms
        self.events = events
        self.sessions: list[V1FakeSession] = []

    def __call__(self, _path, requested, expected_identity=None):
        if expected_identity is not None and expected_identity != self.gms.identity:
            raise FatalGMSError("GMS V1 sidecar incarnation or physical GPU changed")
        session = V1FakeSession(self.gms, requested, self.events)
        self.sessions.append(session)
        return session
