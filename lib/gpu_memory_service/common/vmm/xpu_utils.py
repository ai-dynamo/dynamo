# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU (Intel) VMM helpers for GMS — SYCL-only backend.

All device discovery, queue/stream management, memcpy, pointer validation,
virtual-memory management, physical memory, IPC, and host-register operations
are provided by the ``_sycl_vmm`` pybind11 extension compiled with
``icpx -fsycl``.  No ``dpctl`` dependency.
"""

from __future__ import annotations

import logging

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.vmm.device import VMMDevice

try:
    from gpu_memory_service.common.vmm import _sycl_vmm
except ImportError:

    class _MissingSyclVmm:
        """Stub that raises on any attribute access."""

        def __getattr__(self, name: str):
            raise RuntimeError(
                "The _sycl_vmm native extension is required for XPU support "
                "but is not installed.  Build it with icpx -fsycl; see "
                "common/vmm/_sycl_vmm/CMakeLists.txt."
            )

    _sycl_vmm = _MissingSyclVmm()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Access-mode mapping: GrantedLockType → int expected by the C++ module.
_ACCESS_MODE = {
    GrantedLockType.RO: 1,  # address_access_mode::read
    GrantedLockType.RW: 2,  # address_access_mode::read_write
}

# Pointer-type constants matching the C++ module.
_PTR_TYPE_VMM_MAPPED = 4


class XpuVMM(VMMDevice):
    """``VMMDevice`` implementation backed by ``_sycl_vmm`` (SYCL native)."""

    def __init__(self) -> None:
        self._active_device: int = 0
        # Eagerly initialize SYCL so that the runtime's global state is
        # properly set up before process exit.  Without this, loading the
        # _sycl_vmm .so alongside PyTorch XPU (torch.xpu.is_available)
        # causes a SYCL teardown crash (UR_RESULT_ERROR_UNINITIALIZED).
        _sycl_vmm.ensure_initialized()

    # ----- driver lifecycle -------------------------------------------------

    def ensure_initialized(self) -> None:
        _sycl_vmm.ensure_initialized()

    def synchronize(self) -> None:
        # Device-wide barrier: drain ALL queues
        import torch

        torch.xpu.synchronize()

    # ----- discovery / sizing -----------------------------------------------

    def list_devices(self) -> list[int]:
        return list(range(_sycl_vmm.device_count()))

    def device_memory_info(self, device: int) -> tuple[int, int]:
        # zePhysicalMemCreate allocations not reflected in
        # free-memory queries; subtract internally tracked VMM usage.
        # TODO: remove vmm_used subtraction post gpu runtime updated.
        import torch

        free_bytes, total_bytes = torch.xpu.mem_get_info(device)
        vmm_used = _sycl_vmm.total_allocated_bytes()
        free_bytes = max(0, free_bytes - vmm_used)
        return (free_bytes, total_bytes)

    def get_allocation_granularity(self, device: int) -> int:
        return _sycl_vmm.get_mem_granularity(device)

    # ----- physical memory --------------------------------------------------

    def create_tolerate_oom(self, size: int, device: int) -> tuple[bool, int]:
        return _sycl_vmm.physical_mem_create(device, size, enable_ipc=True)

    def release(self, handle: int) -> None:
        _sycl_vmm.physical_mem_release(handle)

    # ----- shareable-handle export / import ---------------------------------

    def export_to_shareable_handle(self, handle: int) -> int:
        return _sycl_vmm.ipc_export_fd(handle)

    def import_shareable_handle_close_fd(self, fd: int, import_size: int = 0) -> int:
        # L0 takes ownership of the FD on success (closes it internally).
        # On failure the FD is NOT consumed — close it here to avoid leaks.
        try:
            return _sycl_vmm.ipc_import_fd(
                fd, dev_idx=self._active_device, import_size=import_size
            )
        except BaseException:
            import os

            try:
                os.close(fd)
            except OSError:
                pass
            raise

    # ----- virtual address space + mapping ----------------------------------

    def address_reserve(self, size: int, granularity: int) -> int:
        # SYCL reserve_virtual_mem handles alignment internally based on
        # the context's minimum granularity; we pass the requested size.
        return _sycl_vmm.reserve_virtual_mem(size)

    def address_free(self, va: int, size: int) -> None:
        _sycl_vmm.free_virtual_mem(va, size)

    def map(self, va: int, size: int, handle: int) -> None:
        _sycl_vmm.physical_mem_map(handle, va, size, mode=2)  # read_write

    def unmap(self, va: int, size: int) -> None:
        _sycl_vmm.unmap(va, size)

    def set_access(
        self, va: int, size: int, device: int, access: GrantedLockType
    ) -> None:
        mode = _ACCESS_MODE[access]
        _sycl_vmm.set_access_mode(va, size, device, mode)

    # ----- pointer validation -----------------------------------------------

    def validate_pointer(self, va: int) -> None:
        ptype = _sycl_vmm.get_pointer_type(va, dev_idx=self._active_device)
        # Accept USM device (1) or VMM-mapped (4).
        if ptype not in (1, _PTR_TYPE_VMM_MAPPED):
            raise ValueError(
                f"Pointer 0x{va:x} is not a valid device/VMM pointer " f"(type={ptype})"
            )

    # ----- runtime helpers --------------------------------------------------

    def runtime_check_result(self, result, name: str) -> None:
        # SYCL uses exceptions rather than return codes; the C++ module
        # translates sycl::exception to Python exceptions.  This method
        # exists for API parity with CUDA where cuResult must be checked.
        pass

    def runtime_set_device(self, device: int) -> None:
        self._active_device = device
        _sycl_vmm.set_device(device)

    def host_register(self, ptr: int, size: int) -> None:
        _sycl_vmm.host_register(ptr, size)

    def host_unregister(self, ptr: int) -> None:
        _sycl_vmm.host_unregister(ptr)

    def stream_create_nonblocking(self):
        return _sycl_vmm.stream_create(dev_idx=self._active_device)

    def stream_destroy(self, stream) -> None:
        _sycl_vmm.stream_destroy(stream)

    def stream_synchronize(self, stream) -> None:
        _sycl_vmm.stream_synchronize(stream)

    def memcpy_h2d_async(self, dst_ptr: int, src_ptr: int, size: int, stream) -> None:
        _sycl_vmm.memcpy_async(dst_ptr, src_ptr, size, stream)

    def memcpy_d2h_async(self, dst_ptr: int, src_ptr: int, size: int, stream) -> None:
        _sycl_vmm.memcpy_async(dst_ptr, src_ptr, size, stream)
