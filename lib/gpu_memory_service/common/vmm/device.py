# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract VMM device base class.

Defines the per-vendor virtual-memory-management surface that GMS depends on.
Method names are vendor-neutral verbs (``map``, ``address_reserve``,
``create_tolerate_oom``, etc.).

All derived classes must implement every abstract method.

"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gpu_memory_service.common.locks import GrantedLockType


class VMMDevice(ABC):
    """Per-vendor virtual-memory-management device contract.

    A device instance is obtained via ``get_vmm()``
    and used by GMS to allocate physical memory, export/import shareable
    handles for cross-process sharing, reserve and map virtual addresses.

    All subclasses must implement every abstract method; instantiation will
    raise ``TypeError`` if any are missing.

    """

    # ----- driver lifecycle -------------------------------------------------

    @abstractmethod
    def ensure_initialized(self) -> None:
        """Initialize the underlying driver. Idempotent."""

    @abstractmethod
    def synchronize(self) -> None:
        """Block until all in-flight device work for the current context completes."""

    # ----- discovery / sizing -----------------------------------------------

    @abstractmethod
    def list_devices(self) -> list[int]:
        """Return device indices visible to this process."""

    @abstractmethod
    def device_memory_info(self, device: int) -> tuple[int, int]:
        """Return ``(free_bytes, total_bytes)`` for ``device``."""

    @abstractmethod
    def get_allocation_granularity(self, device: int) -> int:
        """Minimum allocation granularity in bytes for ``device``."""

    # ----- physical memory --------------------------------------------------

    @abstractmethod
    def create_tolerate_oom(self, size: int, device: int) -> tuple[bool, int]:
        """Allocate physical memory of exactly ``size`` bytes on ``device``.

        Returns ``(allocated, handle)``. ``allocated`` is ``False`` and
        ``handle`` is ``0`` on OOM; on any other error the implementation
        raises.
        """

    @abstractmethod
    def release(self, handle: int) -> None:
        """Release a physical memory handle returned by ``create_tolerate_oom``."""

    # ----- shareable-handle export / import ---------------------------------

    @abstractmethod
    def export_to_shareable_handle(self, handle: int) -> int:
        """Return a POSIX FD that can be passed cross-process via SCM_RIGHTS."""

    @abstractmethod
    def import_shareable_handle_close_fd(self, fd: int) -> int:
        """Import a shareable FD into a local physical-memory handle.

        The FD is closed on success or failure (matches the existing CUDA
        helper's contract).
        """

    # ----- virtual address space + mapping ----------------------------------

    @abstractmethod
    def address_reserve(self, size: int, granularity: int) -> int:
        """Reserve a contiguous VA range. Returns the base VA."""

    @abstractmethod
    def address_free(self, va: int, size: int) -> None:
        """Release a VA reservation."""

    @abstractmethod
    def map(self, va: int, size: int, handle: int) -> None:
        """Bind the VA range to a physical handle."""

    @abstractmethod
    def unmap(self, va: int, size: int) -> None:
        """Unbind the VA range. The reservation itself is preserved."""

    @abstractmethod
    def set_access(
        self, va: int, size: int, device: int, access: GrantedLockType
    ) -> None:
        """Set device-side access permissions for a mapped VA range."""

    # ----- pointer validation -----------------------------------------------

    @abstractmethod
    def validate_pointer(self, va: int) -> None:
        """Best-effort check that ``va`` refers to a valid device allocation."""

    # ----- runtime helpers --------------------------------------------------

    @abstractmethod
    def runtime_check_result(self, result, name: str) -> None:
        """Check a device-runtime return code; raise on failure."""

    @abstractmethod
    def runtime_set_device(self, device: int) -> None:
        """Set the active device for the current thread."""

    @abstractmethod
    def host_register(self, ptr: int, size: int) -> None:
        """Pin host memory for DMA access."""

    @abstractmethod
    def host_unregister(self, ptr: int) -> None:
        """Unpin previously registered host memory."""

    @abstractmethod
    def stream_create_nonblocking(self):
        """Create a non-blocking execution stream. Returns an opaque handle."""

    @abstractmethod
    def stream_destroy(self, stream) -> None:
        """Destroy an execution stream."""

    @abstractmethod
    def stream_synchronize(self, stream) -> None:
        """Block until all work on ``stream`` completes."""

    @abstractmethod
    def event_create(self):
        """Create a timing event. Returns an opaque handle."""

    @abstractmethod
    def event_record(self, event, stream) -> None:
        """Record ``event`` on ``stream``."""

    @abstractmethod
    def event_elapsed_ns(self, start_event, end_event) -> int:
        """Return elapsed device time between two recorded events."""

    @abstractmethod
    def event_destroy(self, event) -> None:
        """Destroy a timing event."""

    @abstractmethod
    def memcpy_h2d_async(
        self,
        dst_ptr: int,
        src_ptr: int,
        size: int,
        stream,
    ) -> None:
        """Async host-to-device copy on ``stream``."""

    @abstractmethod
    def memcpy_d2h_async(
        self,
        dst_ptr: int,
        src_ptr: int,
        size: int,
        stream,
    ) -> None:
        """Async device-to-host copy on ``stream``."""
