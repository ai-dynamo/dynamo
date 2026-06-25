# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service — VMM device abstraction.

GMS depends on a per-vendor virtual-memory-management surface
(allocate physical memory, export/import shareable handles, reserve and
map virtual addresses, ...).

The VMM instance is process-global and immutable once initialized.
Call ``init_vmm(device_type)`` once at process startup, then use
``get_vmm()`` anywhere a VMMDevice is needed.

"""

from __future__ import annotations

import threading
from enum import Enum

from .device import VMMDevice


class VMMDeviceType(str, Enum):
    """Identify which vendor's VMM driver a GMS instance should use."""

    CUDA = "cuda"
    XPU = "xpu"

    @classmethod
    def from_str(cls, value: str) -> "VMMDeviceType":
        try:
            return cls(value.lower())
        except ValueError as exc:
            valid = ", ".join(b.value for b in cls)
            raise ValueError(
                f"Unknown VMM device type {value!r}; expected one of: {valid}"
            ) from exc


# ---------------------------------------------------------------------------
# Process-global singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_vmm_instance: VMMDevice | None = None
_vmm_device_type: VMMDeviceType | None = None


def init_vmm(device_type: VMMDeviceType) -> None:
    """Initialize the process-global VMM singleton. Idempotent for same kind."""
    global _vmm_instance, _vmm_device_type
    with _lock:
        if _vmm_instance is not None:
            if _vmm_device_type != device_type:
                raise RuntimeError(
                    f"VMM already initialized as {_vmm_device_type!r}; "
                    f"cannot reinitialize as {device_type!r}"
                )
            return
        _vmm_device_type = device_type
        _vmm_instance = _create_vmm(device_type)


def get_vmm() -> VMMDevice:
    """Return the process-global VMM singleton.

    Raises ``RuntimeError`` if ``init_vmm()`` has not been called.
    """
    inst = _vmm_instance
    if inst is None:
        raise RuntimeError("VMM not initialized; call init_vmm() at startup")
    return inst


def get_vmm_device_type() -> VMMDeviceType:
    """Return the active device type.

    Raises ``RuntimeError`` if ``init_vmm()`` has not been called.
    """
    kind = _vmm_device_type
    if kind is None:
        raise RuntimeError("VMM not initialized; call init_vmm() at startup")
    return kind


def _create_vmm(device_type: VMMDeviceType) -> VMMDevice:
    """Construct the appropriate VMMDevice implementation."""
    if device_type is VMMDeviceType.CUDA:
        from .cuda_utils import CudaVMM

        return CudaVMM()

    if device_type is VMMDeviceType.XPU:
        raise NotImplementedError("'xpu' VMM backend is not implemented yet")

    raise ValueError(f"Unhandled VMM device type: {device_type!r}")


# ---------------------------------------------------------------------------
# Test support
# ---------------------------------------------------------------------------


def _reset_vmm_singleton() -> None:
    """Reset the singleton for test isolation. NOT for production use."""
    global _vmm_instance, _vmm_device_type
    with _lock:
        _vmm_instance = None
        _vmm_device_type = None


__all__ = [
    "VMMDevice",
    "VMMDeviceType",
    "get_vmm",
    "get_vmm_device_type",
    "init_vmm",
]
