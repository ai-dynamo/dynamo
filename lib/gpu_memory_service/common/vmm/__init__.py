# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service — VMM device abstraction.

GMS depends on a per-vendor virtual-memory-management surface
(allocate physical memory, export/import shareable handles, reserve and
map virtual addresses, …).

"""

from __future__ import annotations

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
                f"Unknown VMM device kind {value!r}; expected one of: {valid}"
            ) from exc


def get_vmm_device(device_kind: VMMDeviceType) -> VMMDevice:
    """Return the ``VMMDevice`` implementation for ``device_kind``."""

    if device_kind is VMMDeviceType.CUDA:
        from .cuda_utils import CudaVMM

        return CudaVMM()

    if device_kind is VMMDeviceType.XPU:
        raise NotImplementedError("'xpu' is not implemented yet")

    raise ValueError(f"Unhandled VMM device kind: {device_kind!r}")


__all__ = [
    "VMMDevice",
    "VMMDeviceType",
    "get_vmm_device",
]
