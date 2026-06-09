# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRC32 over a raw memory range.

The daemon computes a CRC of each host-tier slot after D2H + sync
(offload integrity) and verifies it before H2D (restore integrity).
This catches:
  - DMA tearing / partial writes on the offload side
  - Bit-flips in pinned RAM between offload and restore
  - Stale-pointer reads (e.g., slot index races) — the CRC won't
    match for unrelated bytes

Implementation: prefer the SIMD-accelerated Rust path (gms_rust_ring,
~25 GB/s on modern x86), fall back to stdlib zlib.crc32 (~500 MB/s).
Both produce the same CRC32 (IEEE 802.3 polynomial).
"""

from __future__ import annotations

import ctypes
import zlib

try:
    from gms_rust_ring import crc32_at_ptr as _rust_crc32_at_ptr

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def crc32_at_ptr(ptr: int, size: int) -> int:
    """CRC32 (IEEE) of `size` bytes starting at raw pointer `ptr`.

    Caller MUST hold a reference to the memory keeping it live for
    the duration of this call. In daemon use, the pointer is a
    cudaHostAlloc'd pinned buffer owned by a HostTier slot.

    Returns the 32-bit CRC as a Python int."""
    if size <= 0:
        return 0
    if _HAS_RUST:
        return int(_rust_crc32_at_ptr(int(ptr), int(size)))
    # Fallback: zero-copy view via ctypes.
    buf = (ctypes.c_ubyte * size).from_address(int(ptr))
    return zlib.crc32(buf) & 0xFFFFFFFF
