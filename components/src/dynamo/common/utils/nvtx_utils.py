# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight NVTX wrappers for Dynamo profiling.

Set DYN_NVTX=1 to enable markers; default is disabled (zero overhead).

Usage — same syntax as the bare nvtx module:

    from dynamo.common.utils import nvtx_utils as _nvtx

    rng = _nvtx.start_range("my:range", color="blue")
    ...
    _nvtx.end_range(rng)

    _nvtx.mark("my:event", color="navy")

When enabled, EventAttributes objects are cached internally by
(message, color) on first call, so subsequent calls avoid re-allocation
with no manual pre-allocation required by the caller.
"""
import os as _os

ENABLED: bool = bool(int(_os.getenv("DYN_NVTX", "0")))

if ENABLED:
    import nvtx as _nvtx_lib

    # Shared cache: (message, color) -> EventAttributes.
    # Populated lazily on first call; reused on every subsequent call.
    _attr_cache: dict = {}

    def start_range(message: str, color: str = "white"):
        """Start an NVTX range. EventAttributes are cached by (message, color)."""
        try:
            attr = _attr_cache[message, color]
        except KeyError:
            attr = _nvtx_lib.EventAttributes(message=message, color=color)
            _attr_cache[message, color] = attr
        return _nvtx_lib.start_range(attributes=attr)

    def end_range(rng) -> None:
        _nvtx_lib.end_range(rng)

    def mark(message: str, color: str = "white") -> None:
        """Emit an NVTX mark. EventAttributes are cached by (message, color)."""
        try:
            attr = _attr_cache[message, color]
        except KeyError:
            attr = _nvtx_lib.EventAttributes(message=message, color=color)
            _attr_cache[message, color] = attr
        _nvtx_lib.mark(attributes=attr)

else:
    # Pure Python no-ops: no C extension calls, no string allocations.
    # The ENV var is read once at import time — no per-call branch overhead.

    def start_range(message: str, color: str = "white"):  # type: ignore[misc]
        return None

    def end_range(rng) -> None:  # type: ignore[misc]
        pass

    def mark(message: str, color: str = "white") -> None:  # type: ignore[misc]
        pass
