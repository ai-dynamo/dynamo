# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared bounds and errors for GMS length-prefixed RPC frames."""

from __future__ import annotations

import os

DEFAULT_MAX_FRAME_BYTES = 64 << 20
MAX_FRAME_ENV = "GMS_RPC_MAX_FRAME_BYTES"


def _configured_max_frame_bytes() -> int:
    raw = os.environ.get(MAX_FRAME_ENV)
    if raw is None:
        return DEFAULT_MAX_FRAME_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{MAX_FRAME_ENV} must be an integer") from exc
    if not 0 < value <= 0xFFFFFFFF:
        raise RuntimeError(f"{MAX_FRAME_ENV} must be between 1 and 4294967295")
    return value


MAX_FRAME_BYTES = _configured_max_frame_bytes()


class FrameProtocolError(RuntimeError):
    """Base class for malformed length-prefixed frames."""


class FrameTooLargeError(FrameProtocolError):
    """A peer advertised a payload larger than the configured limit."""


class TruncatedFrameError(FrameProtocolError, EOFError):
    """A peer closed after sending only part of a frame."""


def validate_frame_length(length: int) -> None:
    if length > MAX_FRAME_BYTES:
        raise FrameTooLargeError(
            f"GMS RPC frame length {length} exceeds limit {MAX_FRAME_BYTES}"
        )


def truncated_frame(part: str, expected: int, received: int) -> TruncatedFrameError:
    return TruncatedFrameError(
        f"truncated GMS RPC {part}: expected {expected} bytes, received {received}"
    )
