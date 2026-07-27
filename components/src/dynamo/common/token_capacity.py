# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for token-capacity runtime metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping


def positive_int(value: object) -> int | None:
    """Return a finite positive value as an integer, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    result = int(value)
    return result if result > 0 else None


def token_capacity_payload(total_tokens: object) -> dict[str, int] | None:
    """Build a token-capacity runtime metadata payload."""
    tokens = positive_int(total_tokens)
    return {"total_tokens": tokens} if tokens is not None else None


def get_capacity_tokens(runtime_data: object, key: str) -> int | None:
    """Read a validated token capacity from runtime metadata."""
    if not isinstance(runtime_data, Mapping):
        return None
    capacity = runtime_data.get(key)
    if not isinstance(capacity, Mapping):
        return None
    return positive_int(capacity.get("total_tokens"))
