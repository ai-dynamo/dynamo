# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime metadata for backend-native KV offloading capacity."""

from __future__ import annotations

import math
from collections.abc import Mapping

NATIVE_OFFLOADING_CAPACITY_RUNTIME_KEY = "native_offloading_capacity"


def native_offloading_capacity(total_tokens: object) -> dict[str, int] | None:
    """Build runtime metadata from an authoritative backend token capacity."""
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, (int, float))
        or total_tokens <= 0
    ):
        return None
    if isinstance(total_tokens, float) and not math.isfinite(total_tokens):
        return None
    tokens = int(total_tokens)
    return {"total_tokens": tokens} if tokens > 0 else None


def get_native_offloading_capacity_tokens(runtime_data: object) -> int | None:
    """Read native offloading capacity from a worker's runtime metadata."""
    if not isinstance(runtime_data, Mapping):
        return None
    capacity = runtime_data.get(NATIVE_OFFLOADING_CAPACITY_RUNTIME_KEY)
    if not isinstance(capacity, Mapping):
        return None
    payload = native_offloading_capacity(capacity.get("total_tokens"))
    return payload["total_tokens"] if payload is not None else None
