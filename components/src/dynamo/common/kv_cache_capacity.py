# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime metadata for authoritative KV cache token capacity."""

from __future__ import annotations

import math
from collections.abc import Mapping

KV_CACHE_CAPACITY_RUNTIME_KEY = "kv_cache_capacity"


def kv_cache_capacity(total_tokens: object) -> dict[str, int] | None:
    """Build runtime metadata from an authoritative KV cache token capacity."""
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


def get_kv_cache_capacity_tokens(runtime_data: object) -> int | None:
    """Read authoritative KV cache capacity from worker runtime metadata."""
    if not isinstance(runtime_data, Mapping):
        return None
    capacity = runtime_data.get(KV_CACHE_CAPACITY_RUNTIME_KEY)
    if not isinstance(capacity, Mapping):
        return None
    payload = kv_cache_capacity(capacity.get("total_tokens"))
    return payload["total_tokens"] if payload is not None else None
