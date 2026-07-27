# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime metadata for authoritative KV cache token capacity."""

from __future__ import annotations

from dynamo.common.token_capacity import (
    get_capacity_tokens,
    token_capacity_payload,
)

KV_CACHE_CAPACITY_RUNTIME_KEY = "kv_cache_capacity"


def kv_cache_capacity(total_tokens: object) -> dict[str, int] | None:
    """Build runtime metadata from an authoritative KV cache token capacity."""
    return token_capacity_payload(total_tokens)


def get_kv_cache_capacity_tokens(runtime_data: object) -> int | None:
    """Read authoritative KV cache capacity from worker runtime metadata."""
    return get_capacity_tokens(runtime_data, KV_CACHE_CAPACITY_RUNTIME_KEY)
