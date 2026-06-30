# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, Optional


def request_cache_salt(request: Mapping[str, Any]) -> Optional[str]:
    """Return cache_salt using routing hints before legacy extra_args."""
    routing = request.get("routing") or {}
    if isinstance(routing, dict):
        cache_salt = routing.get("cache_salt")
        if cache_salt is not None:
            return cache_salt

    extra_args = request.get("extra_args") or {}
    nvext = extra_args.get("nvext") if isinstance(extra_args, dict) else None
    if isinstance(nvext, dict):
        cache_salt = nvext.get("cache_salt")
        if cache_salt is not None:
            return cache_salt

    return None


def stored_event_cache_salt(data: Mapping[str, Any]) -> Optional[str]:
    """Extract one cache salt from a TRT-LLM stored-event payload.

    TRT-LLM 1.3 serializes ``cache_salt`` on each item in ``data["blocks"]``.
    Keep the parent-level lookup as a compatibility fallback, but fail closed
    if an event combines blocks from different cache namespaces.
    """
    cache_salts: set[str] = set()

    parent_cache_salt = data.get("cache_salt")
    if parent_cache_salt is not None:
        cache_salts.add(parent_cache_salt)

    blocks = data.get("blocks") or []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        block_cache_salt = block.get("cache_salt")
        if block_cache_salt is not None:
            cache_salts.add(block_cache_salt)

    if len(cache_salts) > 1:
        raise ValueError("stored KV event contains conflicting cache_salt values")

    return next(iter(cache_salts), None)
