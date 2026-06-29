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
