# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, Optional

from dynamo.sglang._compat import filter_supported_async_generate_kwargs


def request_cache_salt(request: Mapping[str, Any]) -> Optional[str]:
    """Return the first non-empty cache salt in routing-authoritative order."""
    extra_args = request.get("extra_args")
    sources = (
        request.get("routing"),
        request.get("nvext"),
        extra_args.get("nvext") if isinstance(extra_args, Mapping) else None,
    )
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        cache_salt = source.get("cache_salt")
        if cache_salt is None or cache_salt == "":
            continue
        if not isinstance(cache_salt, str):
            raise ValueError("cache_salt must be a string")
        return cache_salt
    return None


def cache_salt_kwargs(engine: Any, request: Mapping[str, Any]) -> dict[str, str]:
    """Build the required SGLang Engine kwarg for a cache-salted request."""
    cache_salt = request_cache_salt(request)
    if cache_salt is None:
        return {}

    kwargs = filter_supported_async_generate_kwargs(engine, {"cache_salt": cache_salt})
    if "cache_salt" not in kwargs:
        raise ValueError(
            "cache_salt requires Engine.async_generate to accept cache_salt "
            "and expose an inspectable signature"
        )
    return kwargs
