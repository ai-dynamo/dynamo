# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any


def request_cache_salt(request: Mapping[str, Any]) -> str | None:
    """Return the first non-empty cache salt, preferring resolved routing hints."""
    extra_args = request.get("extra_args") or {}
    for source in (
        request.get("routing"),
        extra_args.get("nvext") if isinstance(extra_args, Mapping) else None,
        request.get("nvext"),
        request,
    ):
        if isinstance(source, Mapping):
            salt = source.get("cache_salt")
            if salt:
                return salt
    return None
