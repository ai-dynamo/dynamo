# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from dynamo.llm.exceptions import InvalidArgument


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


def nvext_extra_field_requested(request: Mapping[str, Any], field: str) -> bool:
    """Return whether the raw or preprocessed request opted into an nvext field."""
    extra_args = request.get("extra_args") or {}
    for source in (
        request.get("nvext"),
        extra_args.get("nvext") if isinstance(extra_args, Mapping) else None,
    ):
        if not isinstance(source, Mapping):
            continue
        extra_fields = source.get("extra_fields")
        if isinstance(extra_fields, list) and field in extra_fields:
            return True
    return False


def routed_experts_start_len(request: Mapping[str, Any]) -> int | None:
    """Return the top-level routed_experts_start_len forwarded by the frontend."""
    extra_args = request.get("extra_args") or {}
    sampling = (
        extra_args.get("sampling_options") if isinstance(extra_args, Mapping) else None
    )
    if not isinstance(sampling, Mapping):
        return None
    start_len = sampling.get("routed_experts_start_len")
    if start_len is not None and (
        isinstance(start_len, bool) or not isinstance(start_len, int) or start_len < 0
    ):
        raise InvalidArgument("routed_experts_start_len must be a non-negative integer")
    return start_len
