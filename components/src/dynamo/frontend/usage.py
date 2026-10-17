# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def update_usage_metadata(metadata: dict[str, Any], usage: dict[str, Any]) -> None:
    """Retain known prompt/cache details across sparse backend reports."""
    for field in (
        "prompt_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    ):
        if usage.get(field) is not None:
            metadata[field] = usage[field]


def response_usage(
    request: dict[str, Any],
    metadata: dict[str, Any],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    finished: bool,
) -> dict[str, Any] | None:
    """Project locally counted usage according to the client's streaming options."""
    if not metadata:
        return None
    if request.get("stream", False):
        options = request.get("stream_options") or {}
        if not options.get("include_usage", False):
            return None
        if not finished and not options.get("continuous_usage_stats", False):
            return None
    prompt_tokens = metadata.get("prompt_tokens", prompt_tokens)
    return {
        **metadata,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
