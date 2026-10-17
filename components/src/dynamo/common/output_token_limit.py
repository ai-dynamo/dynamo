# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side output ceilings, applied after client sampling overrides."""

from typing import Any, overload

from dynamo.llm.exceptions import InvalidArgument


def cap_output_tokens(
    max_tokens: int | None,
    limit: int | None,
    min_tokens: int | None = None,
) -> int | None:
    """Preserve smaller budgets and reject a floor incompatible with the ceiling."""
    if limit is None:
        return max_tokens
    capped = limit if max_tokens is None else min(max_tokens, limit)
    if min_tokens is not None and min_tokens > capped:
        raise InvalidArgument(
            f"min_tokens ({min_tokens}) exceeds the effective maximum output tokens ({capped})"
        )
    return capped


@overload
def cap_sglang_output_tokens(
    sampling_params: dict[str, Any], limit: int | None, min_tokens: int | None = None
) -> dict[str, Any]:
    pass


@overload
def cap_sglang_output_tokens(
    sampling_params: list[dict[str, Any]],
    limit: int | None,
    min_tokens: int | None = None,
) -> list[dict[str, Any]]:
    pass


@overload
def cap_sglang_output_tokens(
    sampling_params: None, limit: int | None, min_tokens: int | None = None
) -> dict[str, Any] | None:
    pass


def cap_sglang_output_tokens(
    sampling_params: dict[str, Any] | list[dict[str, Any]] | None,
    limit: int | None,
    min_tokens: int | None = None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Cap every member of a native batch without changing the caller's payload."""
    if limit is None:
        return sampling_params
    if isinstance(sampling_params, list):
        return [cap_sglang_output_tokens(params, limit) for params in sampling_params]
    params = dict(sampling_params or {})
    params["max_new_tokens"] = cap_output_tokens(
        params.get("max_new_tokens"),
        limit,
        min_tokens if min_tokens is not None else params.get("min_new_tokens"),
    )
    return params
