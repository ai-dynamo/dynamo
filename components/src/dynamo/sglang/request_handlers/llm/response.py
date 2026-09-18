# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project SGLang terminal metadata into Dynamo's common response fields."""

from typing import Any, Mapping, TypedDict

from dynamo.common.utils.engine_response import normalize_finish_reason

_DYNAMO_FINISH_REASONS = {
    "eos",
    "length",
    "stop",
    "error",
    "cancelled",
    "content_filter",
}


class SglangTerminalFields(TypedDict, total=False):
    finish_reason: str
    stop_reason: Any
    completion_usage: dict[str, Any]


def extract_sglang_stop_reason(
    finish_reason: Mapping[str, Any] | None,
    user_stop_token_ids: set[int] | None = None,
) -> Any | None:
    """Extract SGLang's matched stop value for Dynamo's stop_reason field."""

    if not finish_reason:
        return None

    matched = finish_reason.get("matched")
    if isinstance(matched, bool):
        return None
    if isinstance(matched, str):
        return matched
    if isinstance(matched, int):
        if user_stop_token_ids is not None and matched not in user_stop_token_ids:
            return None
        return matched
    if isinstance(matched, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in matched
    ):
        if user_stop_token_ids is not None and any(
            item not in user_stop_token_ids for item in matched
        ):
            return None
        return matched

    return None


def _completion_usage(meta_info: Mapping[str, Any]) -> dict[str, Any] | None:
    input_tokens = meta_info.get("prompt_tokens")
    completion_tokens = meta_info.get("completion_tokens")
    if input_tokens is None or completion_tokens is None:
        return None

    usage: dict[str, Any] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": input_tokens + completion_tokens,
    }
    cached_tokens = meta_info.get("cached_tokens")
    if cached_tokens is not None and cached_tokens > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return usage


def project_sglang_terminal(
    meta_info: Mapping[str, Any],
    user_stop_token_ids: set[int] | None = None,
    *,
    opaque: bool = False,
) -> SglangTerminalFields:
    """Return common terminal fields without gating opaque native responses."""

    finish_reason = meta_info.get("finish_reason")
    if not finish_reason:
        return {}

    fields = SglangTerminalFields()
    completion_usage = _completion_usage(meta_info)
    if completion_usage is not None:
        fields["completion_usage"] = completion_usage

    if not isinstance(finish_reason, Mapping):
        if opaque:
            return fields
        raise TypeError("SGLang finish_reason must be a mapping")

    if opaque:
        finish_type = finish_reason.get("type")
        if not isinstance(finish_type, str):
            return fields
        normalized = normalize_finish_reason(finish_type)
        if normalized not in _DYNAMO_FINISH_REASONS:
            return fields
    else:
        normalized = normalize_finish_reason(finish_reason["type"])

    fields["finish_reason"] = normalized
    stop_reason = extract_sglang_stop_reason(finish_reason, user_stop_token_ids)
    if stop_reason is not None:
        fields["stop_reason"] = stop_reason
    return fields
