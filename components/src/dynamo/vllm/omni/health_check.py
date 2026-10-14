# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni canary configuration and protocol completion checks."""

import os
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from dynamo.common.utils.output_modalities import RequestType, parse_request_type
from dynamo.health_check import HEALTH_CHECK_KEY, load_health_check_from_env


def build_omni_health_check_payload(output_modalities: list[str]) -> dict[str, Any]:
    """Reject invalid overrides instead of silently downgrading to native health.

    Schema validation cannot establish that a request is supported by a particular
    checkpoint. Operators must first exercise their inference canary successfully
    with that model, including any required reference media or voice parameters.
    """
    payload = load_health_check_from_env()
    if "DYN_HEALTH_CHECK_PAYLOAD" in os.environ and payload is None:
        raise ValueError(
            "Invalid DYN_HEALTH_CHECK_PAYLOAD: expected a JSON object or a readable "
            "@file containing one; refusing to fall back to native-only health"
        )
    payload = {**(payload or {}), HEALTH_CHECK_KEY: True}
    if len(payload) > 1:
        try:
            _, request_type = parse_request_type(payload, output_modalities)
            if request_type == RequestType.CHAT_COMPLETION:
                messages = ChatCompletionRequest.model_validate(payload).messages
                content = next(
                    (
                        message.get("content")
                        for message in reversed(messages)
                        if message["role"] == "user"
                    ),
                    None,
                )
                # Omni's chat input builder uses the last user's string content.
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        "A chat canary requires a nonempty user message with "
                        "string content"
                    )
        except ValueError as error:
            raise ValueError(
                "Invalid DYN_HEALTH_CHECK_PAYLOAD for the configured Omni request "
                "protocol; use a request verified with this model"
            ) from error
    return payload


def has_completed_output(chunk: dict[str, Any], request_type: RequestType) -> bool:
    """Whether a formatted response contains a terminal protocol result.

    This checks completion and nonempty output, not media quality or whether an
    external URL can be downloaded. The formatter owns media serialization.
    Errors must still be checked throughout the remainder of the stream.
    """
    if request_type == RequestType.CHAT_COMPLETION:
        choices = chunk.get("choices")
        if not choices:
            return False
        return all(
            choice.get("finish_reason") in ("stop", "length", "tool_calls")
            for choice in choices
        )

    if request_type in (RequestType.AUDIO_GENERATION, RequestType.VIDEO_GENERATION):
        if chunk.get("status") != "completed":
            return False
    elif chunk.get("status") not in (None, "completed"):
        return False

    data = chunk.get("data")
    return (
        isinstance(data, list)
        and bool(data)
        and all(
            isinstance(item, dict)
            and any(
                isinstance(item.get(field), str) and bool(item[field].strip())
                for field in ("b64_json", "url")
            )
            for item in data
        )
    )


def generation_error(chunk: dict[str, Any]) -> str | None:
    """Extract formatter errors before they can count as successful activity."""
    if (
        chunk.get("error")
        or chunk.get("status") in ("failed", "error")
        or any(
            choice.get("finish_reason") == "error"
            for choice in chunk.get("choices", [])
        )
    ):
        return str(chunk.get("error") or "generation returned an error")
    return None
