#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Shared utilities for frontend chat processors (vLLM, SGLang)."""

import logging
import uuid
from typing import Any

_MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    """Generate a random 16-character hex UUID."""
    return f"{uuid.uuid4().int & _MASK_64_BITS:016x}"


def random_call_id() -> str:
    """Generate a random tool call ID in OpenAI format."""
    return f"call_{uuid.uuid4().int & _MASK_64_BITS:016x}"


def worker_warmup() -> bool:
    """Dummy task to ensure a ProcessPoolExecutor worker is fully initialized."""
    return True


class PreprocessError(Exception):
    """Raised by preprocess workers for user-facing errors (e.g., n!=1)."""

    def __init__(self, error_dict: dict[str, Any]):
        self.error_dict = error_dict
        super().__init__(str(error_dict))


# Content part types that carry media URLs, mapped to the key used in the
# multimodal data dict sent to the backend handler.
_MEDIA_CONTENT_TYPES = ("image_url", "audio_url", "video_url")


def extract_mm_urls(
    messages: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]] | None:
    """Extract multimodal URLs from OpenAI chat completion messages.

    Walks user message content arrays and collects ``image_url``, ``audio_url``,
    and ``video_url`` entries.  Returns them in the format expected by the
    backend handler's ``_extract_multimodal_data()``::

        {
            "image_url": [{"Url": "https://..."}, ...],
            "audio_url": [{"Url": "data:audio/wav;base64,..."}],
        }

    Returns ``None`` if no multimodal content is found.
    """
    mm_data: dict[str, list[dict[str, str]]] = {}

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type not in _MEDIA_CONTENT_TYPES:
                continue
            media_value = part.get(part_type)
            if not isinstance(media_value, dict):
                continue
            url = media_value.get("url")
            if isinstance(url, str) and url:
                mm_data.setdefault(part_type, []).append({"Url": url})

    return mm_data or None


def make_backend_error(engine_response: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-style error dict, guarding against None/missing message."""
    backend_msg = engine_response.get("message") or "unknown backend error"
    return {
        "error": {
            "message": backend_msg,
            "type": "backend_error",
        }
    }


def make_internal_error(request_id: str, detail: str | None = None) -> dict[str, Any]:
    """Build an OpenAI-style internal error dict with request-specific fallback."""
    message = detail or f"Invalid engine response for request {request_id}"
    return {
        "error": {
            "message": message,
            "type": "internal_error",
        }
    }


def handle_engine_error(
    engine_response: Any,
    request_id: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Classify an invalid engine response and return an OpenAI-style error dict.

    Called when engine_response is None or missing 'token_ids'.
    """
    if isinstance(engine_response, dict) and engine_response.get("status") == "error":
        err = make_backend_error(engine_response)
        logger.error(
            "Backend error for request %s: %s", request_id, err["error"]["message"]
        )
        return err
    logger.error(
        "No outputs from engine for request %s: %s", request_id, engine_response
    )
    return make_internal_error(request_id)
