# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for extracting multimodal data from OpenAI chat messages."""

from typing import Any

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
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type")
            if part_type not in _MEDIA_CONTENT_TYPES:
                continue
            url = part.get(part_type, {}).get("url")
            if url:
                mm_data.setdefault(part_type, []).append({"Url": url})

    return mm_data or None
