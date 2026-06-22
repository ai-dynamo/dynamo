# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal media extraction shared by the disaggregated prefill and decode
workers, so both feed identical image/video URLs to the engine and reproduce the
same token layout the transferred KV depends on.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

IMAGE_URL_KEY = "image_url"
VIDEO_URL_KEY = "video_url"


def extract_media_urls(
    mm_data: Optional[Dict[str, Any]], media_key: str
) -> list[str] | None:
    """Return the URLs under ``media_key`` (``{"Url": ...}`` or string items), or
    ``None`` if absent. Raises on malformed or frontend-decoded payloads rather
    than silently degrading a multimodal request to text.
    """
    if not mm_data:
        return None

    items = mm_data.get(media_key)
    if not items:
        return None
    if not isinstance(items, list):
        raise ValueError(
            f"{media_key} must be a list of URL items, got {type(items).__name__}"
        )

    urls: list[str] = []
    for item in items:
        if isinstance(item, str):
            urls.append(item)
            continue
        if isinstance(item, dict):
            url = item.get("Url")
            if isinstance(url, str):
                urls.append(url)
                continue
            if "Decoded" in item:
                raise ValueError(
                    f"Frontend-decoded media is not supported for disaggregated "
                    f"{media_key}; use URL-based inputs."
                )
        raise ValueError(f"Unsupported {media_key} item: {item!r}")

    return urls or None


def build_disagg_mm_kwargs(request: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``image_data``/``video_data`` kwargs for a disaggregated worker's
    ``async_generate`` call. Both keys are always present (``None`` when absent).
    """
    mm_data = request.get("multi_modal_data") or {}
    image_data = extract_media_urls(mm_data, IMAGE_URL_KEY)
    video_data = extract_media_urls(mm_data, VIDEO_URL_KEY)
    if image_data or video_data:
        logger.debug(
            "disaggregated multimodal request: images=%d, videos=%d",
            len(image_data or []),
            len(video_data or []),
        )
    return {"image_data": image_data, "video_data": video_data}
