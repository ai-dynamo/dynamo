# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for multimodal inputs in disaggregated prefill/decode (P/D) serving.

In disaggregated serving the prefill worker computes the KV cache (including the
vision context) and transfers it to the decode worker over NIXL. The decode
worker still has to reproduce the same *token layout* — the image placeholder
must expand to the same number of patch tokens prefill used — so the transferred
KV indices and mRoPE positions line up.

Both workers achieve this by feeding the image/video URLs to ``async_generate``;
SGLang's multimodal path downloads + encodes them and expands the placeholder
tokens identically on each side. This matches upstream SGLang's own native PD
behavior (its router dispatches the request to both prefill and decode, each of
which processes the media independently). The decode worker's vision encode is
redundant with the transferred KV, but guarantees the layout is consistent.

Follow-up (not implemented here): decode could skip the GPU vision tower by
feeding SGLang a synthetic ``precomputed_embedding`` sized from a grid forwarded
by prefill, mirroring the dynamo vLLM backend. On SGLang that requires either a
frontend change to forward the grid (prefill runs asynchronously under the
bootstrap router, so it cannot hand decode the grid) or re-deriving the grid in
decode. Tracked as a separate optimization.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

IMAGE_URL_KEY = "image_url"
VIDEO_URL_KEY = "video_url"


def extract_media_urls(
    mm_data: Optional[Dict[str, Any]], media_key: str
) -> list[str] | None:
    """Normalize multimodal URL items from the frontend wire format.

    The Rust frontend populates ``multi_modal_data`` as
    ``{"image_url": [{"Url": "..."}, ...], "video_url": [...]}``. Plain string
    items are also accepted for forward/backward compatibility. Returns ``None``
    when the modality is absent so callers can pass it straight through to
    ``async_generate`` (which treats ``None`` as "no media").

    Malformed or unsupported payloads raise ``ValueError`` rather than silently
    degrading to text. In disaggregated serving both workers depend on this
    helper, so a dropped item would not just be a local parse bug — it would
    desync the prefill/decode token layout and corrupt the answer. This mirrors
    the explicit wire-variant validation in ``MultimodalEncodeWorkerHandler``;
    in particular, frontend-decoded (``Decoded``) media is rejected because the
    disaggregated path is URL-passthrough only.
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
    """Build the ``image_data``/``video_data`` kwargs for a disaggregated
    worker's ``async_generate`` call.

    Both the prefill and decode workers call this so they extract the media
    identically and reproduce the same expanded token layout — a divergence
    between the two sides would misalign the transferred KV (the exact failure
    this module guards against). Always returns both keys (value ``None`` when a
    modality is absent), matching the aggregated path's long-standing call shape;
    SGLang treats ``None`` as "no media".
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
