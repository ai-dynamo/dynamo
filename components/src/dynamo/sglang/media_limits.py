# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-prompt raw-media limits for SGLang workers."""

import json
from collections.abc import Mapping
from typing import Any

from dynamo.llm.exceptions import InvalidArgument

_MEDIA_KEYS = {
    "image": ("image_url", "image_data"),
    "video": ("video_url", "video_data"),
    "audio": ("audio_url", "audio_data"),
}


def parse_mm_limits(value: str) -> dict[str, int]:
    """Parse vLLM-style modality limits at worker startup."""
    try:
        limits = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--limit-mm-per-prompt must be a JSON object") from exc
    if not isinstance(limits, dict):
        raise ValueError("--limit-mm-per-prompt must be a JSON object")
    for modality, limit in limits.items():
        if modality not in _MEDIA_KEYS:
            raise ValueError(f"--limit-mm-per-prompt: unknown modality {modality!r}")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(
                f"--limit-mm-per-prompt: {modality} limit must be a non-negative integer"
            )
    return limits


def raise_if_mm_limit_exceeded(
    request: Mapping[str, Any], limits: Mapping[str, int] | None
) -> None:
    """Reject excessive raw media before loading or engine dispatch."""
    if not limits:
        return
    media = request.get("multi_modal_data") or {}
    native = (request.get("extra_args") or {}).get("sglang_tito") or {}
    for modality, limit in limits.items():
        for source, key in zip((media, native), _MEDIA_KEYS[modality]):
            items = source.get(key)
            count = 0 if items is None else len(items) if isinstance(items, list) else 1
            if source is native and isinstance(items, list):
                # Native Generate has one canonical prompt; count nested media
                # groups too rather than letting an outer list hide its items.
                count = sum(
                    len(item) if isinstance(item, list) else 1 for item in items
                )
            if count > limit:
                raise InvalidArgument(
                    f"At most {limit} {modality}(s) may be provided in one prompt."
                )
