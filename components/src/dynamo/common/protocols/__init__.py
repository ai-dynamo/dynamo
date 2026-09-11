# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared protocol types used across multiple Dynamo backends.

This module provides protocol types for various modalities:
- video_protocol: NvCreateVideoRequest, NvVideosResponse for video generation
"""

from typing import Any, Dict, Optional

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)

MEDIA_PASSTHROUGH_KEY = "media_passthrough"
"""Key under a media request's ``extra_args`` where the frontend nests
unknown top-level request fields (an OpenAI client's ``extra_body``)."""


class MediaPassthroughRejected(ValueError):
    """A request tried to pass a knob the deployment owns, not the caller."""


# Keys a caller must never set through the passthrough bucket. These name a
# thing to load or a safety/policy control, and the bucket is fully caller
# controlled, so honoring them would let a request point the engine at code
# to load or turn a guardrail off.
_RESERVED_PASSTHROUGH_KEYS = frozenset(
    {
        "extra_args",
        "guardrails",
        "frame_interpolation_model_path",
    }
)

# A knob whose name contains one of these is treated the same way: the first
# group names a fetch or load location, the second a safety/policy switch.
_UNSAFE_KEY_MARKERS = (
    "path",
    "checkpoint",
    "ckpt",
    "repo",
    "hub",
    "url",
    "uri",
    "local_dir",
    "guardrail",
    "safety",
    "nsfw",
    "moderation",
)


def _passthrough_key_rejected(key: str) -> bool:
    if key in _RESERVED_PASSTHROUGH_KEYS or key.startswith("_"):
        return True
    lowered = key.lower()
    return any(marker in lowered for marker in _UNSAFE_KEY_MARKERS)


def sanitize_media_passthrough(
    extra_args: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the passthrough knobs that are safe to hand a backend.

    The frontend nests a request's unknown top-level fields (an OpenAI
    client's ``extra_body``) under ``extra_args[MEDIA_PASSTHROUGH_KEY]``.
    Those are untyped and fully caller controlled, so a knob that names a
    model or checkpoint path, a fetch location, or a safety or policy
    control is refused here rather than passed to the engine, where it could
    reach a load path or a guardrail switch. Everything else is returned
    unchanged for the caller to apply.

    Raises MediaPassthroughRejected (a ValueError) when a rejected knob is
    present, so request building fails before the engine runs.
    """
    knobs = (extra_args or {}).get(MEDIA_PASSTHROUGH_KEY) or {}
    rejected = sorted(k for k in knobs if _passthrough_key_rejected(k))
    if rejected:
        raise MediaPassthroughRejected(
            "these request fields cannot be set per request (a model or "
            "checkpoint path, a fetch location, or a safety or policy "
            "control is deployment configuration): " + ", ".join(rejected)
        )
    return dict(knobs)


__all__ = [
    "MEDIA_PASSTHROUGH_KEY",
    "MediaPassthroughRejected",
    "sanitize_media_passthrough",
    "NvCreateVideoRequest",
    "NvVideosResponse",
    "VideoData",
]
