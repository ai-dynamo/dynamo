# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DYN_MM_VIDEO_NUM_FRAMES = "DYN_MM_VIDEO_NUM_FRAMES"
DEFAULT_FRONTEND_VIDEO_NUM_FRAMES = 32


def _video_num_frames() -> int:
    raw = os.getenv(DYN_MM_VIDEO_NUM_FRAMES, "").strip()
    if not raw:
        return DEFAULT_FRONTEND_VIDEO_NUM_FRAMES

    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %s",
            DYN_MM_VIDEO_NUM_FRAMES,
            raw,
            DEFAULT_FRONTEND_VIDEO_NUM_FRAMES,
        )
        return DEFAULT_FRONTEND_VIDEO_NUM_FRAMES

    if value <= 0:
        logger.warning(
            "Ignoring non-positive %s=%r; using %s",
            DYN_MM_VIDEO_NUM_FRAMES,
            raw,
            DEFAULT_FRONTEND_VIDEO_NUM_FRAMES,
        )
        return DEFAULT_FRONTEND_VIDEO_NUM_FRAMES
    return value


def enable_frontend_video_decoding(media_decoder: Any) -> None:
    """Enable video decoding when the Python binding includes FFmpeg support."""
    enable_video = getattr(media_decoder, "enable_video", None)
    if enable_video is None:
        logger.warning(
            "Frontend video decoding is unavailable because this Dynamo Python "
            "binding was built without the `media-ffmpeg` feature"
        )
        return

    enable_video({"max_frames": _video_num_frames()})
