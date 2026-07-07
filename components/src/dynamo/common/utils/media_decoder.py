# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DYN_MM_VIDEO_DECODER_BACKEND = "DYN_MM_VIDEO_DECODER_BACKEND"
DYN_MM_VIDEO_NUM_FRAMES = "DYN_MM_VIDEO_NUM_FRAMES"
VALID_VIDEO_DECODER_BACKENDS = frozenset({"video_rs", "ffmpeg", "opencv"})

DEFAULT_FRONTEND_IMAGE_DECODER_MAX_ALLOC = 128 * 1024 * 1024
DEFAULT_FRONTEND_VIDEO_DECODER_MAX_ALLOC = 512 * 1024 * 1024
DEFAULT_FRONTEND_VIDEO_NUM_FRAMES = 32


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %s", name, raw, default)
        return default
    return value


def build_frontend_image_decoder_options(
    *,
    max_alloc: int = DEFAULT_FRONTEND_IMAGE_DECODER_MAX_ALLOC,
) -> dict[str, Any]:
    return {"limits": {"max_alloc": max_alloc}}


def build_frontend_video_decoder_options(
    *,
    max_alloc: int = DEFAULT_FRONTEND_VIDEO_DECODER_MAX_ALLOC,
    num_frames: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "limits": {"max_alloc": max_alloc},
        "num_frames": (
            num_frames
            if num_frames is not None
            else _env_int(DYN_MM_VIDEO_NUM_FRAMES, DEFAULT_FRONTEND_VIDEO_NUM_FRAMES)
        ),
    }

    backend = os.getenv(DYN_MM_VIDEO_DECODER_BACKEND, "").strip()
    if backend:
        if backend not in VALID_VIDEO_DECODER_BACKENDS:
            valid_values = ", ".join(sorted(VALID_VIDEO_DECODER_BACKENDS))
            raise ValueError(
                f"{DYN_MM_VIDEO_DECODER_BACKEND} must be one of: {valid_values}"
            )
        options["backend"] = backend

    return options


def enable_frontend_video_decoding(media_decoder: Any) -> None:
    enable_video = getattr(media_decoder, "enable_video", None)
    if enable_video is None:
        if os.getenv(DYN_MM_VIDEO_DECODER_BACKEND, "").strip():
            logger.warning(
                "%s is set, but this Dynamo Python binding was not built with "
                "frontend video decoding support; video decoder backend selection is ignored.",
                DYN_MM_VIDEO_DECODER_BACKEND,
            )
        return

    enable_video(build_frontend_video_decoder_options())
