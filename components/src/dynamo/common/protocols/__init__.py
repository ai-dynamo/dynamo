# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared protocol types used across multiple Dynamo backends.

This module provides protocol types for various modalities:
- video_protocol: NvCreateVideoRequest, NvVideosResponse for video generation
"""

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)

MEDIA_PASSTHROUGH_KEY = "media_passthrough"
"""Key under a media request's ``extra_args`` where the frontend nests
unknown top-level request fields (an OpenAI client's ``extra_body``)."""

__all__ = [
    "MEDIA_PASSTHROUGH_KEY",
    "NvCreateVideoRequest",
    "NvVideosResponse",
    "VideoData",
]
