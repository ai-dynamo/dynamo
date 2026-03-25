# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared protocol types used across multiple Dynamo backends.

This module provides protocol types for various modalities:
- video_protocol: NvCreateVideoRequest, NvVideosResponse for video generation
- audio_protocol: NvCreateAudioSpeechRequest, NvAudiosResponse for audio generation
"""

from dynamo.common.protocols.audio_protocol import (
    NvAudiosResponse,
    NvCreateAudioSpeechRequest,
)
from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)

__all__ = [
    "NvAudiosResponse",
    "NvCreateAudioSpeechRequest",
    "NvCreateVideoRequest",
    "NvVideosResponse",
    "VideoData",
]
