# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for TensorRT-LLM backend.

This module provides protocol types for various modalities:
- video_protocol: NvCreateVideoRequest, NvVideosResponse for video generation
- image_protocol: (future) Protocol types for image generation
"""

from dynamo.trtllm.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)

__all__ = [
    "NvCreateVideoRequest",
    "NvVideosResponse",
    "VideoData",
]
