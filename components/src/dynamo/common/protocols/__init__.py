# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared protocol types used across multiple Dynamo backends.

Re-exports Rust-backed protocol types from PyO3 bindings:
- image_protocol: ImageNvExt, NvCreateImageRequest, ImageData, NvImagesResponse
- video_protocol: VideoNvExt, NvCreateVideoRequest, VideoData, NvVideosResponse
"""

from dynamo.common.protocols.image_protocol import (
    ImageData,
    ImageNvExt,
    NvCreateImageRequest,
    NvImagesResponse,
)
from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
    VideoNvExt,
)

__all__ = [
    "ImageNvExt",
    "NvCreateImageRequest",
    "ImageData",
    "NvImagesResponse",
    "VideoNvExt",
    "NvCreateVideoRequest",
    "VideoData",
    "NvVideosResponse",
]
