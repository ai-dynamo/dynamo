# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Re-export video protocol types from Rust PyO3 bindings.
# These were previously Pydantic models; now backed by Rust types in
# lib/llm/src/protocols/openai/videos.rs via lib/bindings/python.

from dynamo._core import (  # noqa: F401
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
    VideoNvExt,
)

__all__ = [
    "VideoNvExt",
    "NvCreateVideoRequest",
    "VideoData",
    "NvVideosResponse",
]
