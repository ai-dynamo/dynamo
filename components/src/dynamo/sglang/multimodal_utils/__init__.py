# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal utilities for SGLang.

This package contains utilities for processing multimodal (image/video) inputs
for SGLang models.
"""

from dynamo.sglang.multimodal_utils.multimodal_chat_processor import (
    extract_image_video_urls,
    multimodal_request_to_sglang,
    process_sglang_stream_response,
)
from dynamo.sglang.multimodal_utils.multimodal_encode_utils import (
    encode_image_embeddings,
)
from dynamo.sglang.multimodal_utils.multimodal_image_loader import ImageLoader

__all__ = [
    "extract_image_video_urls",
    "multimodal_request_to_sglang",
    "process_sglang_stream_response",
    "encode_image_embeddings",
    "ImageLoader",
]
