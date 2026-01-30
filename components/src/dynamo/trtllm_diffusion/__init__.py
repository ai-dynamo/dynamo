# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM Video Diffusion support for Dynamo.

This module provides video generation capabilities using TensorRT-LLM's visual_gen
module, supporting models like Wan for text-to-video generation.
"""

from dynamo.trtllm_diffusion.engine import WanDiffusionEngine
from dynamo.trtllm_diffusion.protocol import NvCreateVideoRequest, NvVideosResponse, VideoData

__all__ = [
    "WanDiffusionEngine",
    "NvCreateVideoRequest",
    "NvVideosResponse",
    "VideoData",
]
