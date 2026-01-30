# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM Video Diffusion support for Dynamo.

This module provides video generation capabilities using TensorRT-LLM's visual_gen
module, supporting models like Wan for text-to-video generation.

Usage:
    python -m dynamo.trtllm_diffusion --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...
"""

# Lazy imports to avoid loading heavy dependencies at module import time
# Use explicit imports where needed:
#   from dynamo.trtllm_diffusion.engine import WanDiffusionEngine
#   from dynamo.trtllm_diffusion.protocol import NvCreateVideoRequest, NvVideosResponse

__all__ = [
    "WanDiffusionEngine",
    "NvCreateVideoRequest",
    "NvVideosResponse",
    "VideoData",
]


def __getattr__(name):
    """Lazy import for heavy dependencies."""
    if name == "WanDiffusionEngine":
        from dynamo.trtllm_diffusion.engine import WanDiffusionEngine
        return WanDiffusionEngine
    elif name in ("NvCreateVideoRequest", "NvVideosResponse", "VideoData"):
        from dynamo.trtllm_diffusion import protocol
        return getattr(protocol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
