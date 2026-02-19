# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video diffusion request handlers for TensorRT-LLM backend.

This module provides handlers for video generation using diffusion models.
"""

from dynamo.trtllm.request_handlers.video_diffusion.video_handler import (
    VideoGenerationHandler,
)

__all__ = ["VideoGenerationHandler"]
