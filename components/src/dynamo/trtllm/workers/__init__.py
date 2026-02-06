# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker initialization modules for TensorRT-LLM backend.

This package contains worker initialization functions for different modalities:
- video_diffusion_worker: Video generation using diffusion models
"""

from dynamo.trtllm.workers.video_diffusion_worker import init_video_diffusion_worker

__all__ = ["init_video_diffusion_worker"]
