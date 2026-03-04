# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for TensorRT-LLM backend.

This module provides configuration dataclasses:
- DiffusionConfig: Configuration for diffusion model workers
"""

from dynamo.trtllm.configs.diffusion_config import DiffusionConfig

__all__ = ["DiffusionConfig"]
