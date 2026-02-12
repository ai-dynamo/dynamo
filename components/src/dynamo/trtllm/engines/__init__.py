# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine modules for TensorRT-LLM backend.

This module provides engine wrappers for various generative models:
- DiffusionEngine: Generic wrapper for visual_gen diffusion pipelines
"""

from dynamo.trtllm.engines.diffusion_engine import DiffusionEngine

__all__ = ["DiffusionEngine"]
