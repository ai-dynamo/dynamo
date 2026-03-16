# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Worker - Multimodal-aware KV cache routing for vLLM.

This worker sits between the frontend and vLLM workers, computing mm_hash
for images and routing requests to the worker with the best KV cache overlap.
"""

from .handler import MMRouterHandler
from .mm_processor import ProcessedInput, build_block_mm_infos

__all__ = [
    "MMRouterHandler",
    "ProcessedInput",
    "build_block_mm_infos",
]
