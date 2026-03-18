# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Logits Processing - Backend-agnostic logits processors.

This module provides the BaseLogitsProcessor protocol that can be used
across different backend adapters (TRT-LLM, vLLM, SGLang).
"""

from .base import BaseLogitsProcessor

__all__ = ["BaseLogitsProcessor"]
