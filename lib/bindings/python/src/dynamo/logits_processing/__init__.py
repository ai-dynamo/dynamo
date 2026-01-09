# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Logits Processing - Backend-agnostic logits processors.

This module provides the BaseLogitsProcessor protocol that can be used
across different backend adapters (TRT-LLM, vLLM, SGLang).
"""

from .base import BaseLogitsProcessor
from .thinking_budget import ThinkingBudgetLogitsProcessor
from .thinking_tokens import (
    ThinkingTokenConfig,
    get_thinking_token_config,
    register_thinking_token_config,
)

__all__ = [
    "BaseLogitsProcessor",
    "ThinkingBudgetLogitsProcessor",
    "ThinkingTokenConfig",
    "get_thinking_token_config",
    "register_thinking_token_config",
]
