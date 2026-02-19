# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared completion usage builder for Dynamo backends.

This module provides a unified function for building completion usage statistics
dictionaries. Both vLLM and TRT-LLM backends produce the same schema:
{prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details}.
"""

from typing import Any, Optional


def build_completion_usage(
    prompt_tokens: Optional[int],
    completion_tokens: int,
    cached_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Build completion usage statistics dictionary.

    Args:
        prompt_tokens: Number of tokens in the prompt, or None if unknown.
        completion_tokens: Number of tokens generated.
        cached_tokens: Number of prompt tokens served from cache, or None.

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens, and
        prompt_tokens_details (if cached_tokens is provided and > 0).
    """
    prompt_tokens_details = None
    if cached_tokens is not None and cached_tokens > 0:
        prompt_tokens_details = {"cached_tokens": cached_tokens}

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens if prompt_tokens is not None else None
        ),
        "prompt_tokens_details": prompt_tokens_details,
    }
