# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Thinking token configuration for different model architectures.

This module provides model-specific token ID configurations for thinking/reasoning
tokens (e.g., <think>, </think>). It follows the same pattern as the reasoning
parser with hardcoded defaults for known models and dynamic tokenizer fallback.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ThinkingTokenConfig:
    """Configuration for thinking-related token IDs."""

    start_token_id: int
    end_token_id: int
    newline_token_id: int


# Hardcoded defaults per model family (matches reasoning parser pattern)
# Token IDs sourced from SGLang's ThinkingBudgetLogitProcessor implementations
THINKING_TOKEN_CONFIGS = {
    # DeepSeek models (R1, V3, etc.)
    "deepseek": ThinkingTokenConfig(
        start_token_id=128798,  # <think>
        end_token_id=128799,  # </think>
        newline_token_id=201,
    ),
    # Qwen3 models
    "qwen3": ThinkingTokenConfig(
        start_token_id=151667,
        end_token_id=151668,
        newline_token_id=198,
    ),
    # GLM-4.5/4.6 models
    "glm4": ThinkingTokenConfig(
        start_token_id=151350,
        end_token_id=151351,
        newline_token_id=198,
    ),
    "glm-4": ThinkingTokenConfig(
        start_token_id=151350,
        end_token_id=151351,
        newline_token_id=198,
    ),
}


def get_thinking_token_config(
    model_name: str,
    tokenizer: Optional["PreTrainedTokenizer"] = None,
) -> ThinkingTokenConfig:
    """
    Get thinking token configuration for a model.

    Tries hardcoded configurations first, then falls back to dynamic
    tokenizer lookup if available.

    Args:
        model_name: Name or path of the model (e.g., "deepseek-ai/DeepSeek-R1")
        tokenizer: Optional tokenizer for dynamic token ID lookup

    Returns:
        ThinkingTokenConfig with appropriate token IDs

    Example:
        >>> config = get_thinking_token_config("deepseek-ai/DeepSeek-R1")
        >>> config.start_token_id
        128798
        >>> config.end_token_id
        128799
    """
    # Try hardcoded configs first (case-insensitive matching)
    model_lower = model_name.lower()
    for key, config in THINKING_TOKEN_CONFIGS.items():
        if key in model_lower:
            logger.debug(f"Using hardcoded thinking token config for '{key}'")
            return config

    # Dynamic fallback using tokenizer
    if tokenizer is not None:
        config = _get_config_from_tokenizer(tokenizer)
        if config is not None:
            logger.debug("Using dynamic thinking token config from tokenizer")
            return config

    # Final fallback - return zeros to indicate no valid config
    logger.warning(
        f"No thinking token config found for model '{model_name}'. "
        "Thinking budget control will be disabled."
    )
    return ThinkingTokenConfig(
        start_token_id=0,
        end_token_id=0,
        newline_token_id=10,  # Common newline token as fallback
    )


def _get_config_from_tokenizer(
    tokenizer: "PreTrainedTokenizer",
) -> Optional[ThinkingTokenConfig]:
    """
    Attempt to get thinking token config by encoding strings with tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        ThinkingTokenConfig if successful, None otherwise
    """
    try:
        # Try to encode the thinking tokens
        start_ids = tokenizer.encode("<think>", add_special_tokens=False)
        end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)

        # Validate we got single tokens
        if not start_ids or not end_ids:
            logger.debug(
                "Tokenizer did not produce valid token IDs for thinking tokens"
            )
            return None

        return ThinkingTokenConfig(
            start_token_id=start_ids[0],
            end_token_id=end_ids[0],
            newline_token_id=newline_ids[0] if newline_ids else 10,
        )
    except Exception as e:
        logger.debug(f"Failed to get thinking token config from tokenizer: {e}")
        return None


def register_thinking_token_config(
    model_key: str,
    config: ThinkingTokenConfig,
) -> None:
    """
    Register a custom thinking token configuration for a model.

    Useful for adding support for new models without code changes.

    Args:
        model_key: Key to match in model name (lowercase, partial match)
        config: ThinkingTokenConfig with appropriate token IDs

    Example:
        >>> register_thinking_token_config(
        ...     "my-custom-model",
        ...     ThinkingTokenConfig(
        ...         start_token_id=12345,
        ...         end_token_id=12346,
        ...         newline_token_id=10,
        ...     )
        ... )
    """
    THINKING_TOKEN_CONFIGS[model_key.lower()] = config
    logger.info(f"Registered thinking token config for '{model_key}'")
