# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Thinking budget logits processor for controlling reasoning token generation.

This module provides a logits processor that limits the number of tokens
generated within thinking/reasoning blocks (e.g., <think>...</think>) and
forces early termination when the budget is exceeded.

Note: max_tokens (total generation limit) is enforced by the inference engine
(vLLM/SGLang) and will supersede max_thinking_tokens if reached first.

Similar to SGLang's ThinkingBudgetLogitProcessor and NVIDIA NIM's
BudgetControlLogitsProcessor.
"""

from typing import Sequence

import torch

from .base import BaseLogitsProcessor


class ThinkingBudgetLogitsProcessor(BaseLogitsProcessor):
    """
    Logits processor that limits thinking tokens and forces </think> emission.

    When the number of tokens generated within a thinking block exceeds the
    specified budget (with graceful overflow), this processor modifies the
    logits to force emission of the thinking end token (e.g., </think>).

    Note: The inference engine's max_tokens limit is enforced separately and
    will supersede max_thinking_tokens. If max_tokens is reached during thinking,
    generation stops immediately (potentially mid-thought). Set max_tokens
    appropriately to allow for both thinking and the final answer.

    The graceful overflow mechanism allows the model to complete the current
    sentence (until a newline) before forcing the end token, up to a maximum
    overflow percentage.

    Args:
        thinking_start_token_id: Token ID that marks the start of thinking
            (e.g., <think> token ID)
        thinking_end_token_id: Token ID that marks the end of thinking
            (e.g., </think> token ID)
        newline_token_id: Token ID for newline character (for graceful overflow)
        max_thinking_tokens: Maximum number of thinking tokens before forcing end
        overflow_percentage: Percentage of extra tokens allowed for graceful
            overflow (default: 0.1 = 10%)

    Example:
        >>> processor = ThinkingBudgetLogitsProcessor(
        ...     thinking_start_token_id=128798,  # <think>
        ...     thinking_end_token_id=128799,    # </think>
        ...     newline_token_id=201,
        ...     max_thinking_tokens=100,
        ... )
        >>> # Processor will force </think> after ~100-110 thinking tokens
    """

    def __init__(
        self,
        thinking_start_token_id: int,
        thinking_end_token_id: int,
        newline_token_id: int,
        max_thinking_tokens: int,
        overflow_percentage: float = 0.1,
    ):
        self.thinking_start_token_id = thinking_start_token_id
        self.thinking_end_token_id = thinking_end_token_id
        self.newline_token_id = newline_token_id
        self.max_thinking_tokens = max_thinking_tokens
        self.overflow_budget = int(max_thinking_tokens * overflow_percentage)

    def __call__(
        self,
        input_ids: Sequence[int],
        logits: torch.Tensor,
    ) -> None:
        """
        Modify logits in-place to enforce thinking budget.

        Args:
            input_ids: The input token IDs generated so far.
            logits: The raw logits for the next token. Shape: (vocab_size,)
                   Modified in-place.
        """
        # Skip if no valid token IDs configured
        if self.thinking_start_token_id == 0 or self.thinking_end_token_id == 0:
            return

        # Determine thinking state from input_ids
        in_thinking_mode, thinking_tokens_generated = self._get_thinking_state(
            input_ids
        )

        if not in_thinking_mode:
            return

        budget_with_overflow = self.max_thinking_tokens + self.overflow_budget

        # Check if thinking budget exceeded - enter overflow mode
        in_overflow_mode = thinking_tokens_generated >= self.max_thinking_tokens

        if not in_overflow_mode:
            return

        # In overflow mode - check for graceful exit conditions
        last_token = input_ids[-1] if input_ids else None
        is_newline = last_token == self.newline_token_id
        at_hard_limit = thinking_tokens_generated >= budget_with_overflow

        if is_newline or at_hard_limit:
            # Force </think> token by setting all other logits to -inf
            logits.fill_(-float("inf"))
            logits[self.thinking_end_token_id] = 0.0

    def _get_thinking_state(self, input_ids: Sequence[int]) -> tuple[bool, int]:
        """
        Determine thinking state from input_ids.

        Returns:
            Tuple of (in_thinking_mode, thinking_tokens_generated)
        """
        # Find last occurrence of thinking start token (search from end)
        start_idx = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self.thinking_start_token_id:
                start_idx = i
                break

        if start_idx == -1:
            # No thinking start token found
            return False, 0

        # Check if thinking has already ended (</think> after <think>)
        for i in range(start_idx + 1, len(input_ids)):
            if input_ids[i] == self.thinking_end_token_id:
                # Thinking block is already closed
                return False, 0

        # We're in thinking mode
        thinking_tokens_generated = len(input_ids) - start_idx - 1
        return True, thinking_tokens_generated

    def reset(self) -> None:
        """Reset internal state. Called between requests if needed."""
        # This processor is stateless - state is computed from input_ids
        pass
