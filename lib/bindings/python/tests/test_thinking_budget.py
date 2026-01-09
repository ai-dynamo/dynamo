# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ThinkingBudgetLogitsProcessor.

Important behavior note:
- max_thinking_tokens: Limits tokens within <think>...</think> blocks
- max_tokens: Total generation limit enforced by the inference engine (vLLM/SGLang)
- max_tokens SUPERSEDES max_thinking_tokens - if the engine hits max_tokens during
  thinking, generation stops immediately (potentially mid-thought)
"""

import torch

from dynamo.logits_processing import (
    ThinkingBudgetLogitsProcessor,
    ThinkingTokenConfig,
    get_thinking_token_config,
    register_thinking_token_config,
)

# Test token IDs (using DeepSeek-style tokens)
THINK_START = 128798
THINK_END = 128799
NEWLINE = 201
VOCAB_SIZE = 130000


class TestThinkingBudgetLogitsProcessor:
    """Tests for ThinkingBudgetLogitsProcessor."""

    def test_no_modification_outside_thinking_mode(self):
        """Processor should not modify logits when not in thinking mode."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=100,
        )

        # No <think> token in input
        input_ids = [1, 2, 3, 4, 5]
        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0  # Some token has high logit

        processor(input_ids, logits)

        # Logits should be unchanged
        assert logits[100] == 10.0
        assert logits[THINK_END] == 0.0

    def test_no_modification_after_thinking_closed(self):
        """Processor should not modify logits after </think> is emitted."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=10,
        )

        # <think> ... </think> already in sequence
        input_ids = [1, THINK_START, 2, 3, 4, THINK_END, 5, 6]
        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0

        processor(input_ids, logits)

        # Logits should be unchanged (thinking is closed)
        assert logits[100] == 10.0

    def test_no_modification_before_budget_exceeded(self):
        """Processor should not modify logits before budget is exceeded."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=100,
        )

        # In thinking mode but only 5 tokens generated (well under budget)
        input_ids = [1, THINK_START, 2, 3, 4, 5, 6]
        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0

        processor(input_ids, logits)

        # Logits should be unchanged (under budget)
        assert logits[100] == 10.0

    def test_force_end_at_hard_limit(self):
        """Processor should force </think> at hard limit (budget + overflow)."""
        max_thinking = 10
        overflow_pct = 0.1
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=max_thinking,
            overflow_percentage=overflow_pct,
        )

        # Generate enough tokens to hit hard limit (10 + 1 overflow = 11)
        thinking_tokens = list(range(1, 13))  # 12 tokens after <think>
        input_ids = [1, THINK_START] + thinking_tokens
        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0

        processor(input_ids, logits)

        # Should force </think>
        assert logits[THINK_END] == 0.0
        assert logits[100] == float("-inf")

    def test_force_end_on_newline_in_overflow(self):
        """Processor should force </think> on newline when in overflow mode."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=10,
            overflow_percentage=0.5,  # 50% overflow for this test
        )

        # In thinking mode, past budget, last token is newline
        thinking_tokens = list(range(1, 12))  # 11 tokens (past 10 budget)
        input_ids = [1, THINK_START] + thinking_tokens + [NEWLINE]
        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0

        processor(input_ids, logits)

        # Should force </think> because last token is newline
        assert logits[THINK_END] == 0.0
        assert logits[100] == float("-inf")

    def test_no_force_in_overflow_without_newline(self):
        """Processor should not force </think> in overflow without newline."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=10,
            overflow_percentage=0.5,  # 50% overflow
        )

        # In thinking mode, past budget but not at hard limit, no newline
        thinking_tokens = list(range(1, 12))  # 11 tokens (past 10, under 15)
        input_ids = [1, THINK_START] + thinking_tokens + [999]  # Not newline
        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0

        processor(input_ids, logits)

        # Should NOT force </think> yet (waiting for newline or hard limit)
        assert logits[100] == 10.0

    def test_invalid_token_ids_skipped(self):
        """Processor should skip processing with invalid (zero) token IDs."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=0,  # Invalid
            thinking_end_token_id=0,  # Invalid
            newline_token_id=NEWLINE,
            max_thinking_tokens=10,
        )

        input_ids = [1, 2, 3, 4, 5]
        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0

        processor(input_ids, logits)

        # Logits should be unchanged
        assert logits[100] == 10.0

    def test_stateless_between_calls(self):
        """Processor should be stateless - state derived from input_ids only."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=10,
        )

        # First call: past budget, force </think>
        input_ids_1 = [1, THINK_START] + list(range(1, 15))
        logits_1 = torch.zeros(VOCAB_SIZE)
        processor(input_ids_1, logits_1)
        assert logits_1[THINK_END] == 0.0

        # Second call: new request, not in thinking mode
        input_ids_2 = [1, 2, 3]
        logits_2 = torch.zeros(VOCAB_SIZE)
        logits_2[100] = 10.0
        processor(input_ids_2, logits_2)

        # Should NOT force </think> (different request, no thinking mode)
        assert logits_2[100] == 10.0
        assert logits_2[THINK_END] == 0.0


class TestMaxTokensSupersedes:
    """
    Tests documenting that max_tokens (engine limit) supersedes max_thinking_tokens.

    These tests document expected behavior - the processor itself doesn't enforce
    max_tokens, but the inference engine does. If max_tokens is hit during thinking,
    generation stops immediately.
    """

    def test_processor_does_not_track_max_tokens(self):
        """
        Processor only tracks thinking tokens, not total tokens.

        max_tokens enforcement is the responsibility of the inference engine.
        This test documents that the processor will continue to allow generation
        until its own budget is hit, regardless of total token count.
        """
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=1000,  # High thinking budget
        )

        # Simulate: 500 tokens before <think>, then 50 thinking tokens
        # Total = 552 tokens, but only 50 thinking tokens
        pre_think_tokens = list(range(1, 501))  # 500 tokens
        thinking_tokens = list(range(1, 51))  # 50 thinking tokens
        input_ids = pre_think_tokens + [THINK_START] + thinking_tokens

        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0

        processor(input_ids, logits)

        # Processor should NOT force </think> - only 50 thinking tokens (under 1000)
        # Even though total is 552 tokens
        assert logits[100] == 10.0
        assert logits[THINK_END] == 0.0

    def test_thinking_budget_enforced_independently(self):
        """
        Thinking budget is enforced based on tokens AFTER <think>, not total.
        """
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=10,
        )

        # 100 tokens before <think>, then 12 thinking tokens (exceeds budget+overflow)
        pre_think_tokens = list(range(1, 101))
        thinking_tokens = list(range(1, 13))
        input_ids = pre_think_tokens + [THINK_START] + thinking_tokens

        logits = torch.zeros(VOCAB_SIZE)
        logits[100] = 10.0

        processor(input_ids, logits)

        # Should force </think> because thinking budget exceeded
        assert logits[THINK_END] == 0.0
        assert logits[100] == float("-inf")


class TestThinkingTokenConfig:
    """Tests for thinking token configuration."""

    def test_get_deepseek_config(self):
        """Should return hardcoded config for DeepSeek models."""
        config = get_thinking_token_config("deepseek-ai/DeepSeek-R1")
        assert config.start_token_id == 128798
        assert config.end_token_id == 128799
        assert config.newline_token_id == 201

    def test_get_qwen3_config(self):
        """Should return hardcoded config for Qwen3 models."""
        config = get_thinking_token_config("Qwen/Qwen3-32B")
        assert config.start_token_id == 151667
        assert config.end_token_id == 151668

    def test_get_glm4_config(self):
        """Should return hardcoded config for GLM-4 models."""
        config = get_thinking_token_config("THUDM/glm-4-9b")
        assert config.start_token_id == 151350
        assert config.end_token_id == 151351

    def test_unknown_model_returns_zeros(self):
        """Should return zero IDs for unknown models without tokenizer."""
        config = get_thinking_token_config("unknown-model-xyz")
        assert config.start_token_id == 0
        assert config.end_token_id == 0

    def test_register_custom_config(self):
        """Should allow registering custom configurations."""
        custom_config = ThinkingTokenConfig(
            start_token_id=99999,
            end_token_id=99998,
            newline_token_id=100,
        )
        register_thinking_token_config("my-custom-model", custom_config)

        config = get_thinking_token_config("my-custom-model-v1")
        assert config.start_token_id == 99999
        assert config.end_token_id == 99998

    def test_case_insensitive_matching(self):
        """Config lookup should be case-insensitive."""
        config1 = get_thinking_token_config("DEEPSEEK-R1")
        config2 = get_thinking_token_config("DeepSeek-R1")
        assert config1.start_token_id == config2.start_token_id


class TestGetThinkingState:
    """Tests for _get_thinking_state helper method."""

    def test_multiple_thinking_blocks(self):
        """Should handle multiple thinking blocks (use last one)."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=100,
        )

        # First block closed, second block open
        input_ids = [
            1,
            THINK_START,
            2,
            3,
            THINK_END,  # First block closed
            4,
            THINK_START,
            5,
            6,
            7,  # Second block open, 3 tokens
        ]

        in_thinking, count = processor._get_thinking_state(input_ids)
        assert in_thinking is True
        assert count == 3

    def test_empty_input_ids(self):
        """Should handle empty input_ids."""
        processor = ThinkingBudgetLogitsProcessor(
            thinking_start_token_id=THINK_START,
            thinking_end_token_id=THINK_END,
            newline_token_id=NEWLINE,
            max_thinking_tokens=100,
        )

        in_thinking, count = processor._get_thinking_state([])
        assert in_thinking is False
        assert count == 0
