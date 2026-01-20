# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Tokenizer Wrapper for Dynamo

This module provides a wrapper around vLLM's tokenizer that implements
the DynamoTokenizer protocol for use with Dynamo's Rust preprocessor.
"""

from typing import Sequence

from dynamo.common.tokenizers.protocol import BaseTokenizer


class VLLMTokenizer(BaseTokenizer):
    """Wrapper around vLLM's tokenizer implementing the DynamoTokenizer protocol.

    This tokenizer uses vLLM's `get_tokenizer` function to load the appropriate
    tokenizer for a given model. It supports models that require specialized
    tokenizers.

    Example:
        >>> tokenizer = VLLMTokenizer("Qwen/Qwen3-0.6B")
        >>> token_ids = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(token_ids)
    """

    def __init__(self, model_path: str):
        """Initialize the vLLM tokenizer.

        Args:
            model_path: Path or HuggingFace model name to load the tokenizer for.
        """
        try:
            from vllm.transformers_utils.tokenizer import get_tokenizer
        except ImportError as e:
            raise ImportError(
                "vLLM is required for VLLMTokenizer. "
                "Please install it with: pip install vllm"
            ) from e

        self._tokenizer = get_tokenizer(model_path, trust_remote_code=True)
        self._model_path = model_path

    def encode(self, text: str) -> list[int]:
        """Encode a single text string to a list of token IDs.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of integer token IDs.

        Note:
            This method encodes without adding special tokens by default,
            as the chat template typically handles BOS/EOS tokens.
        """
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode multiple text strings to lists of token IDs.

        Args:
            texts: A sequence of input texts to tokenize.

        Returns:
            A list of lists of integer token IDs.
        """
        # Use the batch method if available for better performance
        if hasattr(self._tokenizer, "__call__"):
            # HuggingFace tokenizers support batch encoding via __call__
            batch_output = self._tokenizer(
                list(texts), add_special_tokens=False, return_attention_mask=False
            )
            return [list(ids) for ids in batch_output["input_ids"]]

        # Fall back to sequential encoding
        return [self.encode(text) for text in texts]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to text.

        Args:
            token_ids: A sequence of token IDs to decode.
            skip_special_tokens: If True, special tokens are excluded from output.

        Returns:
            The decoded text string.
        """
        return self._tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )
