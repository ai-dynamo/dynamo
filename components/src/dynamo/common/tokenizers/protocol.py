# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizer Protocol for Dynamo's Python Tokenizer Support

This module defines the protocol that Python tokenizers must implement
to be used with Dynamo's Rust preprocessor via PyO3.
"""

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class DynamoTokenizer(Protocol):
    """Protocol defining the interface for Dynamo-compatible tokenizers.

    Any tokenizer that implements this protocol can be used with Dynamo's
    Rust preprocessor through the PythonTokenizer wrapper.

    Methods:
        encode: Tokenize a single text string to token IDs
        encode_batch: Tokenize multiple text strings to token IDs
        decode: Convert token IDs back to text
    """

    def encode(self, text: str) -> list[int]:
        """Encode a single text string to a list of token IDs.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of integer token IDs.
        """
        ...

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode multiple text strings to lists of token IDs.

        Args:
            texts: A sequence of input texts to tokenize.

        Returns:
            A list of lists of integer token IDs.
        """
        ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to text.

        Args:
            token_ids: A sequence of token IDs to decode.
            skip_special_tokens: If True, special tokens are excluded from output.

        Returns:
            The decoded text string.
        """
        ...


class BaseTokenizer:
    """Base class for Dynamo tokenizers with default implementations.

    Subclasses should override `encode` and `decode` at minimum.
    The `encode_batch` method has a default implementation that calls
    `encode` in a loop, but subclasses may override for better performance.
    """

    def encode(self, text: str) -> list[int]:
        """Encode a single text string to a list of token IDs.

        Subclasses must implement this method.

        Args:
            text: The input text to tokenize.

        Returns:
            A list of integer token IDs.
        """
        raise NotImplementedError("Subclasses must implement encode()")

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode multiple text strings to lists of token IDs.

        Default implementation calls encode() for each text.
        Subclasses may override for better batch performance.

        Args:
            texts: A sequence of input texts to tokenize.

        Returns:
            A list of lists of integer token IDs.
        """
        return [self.encode(text) for text in texts]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to text.

        Subclasses must implement this method.

        Args:
            token_ids: A sequence of token IDs to decode.
            skip_special_tokens: If True, special tokens are excluded from output.

        Returns:
            The decoded text string.
        """
        raise NotImplementedError("Subclasses must implement decode()")
