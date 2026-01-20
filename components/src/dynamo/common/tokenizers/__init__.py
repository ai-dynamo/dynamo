# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Tokenizers Module

This module provides tokenizer abstractions for use with Dynamo's preprocessor.
It allows using Python-based tokenizers (SGLang, vLLM, or custom) instead of
the default HuggingFace Rust tokenizer.

Main components:
    - DynamoTokenizer: Protocol defining the tokenizer interface
    - BaseTokenizer: Base class with default implementations
    - SGLangTokenizer: Wrapper for SGLang's tokenizer
    - VLLMTokenizer: Wrapper for vLLM's tokenizer
"""

from dynamo.common.tokenizers.protocol import BaseTokenizer, DynamoTokenizer

__all__ = ["DynamoTokenizer", "BaseTokenizer"]


# Lazy imports for optional dependencies
def get_sglang_tokenizer():
    """Get the SGLang tokenizer class (lazy import to avoid dependency issues)."""
    from dynamo.common.tokenizers.sglang import SGLangTokenizer

    return SGLangTokenizer


def get_vllm_tokenizer():
    """Get the vLLM tokenizer class (lazy import to avoid dependency issues)."""
    from dynamo.common.tokenizers.vllm import VLLMTokenizer

    return VLLMTokenizer
