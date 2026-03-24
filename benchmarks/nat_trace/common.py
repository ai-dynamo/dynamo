# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and utilities for trace converters."""

from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.synthesis.rolling_hasher import RollingHasher

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_BLOCK_SIZE = 64


def texts_to_hashes_and_lengths(
    tokenizer: Tokenizer,
    texts: list[str],
    block_size: int,
) -> tuple[list[list[int]], list[int]]:
    """
    Convert texts to hash IDs and token lengths.

    Returns:
        Tuple of (hash_ids_list, token_lengths) where:
        - hash_ids_list: List of hash ID sequences, one per input text
        - token_lengths: List of token counts, one per input text
    """
    hasher = RollingHasher(block_size=block_size)
    hash_results: list[list[int]] = []
    length_results: list[int] = []

    for text in texts:
        tokens = tokenizer.encode(text)
        length_results.append(len(tokens))

        blocks: list[list[int]] = [
            tokens[i : i + block_size] for i in range(0, len(tokens), block_size)
        ]
        if blocks:
            hashes = hasher.hash_token_blocks(blocks)
            hash_results.append(hashes)
        else:
            hash_results.append([])

    return hash_results, length_results
