# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix hash utilities for converting between prompt text and trace hash IDs."""

from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
from typing import Protocol

DEFAULT_BLOCK_SIZE = 512
_ROOT_ID = -1

_Block = str | tuple[int, ...]


class Tokenizer(Protocol):
    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
    ) -> Mapping[str, Sequence[Sequence[int]]]:
        ...


class PromptGenerator(Protocol):
    def generate(self, *, mean: int, hash_ids: Sequence[int] | None = None) -> str:
        ...


class RollingHasher:
    """Assign consecutive IDs to nodes using fast chained hashes."""

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        _validate_block_size(block_size)
        self.block_size = block_size
        self._hash_to_id: dict[int, int] = {}

    def __call__(self, blocks: Iterable[Sequence[int]]) -> list[int]:
        return self.hash_token_blocks(blocks)

    def hash_blocks(self, blocks: Iterable[str]) -> list[int]:
        return self._intern(blocks)

    def hash_token_blocks(self, blocks: Iterable[Sequence[int]]) -> list[int]:
        return self._intern(
            block if isinstance(block, tuple) else tuple(block) for block in blocks
        )

    def _intern(self, blocks: Iterable[_Block]) -> list[int]:
        parent_id = _ROOT_ID
        hash_ids: list[int] = []

        for block in blocks:
            node_hash = hash((parent_id, block))
            hash_id = self._hash_to_id.setdefault(node_hash, len(self._hash_to_id))
            hash_ids.append(hash_id)
            parent_id = hash_id

        return hash_ids

    def reset(self) -> None:
        self._hash_to_id.clear()

    def get_stats(self) -> dict[str, int]:
        total_hashes = len(self._hash_to_id)
        return {
            "total_hashes": total_hashes,
            "max_id": total_hashes - 1,
        }


def texts_to_hashes(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[list[int]]:
    hash_ids, _ = texts_to_hashes_and_lengths(tokenizer, texts, block_size)
    return hash_ids


def texts_to_hashes_and_lengths(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[list[list[int]], list[int]]:
    _validate_block_size(block_size)
    if not texts:
        return [], []

    batch = tokenizer(
        list(texts),
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_sequences = batch["input_ids"]
    if len(token_sequences) != len(texts):
        raise ValueError(
            "Tokenizer returned a different number of sequences than requested"
        )

    hasher = RollingHasher(block_size=block_size)
    hash_ids = [
        hasher.hash_token_blocks(_batched(tokens, block_size))
        for tokens in token_sequences
    ]
    input_lengths = [len(tokens) for tokens in token_sequences]
    return hash_ids, input_lengths


def hashes_to_texts(
    prompt_generator: PromptGenerator,
    hash_ids_list: Sequence[Sequence[int]],
    input_lengths: Sequence[int],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> list[str]:
    _validate_block_size(block_size)
    texts: list[str] = []

    for hash_ids, input_length in zip(hash_ids_list, input_lengths, strict=True):
        if hash_ids and len(hash_ids) * block_size < input_length:
            raise ValueError(
                "Hash blocks do not cover the requested input length: "
                f"{len(hash_ids) * block_size} < {input_length}"
            )

        if hash_ids:
            text = prompt_generator.generate(mean=input_length, hash_ids=list(hash_ids))
        else:
            text = prompt_generator.generate(mean=input_length)
        texts.append(text)

    return texts


def _batched(tokens: Sequence[int], block_size: int) -> Iterable[tuple[int, ...]]:
    iterator = iter(tokens)
    while block := tuple(islice(iterator, block_size)):
        yield block


def _validate_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")


__all__ = [
    "RollingHasher",
    "hashes_to_texts",
    "texts_to_hashes",
    "texts_to_hashes_and_lengths",
]
