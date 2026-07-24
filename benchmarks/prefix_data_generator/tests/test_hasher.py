# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from aiperf.common.tokenizer import Tokenizer
from prefix_data_generator.hasher import texts_to_hashes


class _WordTokenizer:
    def __init__(self) -> None:
        self._token_ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        return [
            self._token_ids.setdefault(word, len(self._token_ids))
            for word in text.split()
        ]


def test_texts_to_hashes_preserves_shared_prefixes() -> None:
    tokenizer = cast(Tokenizer, _WordTokenizer())

    hash_ids = texts_to_hashes(
        tokenizer,
        ["a b c d", "a b e f", "x y c d"],
        block_size=2,
    )

    assert hash_ids == [[0, 1], [0, 2], [3, 4]]
