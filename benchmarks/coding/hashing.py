from __future__ import annotations

import sys
from typing import Sequence

from transformers import AutoTokenizer

from benchmarks.coding.common import stable_hash


class TokenizerWrapper:
    def __init__(self, tokenizer_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._tokenizer.model_max_length = sys.maxsize

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))


class RollingHasher:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self._hash_to_id: dict[int, int] = {}
        self._id_counter = 0

    def hash_token_blocks(self, blocks: Sequence[Sequence[int]]) -> list[int]:
        hash_ids: list[int] = []
        parent_hash = 0

        for block in blocks:
            block_tuple = tuple(block)
            combined_hash = stable_hash((parent_hash, stable_hash(block_tuple)))
            if combined_hash not in self._hash_to_id:
                self._hash_to_id[combined_hash] = self._id_counter
                self._id_counter += 1
            hash_ids.append(self._hash_to_id[combined_hash])
            parent_hash = combined_hash

        return hash_ids
