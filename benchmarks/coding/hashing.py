from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator, Sequence

from transformers import AutoTokenizer

from benchmarks.coding.common import stable_hash


class TokenizerWrapper:
    def __init__(self, tokenizer_name: str) -> None:
        self._tokenizer_name = tokenizer_name
        self._thread_local = threading.local()
        self._tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        tokenizer.model_max_length = sys.maxsize
        return tokenizer

    def _thread_tokenizer(self) -> AutoTokenizer:
        tokenizer = getattr(self._thread_local, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self._load_tokenizer()
            self._thread_local.tokenizer = tokenizer
        return tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _encode_with_thread_tokenizer(self, text: str) -> list[int]:
        tokenizer = self._thread_tokenizer()
        return list(tokenizer.encode(text, add_special_tokens=False))

    def iter_encode_many(
        self,
        texts: Iterable[str],
        max_workers: int | None = None,
    ) -> Iterator[list[int]]:
        worker_count = min(max_workers or self._default_workers(), 32)
        if worker_count <= 1:
            for text in texts:
                yield self.encode(text)
            return

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            yield from executor.map(self._encode_with_thread_tokenizer, texts)

    @staticmethod
    def _default_workers() -> int:
        return max(1, min(8, os.cpu_count() or 1))


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
