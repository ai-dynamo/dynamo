# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, List, Union, cast

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

lorem_text = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
    "eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt "
    "in culpa qui officia deserunt mollit anim id est laborum."
)
words = np.array(list(set(re.findall(r"\b[a-zA-Z]+\b", lorem_text))))


def texts_to_hashes(
    tokenizer: Union[str, PreTrainedTokenizerBase],
    texts: List[str],
    block_size: int = 512,
) -> List[List[int]]:
    """
    Tokenizes a list of strings (without special tokens), splits tokens into blocks,
    computes rolling hashes, and returns a list of lists of integer-mapped rolling hashes
    for each input string.

    Args:
        tokenizer: Tokenizer object with a .encode method or string name to load from HuggingFace.
        texts (List[str]): List of input strings.
        block_size (int): Size of each token block for hashing.

    Returns:
        List[List[int]]: List of lists of integer-mapped rolling hashes for each block of each input string.
    """
    # Load tokenizer if string is provided
    if isinstance(tokenizer, str):
        tokenizer = cast(
            PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizer)
        )

    # Batch tokenize for efficiency
    batch_encoding = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    # batch_encoding["input_ids"] is a List[List[int]]
    all_tokens: List[List[int]] = batch_encoding["input_ids"]

    results: List[List[int]] = []
    hash_to_int: Dict[int, int] = {}
    next_int = 0

    for tokens in all_tokens:
        blocks: List[List[int]] = [
            tokens[i : i + block_size] for i in range(0, len(tokens), block_size)
        ]

        parent_hash = 0
        hashes: List[int] = []

        for block in blocks:
            combined = (parent_hash, hash(tuple(block)))
            global_hash = hash(combined)

            # Map global_hash to a unique integer
            if global_hash not in hash_to_int:
                hash_to_int[global_hash] = next_int
                next_int += 1

            hashes.append(hash_to_int[global_hash])
            parent_hash = global_hash

        results.append(hashes)

    return results


def hashes_to_texts(
    tokenizer: Union[str, PreTrainedTokenizerBase],
    hash_ids_list: List[List[int]],
    input_lengths: List[int],
    block_size: int = 512,
) -> List[str]:
    """
    Converts a list of hash ID sequences back to text strings using a global token mapping.

    Args:
        tokenizer: Tokenizer object with a .decode method or string name to load from HuggingFace.
        hash_ids_list (List[List[int]]): List of hash ID sequences for each input.
        input_lengths (List[int]): Target input lengths for each sequence.
        block_size (int): Size of each token block for reconstruction.

    Returns:
        List[str]: List of reconstructed text strings.
    """
    # Load tokenizer if string is provided
    if isinstance(tokenizer, str):
        tokenizer = cast(
            PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizer)
        )

    results: List[str] = []
    _hash_id_to_tokens: Dict[int, np.ndarray] = {}

    for hash_ids, input_len in zip(hash_ids_list, input_lengths):
        # Verify constraint: len(hash_ids) * block_size <= input_len
        if len(hash_ids) * block_size < input_len:
            raise ValueError(
                f"Constraint violation: len(hash_ids) * block_size ({len(hash_ids) * block_size}) > input_len ({input_len})"
            )

        token_arrays: List[np.ndarray] = []

        for i, hash_id in enumerate(hash_ids):
            # Determine the block size for this hash_id
            remaining_tokens = input_len - sum(len(arr) for arr in token_arrays)
            current_block_size = min(block_size, remaining_tokens)

            if current_block_size <= 0:
                break

            # Check if hash_id already exists in global dict
            if hash_id in _hash_id_to_tokens:
                # Use existing array, but assert it matches current_block_size
                existing_array = _hash_id_to_tokens[hash_id]
                assert (
                    len(existing_array) == current_block_size
                ), f"Existing array length {len(existing_array)} does not match current block size {current_block_size}"
                token_array = existing_array
            else:
                # Generate new random array by sampling words, tokenizing, and taking first tokens
                sampled_words = np.random.choice(words, size=current_block_size)
                sampled_text = " ".join(sampled_words)
                tokens = tokenizer.encode(sampled_text, add_special_tokens=False)
                token_array = np.array(tokens[:current_block_size], dtype=np.int32)
                if getattr(tokenizer, "bos_token_id", None) is not None:
                    token_array[0] = tokenizer.bos_token_id
                _hash_id_to_tokens[hash_id] = token_array

            token_arrays.append(token_array)

        all_tokens = np.concatenate(token_arrays)

        # Decode to text
        text = tokenizer.decode(all_tokens, skip_special_tokens=False)
        results.append(text)

    return results
