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

from typing import Dict, List

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def texts_to_hashes(
    tokenizer: PreTrainedTokenizerBase, texts: List[str], block_size: int = 512
) -> List[List[int]]:
    """
    Tokenizes a list of strings (without special tokens), splits tokens into blocks,
    computes rolling hashes, and returns a list of lists of integer-mapped rolling hashes
    for each input string.

    Args:
        tokenizer: Tokenizer object with a .encode method.
        texts (List[str]): List of input strings.
        block_size (int): Size of each token block for hashing.

    Returns:
        List[List[int]]: List of lists of integer-mapped rolling hashes for each block of each input string.
    """
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

        print(blocks)
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
