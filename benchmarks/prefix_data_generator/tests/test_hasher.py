# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import random

import pytest
from prefix_data_generator.hasher import hashes_to_texts, texts_to_hashes
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@pytest.fixture(scope="module")
def dummy_tokenizer():
    vocab = [chr(i) for i in range(ord("a"), ord("z") + 1)]
    vocab.append("[UNK]")
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}

    tokenizer_model = models.WordLevel(vocab=vocab_dict, unk_token="[UNK]")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="")

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


@pytest.fixture(scope="module")
def deepseek_tokenizer():
    return AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")


def test_texts_to_hashes_blocks(dummy_tokenizer):
    dum1 = "a b c d"
    dum2 = "e f g h"
    dum3 = "i j k l"

    texts = [dum1, dum1 + " " + dum2, dum1 + " " + dum3, dum2 + " " + dum1]
    expected = [[0], [0, 1], [0, 2], [3, 4]]

    result = texts_to_hashes(dummy_tokenizer, texts, block_size=4)
    assert result == expected, f"Expected {expected}, got {result}"


def test_hashes_to_texts_with_deepseek(deepseek_tokenizer):
    """Test hashes_to_texts with deepseek tokenizer using increasing hash IDs globally."""
    # Test parameters
    block_size = 64
    num_entries = 100

    # Generate test data
    hash_ids_list = []
    input_lengths = []
    global_hash_id = 0

    for _ in range(num_entries):
        # Random input length between 1 and 20 times block_size
        input_length = random.randint(block_size, 20 * block_size)
        input_lengths.append(input_length)

        # Calculate number of hash_ids needed (ceil div)
        num_hash_ids = math.ceil(input_length / block_size)
        hash_ids = list(range(global_hash_id, global_hash_id + num_hash_ids))
        hash_ids_list.append(hash_ids)

        global_hash_id += num_hash_ids

    # Call hashes_to_texts
    texts = hashes_to_texts(
        deepseek_tokenizer, hash_ids_list, input_lengths, block_size
    )

    # Retokenize and verify input lengths are preserved
    for i, (text, expected_length) in enumerate(zip(texts, input_lengths)):
        tokens = deepseek_tokenizer(text, add_special_tokens=False)["input_ids"]
        actual_length = len(tokens)
        assert (
            actual_length == expected_length
        ), f"Entry {i}: expected length {expected_length}, got {actual_length}"
