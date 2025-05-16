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

import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from benchmarks.data_utils.hasher import texts_to_hashes


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


def test_texts_to_hashes_blocks(dummy_tokenizer):
    dum1 = "a b c d"
    dum2 = "e f g h"
    dum3 = "i j k l"

    texts = [dum1, dum1 + " " + dum2, dum1 + " " + dum3, dum2 + " " + dum1]
    expected = [[0], [0, 1], [0, 2], [3, 4]]

    result = texts_to_hashes(dummy_tokenizer, texts, block_size=4)
    assert result == expected, f"Expected {expected}, got {result}"
