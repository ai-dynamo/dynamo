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


from dynamo._core import TokenBlockSequence


def test_token_block_sequence_construct():
    tokens = [1, 2, 3, 4, 5]

    sequence = TokenBlockSequence(tokens, 4)

    assert sequence.total_tokens == 5
    assert sequence.salt_hash == 0
    assert sequence.current_tokens() == [5]


def test_token_block_sequence_append():
    tokens = [1, 2, 3, 4, 5]

    sequence = TokenBlockSequence(tokens, 4)

    assert sequence.append(6) is None
    assert sequence.append(7) is None
    assert sequence.append(8) is None
    assert sequence.append(9) == 1

    assert sequence.extend([10]) is None
    assert sequence.extend([11, 12, 13, 14, 15, 16]) == (2, 3)
    assert sequence.blocks()[2].tokens == [9, 10, 11, 12]


def test_token_block_hashes():
    tokens = list(range(17))

    sequence = TokenBlockSequence(tokens, 4)

    assert len(sequence.blocks()) == 4

    assert sequence.blocks()[0].parent_sequence_hash is None
    assert (
        sequence.blocks()[1].parent_sequence_hash == sequence.blocks()[0].sequence_hash
    )

    assert (
        sequence.blocks()[2].parent_sequence_hash == sequence.blocks()[1].sequence_hash
    )
    assert (
        sequence.blocks()[3].parent_sequence_hash == sequence.blocks()[2].sequence_hash
    )
